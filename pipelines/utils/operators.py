import base64
from collections.abc import Sequence
from io import BytesIO
import math
import os
import random
import re
import sys
import six
import uuid
import cv2
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession, SessionOptions


class ONNXInferBackend(object):
    def __init__(self, onnx_model, device="cpu", use_fp16=False):
        if not os.path.exists(onnx_model):
            raise OSError(f"{onnx_model} not exists!")

        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = SessionOptions()
        self.predictor = InferenceSession(
            onnx_model, sess_options=sess_options, providers=providers
        )
        if device == "gpu":
            try:
                assert "CUDAExecutionProvider" in self.predictor.get_providers()
            except AssertionError:
                raise AssertionError(  # noqa: B904
                    "The environment for GPU inference is not set properly. "
                )

    def infer(self, input_dict: dict):
        result = self.predictor.run(None, input_dict)
        return result


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays,
    which is greater than a limitation.

    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """  # noqa: E501
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is 
        a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once.
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= prompt_length + 1
        offset_map[i][1] -= prompt_length + 1

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append((offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xFEE0
        if not (0x0021 <= code and code <= 0x7E):
            rs += char
            continue
        rs += chr(code)
    return rs


class DecodeImage(object):
    """decode image"""

    def __init__(
        self,
        img_mode="RGB",
        channel_first=False,
        ignore_orientation=False,
        **kwargs
    ):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data["image"]
        if six.PY2:
            assert (
                type(img) is str and len(img) > 0
            ), "invalid input 'img' in DecodeImage"
        else:
            assert (
                type(img) is bytes and len(img) > 0
            ), "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype="uint8")
        if self.ignore_orientation:
            img = cv2.imdecode(
                img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR
            )
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == "RGB":
            assert img.shape[2] == 3, "invalid shape of image[%s]" % (
                img.shape
            )
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data["image"] = img
        return data


class NormalizeImage(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(
            img, np.ndarray
        ), "invalid input 'img' in NormalizeImage"
        data["image"] = (
            img.astype("float32") * self.scale - self.mean
        ) / self.std
        return data


class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class Pad(object):
    def __init__(self, size=None, size_div=32, **kwargs):
        if size is not None and not isinstance(size, (int, list, tuple)):
            raise TypeError(
                "Type of target_size is invalid. Now is {}".format(type(size))
            )
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.size_div = size_div

    def __call__(self, data):
        img = data["image"]
        img_h, img_w = img.shape[0], img.shape[1]
        if self.size:
            resize_h2, resize_w2 = self.size
            assert (
                img_h < resize_h2 and img_w < resize_w2
            ), "(h, w) of target size should be greater than (img_h, img_w)"
        else:
            resize_h2 = max(
                int(math.ceil(img.shape[0] / self.size_div) * self.size_div),
                self.size_div,
            )
            resize_w2 = max(
                int(math.ceil(img.shape[1] / self.size_div) * self.size_div),
                self.size_div,
            )
        img = cv2.copyMakeBorder(
            img,
            0,
            resize_h2 - img_h,
            0,
            resize_w2 - img_w,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        data["image"] = img
        return data


class Resize(object):
    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    def resize_image(self, img):
        resize_h, resize_w = self.size
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def __call__(self, data):
        img = data["image"]
        if "polys" in data:
            text_polys = data["polys"]

        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        if "polys" in data:
            new_boxes = []
            for box in text_polys:
                new_box = []
                for cord in box:
                    new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
                new_boxes.append(new_box)
            data["polys"] = np.array(new_boxes, dtype=np.float32)
        data["image"] = img_resize
        return data


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.keep_ratio = False
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs and kwargs["limit_side_len"] is not None:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

    def __call__(self, data):
        img = data["image"]
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data["image"] = img
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def image_padding(self, im, value=0):
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception("not support limit type, image ")
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:  # noqa: E722
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]

# =================UIEX=====================
class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + "_" + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class ResizeImage(BaseOperator):
    def __init__(self, target_size=0, interp=1):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            interp (int): the interpolation method
        """
        super(ResizeImage, self).__init__()
        self.interp = int(interp)
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".format(type(target_size))
            )
        self.target_size = target_size

    def __call__(self, sample, context=None, save_real_img=False):
        """Resize the image numpy."""
        im = sample["image"]
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError("{}: min size of image is 0".format(self))

        resize_w = selected_size
        resize_h = selected_size

        im = Image.fromarray(im.astype("uint8"))
        im = im.resize((int(resize_w), int(resize_h)), self.interp)
        sample["image"] = np.array(im)
        return sample


class Permute(BaseOperator):
    def __init__(self, to_bgr=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert "image" in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith("image"):
                    im = sample[k]
                    im = np.swapaxes(im, 1, 2)
                    im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class UIEXNormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1], is_channel_first=True, is_scale=False):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
            channel_first (bool): confirm whether to change channel
        """
        super(UIEXNormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_channel_first = is_channel_first
        self.is_scale = is_scale
        from functools import reduce

        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError("{}: std is invalid!".format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                if k.startswith("image"):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


def pad_image_data(image_data):
    resize_func = ResizeImage(target_size=224, interp=1)
    norm_func = UIEXNormalizeImage(is_channel_first=False, mean=[123.675, 116.280, 103.530], std=[58.395, 57.120, 57.375])
    permute_func = Permute(to_bgr=False)
    if not image_data:
        image = np.zeros([3, 224, 224])
        return image
    # decode image
    data = np.frombuffer(bytearray(image_data), dtype="uint8")
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    sample = {"image": image}
    # resize image
    sample = resize_func(sample)
    # norm image
    sample = norm_func(sample)
    # permute
    sample = permute_func(sample)
    return sample["image"]


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def np2base64(image_np):
    img = Image.fromarray(image_np)
    base64_str = pil2base64(img)
    return base64_str


def pil2base64(image, image_type=None, size=False):
    if not image_type:
        image_type = "JPEG"
    img_buffer = BytesIO()
    image.save(img_buffer, format=image_type)

    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)

    base64_string = base64_str.decode("utf-8")

    if size:
        return base64_string, image.size
    else:
        return base64_string