import os
import cv2
import copy
import math
import numpy as np
from .operators import *  # noqa: F403
from .db_postprocess import *  # noqa: F403
from .rec_postprocess import *  # noqa: F403
from .cls_postprocess import *  # noqa: F403
from numpy.linalg import norm
from shapely.geometry import Polygon
from .seal_det_warp import AutoRectifier

__all__ = ["TextSystem"]


class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.use_onnx = args.use_onnx
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": args.det_limit_side_len,
                    "limit_type": args.det_limit_type,
                    "resize_long": args.det_resize_long,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params["name"] = "DBPostProcess"
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(args, "det")

        if self.use_onnx:
            img_h, img_w = self.input_tensor.shape[2:]
            if isinstance(img_h, str) or isinstance(img_w, str):
                pass
            elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
                pre_process_list[0] = {
                    "DetResizeForTest": {"image_shape": [img_h, img_w]}
                }

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)

        if len(dt_boxes_new) > 0:
            max_points = max(len(polygon) for polygon in dt_boxes_new)
            dt_boxes_new = [
                self.pad_polygons(polygon, max_points) for polygon in dt_boxes_new
            ]

        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def pad_polygons(self, polygon, max_points):
        padding_size = max_points - len(polygon)
        if padding_size == 0:
            return polygon
        last_point = polygon[-1]
        padding = np.repeat([last_point], padding_size, axis=0)
        return np.vstack([polygon, padding])

    @staticmethod
    def transform(data, ops=None):
        """transform"""
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def __call__(self, img):
        ori_im = img.copy()
        data = {"image": img}

        data = self.transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)

        preds = {}
        preds["maps"] = outputs[0]
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        if self.args.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        postprocess_params = {
            "name": "CTCLabelDecode",
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(args, "rec")
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if isinstance(w, str):
                pass
            elif w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.use_onnx:
                input_dict = {}
                input_dict[self.input_tensor.name] = norm_img_batch
                outputs = self.predictor.run(self.output_tensors, input_dict)
                preds = outputs[0]

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res


class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        postprocess_params = {
            "name": "ClsPostProcess",
            "label_list": args.label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            _,
        ) = create_predictor(args, "cls")
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [["", 0.0]] * img_num
        batch_num = self.cls_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            if self.use_onnx:
                input_dict = {}
                input_dict[self.input_tensor.name] = norm_img_batch
                outputs = self.predictor.run(self.output_tensors, input_dict)
                prob_out = outputs[0]
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()
                prob_out = self.output_tensors[0].copy_to_cpu()
                self.predictor.try_shrink_memory()
            cls_result = self.postprocess_op(prob_out)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return img_list, cls_res


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = TextDetector(args)
        self.text_recognizer = TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = TextClassifier(args)
        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        if img is None:
            # logger.debug("no valid image provided")
            return None, None

        ori_im = img.copy()
        dt_boxes = self.text_detector(img)

        if dt_boxes is None:
            return None, None
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            elif self.args.det_box_type == "poly":
                img_crop = get_poly_rect_crop(ori_im.copy(), tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list = self.text_classifier(img_crop_list)

        rec_res = self.text_recognizer(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


def build_post_process(config, global_config=None):
    support_dict = ["DBPostProcess", "CTCLabelDecode", "ClsPostProcess"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def create_predictor(args, mode):
    if mode == "det":
        sess = args.det_model
    elif mode == "cls":
        sess = args.cls_model
    elif mode == "rec":
        sess = args.rec_model

    return sess, sess.get_inputs()[0], None, None


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_poly_rect_crop(img, points):
    """
    修改该函数，实现使用polygon，对不规则、弯曲文本的矫正以及crop
    args： img: 图片 ndarrary格式
    points： polygon格式的多点坐标 N*2 shape， ndarray格式
    return： 矫正后的图片 ndarray格式
    """
    points = np.array(points).astype(np.int32).reshape(-1, 2)
    temp_crop_img, temp_box = get_minarea_rect(img, points)

    # 计算最小外接矩形与polygon的IoU
    def get_union(pD, pG):
        return Polygon(pD).union(Polygon(pG)).area

    def get_intersection_over_union(pD, pG):
        return get_intersection(pD, pG) / (get_union(pD, pG) + 1e-10)

    def get_intersection(pD, pG):
        return Polygon(pD).intersection(Polygon(pG)).area

    cal_IoU = get_intersection_over_union(points, temp_box)

    if cal_IoU >= 0.7:
        points = sample_points_on_bbox_bp(points, 31)
        return temp_crop_img

    points_sample = sample_points_on_bbox(points)
    points_sample = points_sample.astype(np.int32)
    head_edge, tail_edge, top_line, bot_line = reorder_poly_edge(points_sample)

    resample_top_line = sample_points_on_bbox_bp(top_line, 15)
    resample_bot_line = sample_points_on_bbox_bp(bot_line, 15)

    sideline_mean_shift = np.mean(resample_top_line, axis=0) - np.mean(
        resample_bot_line, axis=0
    )
    if sideline_mean_shift[1] > 0:
        resample_bot_line, resample_top_line = resample_top_line, resample_bot_line
    rectifier = AutoRectifier()
    new_points = np.concatenate([resample_top_line, resample_bot_line])
    new_points_list = list(new_points.astype(np.float32).reshape(1, -1).tolist())

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    img_crop, image = rectifier.run(img, new_points_list, mode="homography")
    return np.array(img_crop[0], dtype=np.uint8)


def get_minarea_rect(img, points):
    bounding_box = cv2.minAreaRect(points)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img, box


def sample_points_on_bbox_bp(line, n=50):
    """Resample n points on a line.

    Args:
        line (ndarray): The points composing a line.
        n (int): The resampled points number.

    Returns:
        resampled_line (ndarray): The points composing the resampled line.
    """
    from numpy.linalg import norm

    # 断言检查输入参数的有效性
    assert line.ndim == 2
    assert line.shape[0] >= 2
    assert line.shape[1] == 2
    assert isinstance(n, int)
    assert n > 0

    length_list = [norm(line[i + 1] - line[i]) for i in range(len(line) - 1)]
    total_length = sum(length_list)
    length_cumsum = np.cumsum([0.0] + length_list)
    delta_length = total_length / (float(n) + 1e-8)
    current_edge_ind = 0
    resampled_line = [line[0]]

    for i in range(1, n):
        current_line_len = i * delta_length
        while (
            current_edge_ind + 1 < len(length_cumsum)
            and current_line_len >= length_cumsum[current_edge_ind + 1]
        ):
            current_edge_ind += 1
        current_edge_end_shift = current_line_len - length_cumsum[current_edge_ind]
        if current_edge_ind >= len(length_list):
            break
        end_shift_ratio = current_edge_end_shift / length_list[current_edge_ind]
        current_point = (
            line[current_edge_ind]
            + (line[current_edge_ind + 1] - line[current_edge_ind])
            * end_shift_ratio
        )
        resampled_line.append(current_point)
    resampled_line.append(line[-1])
    resampled_line = np.array(resampled_line)
    return resampled_line


def sample_points_on_bbox(line, n=50):
    """Resample n points on a line.

    Args:
        line (ndarray): The points composing a line.
        n (int): The resampled points number.

    Returns:
        resampled_line (ndarray): The points composing the resampled line.
    """
    assert line.ndim == 2
    assert line.shape[0] >= 2
    assert line.shape[1] == 2
    assert isinstance(n, int)
    assert n > 0

    length_list = [norm(line[i + 1] - line[i]) for i in range(len(line) - 1)]
    total_length = sum(length_list)
    mean_length = total_length / (len(length_list) + 1e-8)
    group = [[0]]
    for i in range(len(length_list)):
        point_id = i + 1
        if length_list[i] < 0.9 * mean_length:
            for g in group:
                if i in g:
                    g.append(point_id)
                    break
        else:
            g = [point_id]
            group.append(g)

    top_tail_len = norm(line[0] - line[-1])
    if top_tail_len < 0.9 * mean_length:
        group[0].extend(g)
        group.remove(g)
    mean_positions = []
    for indices in group:
        x_sum = 0
        y_sum = 0
        for index in indices:
            x, y = line[index]
            x_sum += x
            y_sum += y
        num_points = len(indices)
        mean_x = x_sum / num_points
        mean_y = y_sum / num_points
        mean_positions.append((mean_x, mean_y))
    resampled_line = np.array(mean_positions)
    return resampled_line


def reorder_poly_edge(points):
    """Get the respective points composing head edge, tail edge, top
    sideline and bottom sideline.

    Args:
        points (ndarray): The points composing a text polygon.

    Returns:
        head_edge (ndarray): The two points composing the head edge of text
            polygon.
        tail_edge (ndarray): The two points composing the tail edge of text
            polygon.
        top_sideline (ndarray): The points composing top curved sideline of
            text polygon.
        bot_sideline (ndarray): The points composing bottom curved sideline
            of text polygon.
    """

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2

    orientation_thr = 2.0  # 一个经验超参数

    head_inds, tail_inds = find_head_tail(points, orientation_thr)
    head_edge, tail_edge = points[head_inds], points[tail_inds]

    pad_points = np.vstack([points, points])
    if tail_inds[1] < 1:
        tail_inds[1] = len(points)
    sideline1 = pad_points[head_inds[1] : tail_inds[1]]
    sideline2 = pad_points[tail_inds[1] : (head_inds[1] + len(points))]
    return head_edge, tail_edge, sideline1, sideline2

def find_head_tail(points, orientation_thr):
    """Find the head edge and tail edge of a text polygon.

    Args:
        points (ndarray): The points composing a text polygon.
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle.

    Returns:
        head_inds (list): The indexes of two points composing head edge.
        tail_inds (list): The indexes of two points composing tail edge.
    """

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    assert isinstance(orientation_thr, float)

    if len(points) > 4:
        pad_points = np.vstack([points, points[0]])
        edge_vec = pad_points[1:] - pad_points[:-1]

        theta_sum = []
        adjacent_vec_theta = []
        for i, edge_vec1 in enumerate(edge_vec):
            adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
            adjacent_edge_vec = edge_vec[adjacent_ind]
            temp_theta_sum = np.sum(vector_angle(edge_vec1, adjacent_edge_vec))
            temp_adjacent_theta = vector_angle(
                adjacent_edge_vec[0], adjacent_edge_vec[1]
            )
            theta_sum.append(temp_theta_sum)
            adjacent_vec_theta.append(temp_adjacent_theta)
        theta_sum_score = np.array(theta_sum) / np.pi
        adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
        poly_center = np.mean(points, axis=0)
        edge_dist = np.maximum(
            norm(pad_points[1:] - poly_center, axis=-1),
            norm(pad_points[:-1] - poly_center, axis=-1),
        )
        dist_score = edge_dist / np.max(edge_dist)
        position_score = np.zeros(len(edge_vec))
        score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
        score += 0.35 * dist_score
        if len(points) % 2 == 0:
            position_score[(len(score) // 2 - 1)] += 1
            position_score[-1] += 1
        score += 0.1 * position_score
        pad_score = np.concatenate([score, score])
        score_matrix = np.zeros((len(score), len(score) - 3))
        x = np.arange(len(score) - 3) / float(len(score) - 4)
        gaussian = (
            1.0
            / (np.sqrt(2.0 * np.pi) * 0.5)
            * np.exp(-np.power((x - 0.5) / 0.5, 2.0) / 2)
        )
        gaussian = gaussian / np.max(gaussian)
        for i in range(len(score)):
            score_matrix[i, :] = (
                score[i]
                + pad_score[(i + 2) : (i + len(score) - 1)] * gaussian * 0.3
            )

        head_start, tail_increment = np.unravel_index(
            score_matrix.argmax(), score_matrix.shape
        )
        tail_start = (head_start + tail_increment + 2) % len(points)
        head_end = (head_start + 1) % len(points)
        tail_end = (tail_start + 1) % len(points)

        if head_end > tail_end:
            head_start, tail_start = tail_start, head_start
            head_end, tail_end = tail_end, head_end
        head_inds = [head_start, head_end]
        tail_inds = [tail_start, tail_end]
    else:
        if vector_slope(points[1] - points[0]) + vector_slope(
            points[3] - points[2]
        ) < vector_slope(points[2] - points[1]) + vector_slope(
            points[0] - points[3]
        ):
            horizontal_edge_inds = [[0, 1], [2, 3]]
            vertical_edge_inds = [[3, 0], [1, 2]]
        else:
            horizontal_edge_inds = [[3, 0], [1, 2]]
            vertical_edge_inds = [[0, 1], [2, 3]]

        vertical_len_sum = norm(
            points[vertical_edge_inds[0][0]] - points[vertical_edge_inds[0][1]]
        ) + norm(
            points[vertical_edge_inds[1][0]] - points[vertical_edge_inds[1][1]]
        )
        horizontal_len_sum = norm(
            points[horizontal_edge_inds[0][0]] - points[horizontal_edge_inds[0][1]]
        ) + norm(
            points[horizontal_edge_inds[1][0]] - points[horizontal_edge_inds[1][1]]
        )

        if vertical_len_sum > horizontal_len_sum * orientation_thr:
            head_inds = horizontal_edge_inds[0]
            tail_inds = horizontal_edge_inds[1]
        else:
            head_inds = vertical_edge_inds[0]
            tail_inds = vertical_edge_inds[1]

    return head_inds, tail_inds

def vector_angle(vec1, vec2):
    if vec1.ndim > 1:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
    if vec2.ndim > 1:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
    return np.arccos(np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

def vector_slope(vec):
    assert len(vec) == 2
    return abs(vec[1] / (vec[0] + 1e-8))