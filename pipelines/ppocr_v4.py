import os
import yaml
import onnxruntime as ort
from collections import namedtuple

from .utils.text_system import TextSystem



class PPOCRv4(object):
    """PaddlePaddle OCR-v4"""

    def __init__(self, model_config, device='CPU') -> None:
        # Load config
        if isinstance(model_config, str):
            if not os.path.isfile(model_config):
                raise FileNotFoundError(f"[ERROR] Config file not found: {model_config}")
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise ValueError(f"[ERROR] Unknown config type: {type}")

        self.det_net = self.load_model("det_model_path", device)
        self.rec_net = self.load_model("rec_model_path", device)
        
        self.use_angle_cls = self.config["use_angle_cls"]
        if self.use_angle_cls:
            self.cls_net = self.load_model("cls_model_path", device)

    def load_model(self, model_name, device='CPU'):
        model_type = model_name.rsplit("_", 1)[0]
        model_path = self.config[model_type][model_name]
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )
        self.providers = ["CPUExecutionProvider"]

        if device == "GPU":
            self.providers = ["CUDAExecutionProvider"]
        net = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )
        return net

    def parse_args(self):
        args = {}
        for key, value in self.config.items():
            if isinstance(value, dict):
                args.update(value)
            else:
                args[key] = value
        args['det_model'] = self.det_net
        args['rec_model'] = self.rec_net
        if self.use_angle_cls:
            args['cls_model'] = self.cls_net
        return namedtuple('Args', args.keys())(**args)

    def predict(self, image):
        """
        Predict shapes from image
        """
        args = self.parse_args()
        text_sys = TextSystem(args)
        dt_boxes, rec_res = text_sys(image)
        resMsg = [[points.tolist(), content]for points, content in zip(dt_boxes, rec_res)]
        if len(resMsg) == 0:
            return []
        return resMsg

    def unload(self):
        del self.det_net
        del self.rec_net
        if self.use_angle_cls:
            del self.cls_net
