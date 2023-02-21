import cv2
import numpy as np
import paddle
import os

from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

from annotator.util import annotator_ckpts_path
from cldm.model import load

remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"


class MLSDdetector:
    def __init__(self):
        model_path = os.path.join(annotator_ckpts_path, "mlsd_large_512_fp32.pth")
        if not os.path.exists(model_path):
            from paddlenlp.utils.downloader import get_path_from_url_with_filelock
            get_path_from_url_with_filelock(remote_model_path, root_dir=annotator_ckpts_path)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(load(model_path), strict=True)
        self.model = model.cuda().eval()

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with paddle.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass
        return img_output[:, :, 0]
