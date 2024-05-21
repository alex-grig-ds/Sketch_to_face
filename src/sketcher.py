
from pathlib import Path

import cv2 as cv
import torch
import numpy as np
import kornia

from src.DexiNed.model import DexiNed


class ImageSketch:
    def __init__(self,
                 sketch_weights: Path,
                 device: torch.device,
                 image_size: int=256
    ):
        assert sketch_weights.is_file(), "Sketch model file not found."
        self.device = device
        self.model = DexiNed().to(self.device)
        self.model.load_state_dict(torch.load(sketch_weights, map_location=self.device))
        self.model.eval()
        self.model_input_size = image_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_sketch = self.get_sketch(image)
        return image_sketch

    def get_sketch(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: cv.BGR image
        :return: cv.GRAYSCALE image
        """
        image_ = cv.resize(image, (self.model_input_size, self.model_input_size))
        img_tns = torch.from_numpy(image_.transpose((2, 0, 1))).float().to(self.device)
        img_tns = img_tns[None, :, :, :]
        pred = self.get_sketch_tensor(img_tns)
        image_sketch = kornia.utils.tensor_to_image(pred)
        image_sketch = image_sketch.astype(np.uint8)
        return image_sketch

    def get_sketch_tensor(self,  image: torch.Tensor) -> torch.Tensor:
        """
        :param image: Bx3xHxW
        :return: BxHxW
        """
        pred = self.model(image)
        pred = torch.cat(pred, dim=1)
        pred = torch.sigmoid(pred)
        pred = pred.mean(dim=1)
        pred = (255.0 * (1.0 - pred))
        pred = pred[:, None, :, :]
        return pred

