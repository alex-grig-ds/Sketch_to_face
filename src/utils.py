
import os
from pathlib import Path
import pickle
import sys

import torch
import numpy as np

from src.app_logger import logger

STYLE_GAN_PATH = './src/stylegan2_ada'
sys.path.append(STYLE_GAN_PATH)
from src.stylegan2_ada.torch_utils import persistence


def download_stylegan2_ffhq_256(weights_path: Path, device):
    if not weights_path.is_file():
        weights_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info('stylegan2-ffhq-256 model not found, downloading...')
        os.system("wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan2/1/files?redirect=true&path=stylegan2-ffhq-256x256.pkl' -O stylegan2-ffhq-256x256.pkl")
        os.rename('stylegan2-ffhq-256x256.pkl', str(weights_path))
        logger.info('done')
        assert weights_path.is_file(), "File not saved"

    with open(str(weights_path), 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)

    return G

def tensor2img(tensor: torch.Tensor) -> np.ndarray:
    """
    Конвертирует один тензор формата BCHW в numpy изображения в формате BHWC
    Выходные изображения переводятся в uint8 0-255
    """

    return (127.5 * (tensor + 1.0)).clamp(0, 255).permute(0, 2, 3, 1).cpu().to(torch.uint8).numpy()

