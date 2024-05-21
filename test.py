
from pathlib import Path
import datetime

import warnings

import numpy as np

warnings.filterwarnings('ignore')

from omegaconf import OmegaConf
import click
import torch
import cv2 as cv
from tqdm import tqdm

from src.app_logger import logger
from src.utils import download_stylegan2_ffhq_256, tensor2img
from src.sketcher import ImageSketch
from src.mapper import LatentMapper


def test(config: dict, image_folder: Path):
    pretrained_mapper_path = Path(config['mapper_pretrained_weights'])
    sketch_model_path = Path(config['sketch_weights'])
    sg_weights_path = Path(config['sg_weight_path'])

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    output_folder = Path(config['output_folder'])
    output_folder = output_folder.joinpath(now)
    output_folder.mkdir(exist_ok=True, parents=True)

    logger.info("Load StyleGAN model.")
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    stylegan_generator = download_stylegan2_ffhq_256(sg_weights_path, device)
    stylegan_generator.eval()

    logger.info("Load Sketch model.")
    assert sketch_model_path.is_file(), 'Sketch model file is not found.'
    sketch_model = ImageSketch(sketch_model_path, device, config['image_size'])

    logger.info("Load mapper model.")
    assert pretrained_mapper_path.is_file(), 'Mapper model file is not found.'
    mapper_model = LatentMapper()
    mapper_model.to(device)
    mapper_model.load_state_dict(torch.load(pretrained_mapper_path))
    mapper_model.eval()

    logger.info("Start testing.")
    for img_file in tqdm(list(image_folder.glob('*.*'))):
        image = cv.imread(str(img_file))
        sketch = sketch_model(image)
        sketch_tns = torch.from_numpy(sketch[None, None, :, :]).float().to(device)
        wplus = mapper_model(sketch_tns)
        face = stylegan_generator.synthesis(wplus, noise_mode='const')
        image_pred = tensor2img(face)[0]
        image_stack = np.hstack((
            cv.resize(image, (config['image_size'], config['image_size'])),
            image_pred
        ))
        out_file = output_folder.joinpath(img_file.name)
        cv.imwrite(str(out_file), image_stack)


@click.command()
@click.option('--config_file', '-cf', type=click.Path(path_type=Path, exists=True), required=True,
              default = './configs/config.yaml', help = 'Config file.')
@click.option('--image_folder', '-if', type=click.Path(path_type=Path, exists=True), required=True,
              default = './test_img', help = 'Folder with test images for testing.')
def main(config_file: Path, image_folder: Path):
    config = OmegaConf.load(config_file)
    test(config, image_folder)

if __name__ == '__main__':
    main()
