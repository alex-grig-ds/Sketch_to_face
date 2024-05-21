
from pathlib import Path
import datetime

import warnings
warnings.filterwarnings('ignore')

from omegaconf import OmegaConf
import click
import torch
from torch import nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from tqdm import tqdm
import lpips

from src.app_logger import logger
from src.utils import download_stylegan2_ffhq_256
from src.sketcher import ImageSketch
from src.mapper import LatentMapper
from src.datasets import FaceDataSet


def train_loop(config: dict):
    pretrained_mapper_path = Path(config['mapper_pretrained_weights'])
    sketch_model_path = Path(config['sketch_weights'])
    sg_weights_path = Path(config['sg_weight_path'])

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    checkpoint_save_dir = Path(config['checkpoint_folder'])
    checkpoint_save_dir = checkpoint_save_dir.joinpath(now)
    checkpoint_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Load StyleGAN model.")
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    stylegan_generator = download_stylegan2_ffhq_256(sg_weights_path, device)
    stylegan_generator.eval()

    logger.info("Load Sketch model.")
    sketch_model = ImageSketch(sketch_model_path, device)

    logger.info("Load mapper model.")
    mapper_model = LatentMapper()
    mapper_model.to(device)
    if config['continue_training']:
        mapper_model.load_state_dict(torch.load(pretrained_mapper_path))
    mapper_model.train()

    logger.info("Prepare datasets.")
    train_data = FaceDataSet(stylegan_generator, device, random_epoch=True, dataset_size=config['train_dataset_size'], seed=123)
    test_data = FaceDataSet(stylegan_generator, device, random_epoch=False, dataset_size=config['valid_dataset_size'], seed=345)
    train_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'])
    test_loader = DataLoader(dataset=test_data, batch_size=config['batch_size'])

    logger.info(f"Start training with {device}.")
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    optimizer = optim.Adam(mapper_model.parameters(), lr=config['start_learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
    mse_loss_fn = nn.MSELoss()
    lpips_loss_fn = lpips.LPIPS(net="vgg").eval().to(device)
    for epoch in range(config['num_epochs']):
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Start {epoch} epoch. LR: {curr_lr:.5f}")
        train_loss_epoch = 0
        cos_sim_loss_epoch = 0
        mse_loss_epoch = 0
        lpips_loss_epoch = 0
        for batch in tqdm(train_loader, desc=f"Train, epoch {epoch}"):
            optimizer.zero_grad()
            faces, wplus = batch[0].to(device), batch[1].to(device)
            sketches = sketch_model.get_sketch_tensor(faces)
            wplus_pred = mapper_model(sketches)
            faces_pred = stylegan_generator.synthesis(wplus_pred, noise_mode='const')

            cos_similar_loss = (1.0 - cosine_similarity(wplus, wplus_pred)).mean()
            mse_loss = mse_loss_fn(faces_pred, faces)
            lpips_loss = lpips_loss_fn(faces_pred, faces).mean()
            loss = 0.4 * cos_similar_loss + 0.4 * lpips_loss + 0.2 * mse_loss
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            cos_sim_loss_epoch += cos_similar_loss.item()
            mse_loss_epoch += mse_loss.item()
            lpips_loss_epoch += lpips_loss.item()
        logger.info(f"Train loss: {train_loss_epoch / len(train_loader):.5f}. Cossim loss: "
                    f"{cos_sim_loss_epoch / len(train_loader):.5f}. MSE loss: {mse_loss_epoch / len(train_loader):.5f}. "
                    f"LPIPS loss: {lpips_loss_epoch / len(train_loader):.5f}.")

        valid_lpips_loss = 0
        for batch in tqdm(test_loader, desc=f"Validation, epoch {epoch}"):
            with torch.no_grad():
                faces, wplus = batch[0].to(device), batch[1].to(device)
                sketches = sketch_model.get_sketch_tensor(faces)
                wplus_pred = mapper_model(sketches)
                faces_pred = stylegan_generator.synthesis(wplus_pred, noise_mode='const')
                lpips_loss = lpips_loss_fn(faces_pred, faces).mean()
                valid_lpips_loss += lpips_loss.item()
        epoch_valid_loss = valid_lpips_loss / len(test_loader)
        logger.info(f"Validation LPIPS loss: {epoch_valid_loss:.5f}.")
        epoch_dir = checkpoint_save_dir.joinpath(str(epoch).rjust(5, '0'))
        epoch_dir.mkdir(exist_ok=True, parents=True)
        model_path = epoch_dir.joinpath(f"mapper_weights_{epoch_valid_loss:.5f}.pt")
        torch.save(mapper_model.state_dict(), model_path)
        logger.info(f"Save model to: {model_path}")
        faces_stack = torch.cat((faces, faces_pred))
        torchvision.utils.save_image(faces_stack, epoch_dir.joinpath(f"batch_example.png"))
        scheduler.step()

@click.command()
@click.option('--config_file', '-cf', type=click.Path(path_type=Path, exists=True), required=True,
              default = './configs/config.yaml', help = 'Config file.')
def main(config_file: Path):
    config = OmegaConf.load(config_file)
    train_loop(config)

if __name__ == '__main__':
    main()
