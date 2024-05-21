
import torch
from torch import nn
from torch.utils.data import Dataset


class FaceDataSet(Dataset):
    """
    Возвращает пары: тензор изображения лица, W+ вектор
    """
    def __init__(self,
                 stylegan_generator: nn.Module,
                 device: torch.device,
                 seed: int,
                 z_dim: int = 512,
                 dataset_size: int = 300,
                 random_epoch: bool = False,  # Если True, то каждую эпоху генерируются новые данные
    ):
        super().__init__()
        self.stylegan_generator = stylegan_generator
        self.device = device
        self.z_dim = z_dim
        self.dataset_size = dataset_size
        self.random_epoch = random_epoch
        self.seed = seed
        self.z_generator = torch.Generator(self.device).manual_seed(self.seed)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        if index == 0 and not self.random_epoch:
            self.z_generator = torch.Generator(self.device).manual_seed(self.seed)
        if index > self.dataset_size:
            raise StopIteration
        z = torch.randn([1, self.z_dim], generator=self.z_generator, device=self.device)
        with torch.no_grad():
            wplus = self.stylegan_generator.mapping(z, None, truncation_psi=0.5, update_emas=False)
            image = self.stylegan_generator.synthesis(wplus, noise_mode='const')
        wplus = wplus.squeeze()
        image = image.squeeze()
        return image, wplus

