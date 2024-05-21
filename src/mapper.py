
import torch
from torch import nn
import torchvision


class LatentMapper(nn.Module):
    """
    Отображает grayscale sketch лица в пространство W+
    """
    def __init__(self,
                 num_latents: int = 14,
                 w_dim: int = 512
                 ):
        super().__init__()
        self.num_latents = num_latents
        self.w_dim = w_dim
        resnet = torchvision.models.resnet50(pretrained=True)
        self.mapper = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            *list(resnet.children())[1 : -1],
            nn.Flatten(),
            nn.Linear(2048, self.num_latents * self.w_dim)
        )

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        :param sketch: Bx1xHxW
        :return: W+ vectors, Bx14x512
        """
        wplus = self.mapper(sketch)
        wplus = wplus.view((-1, self.num_latents, self.w_dim))
        return wplus
