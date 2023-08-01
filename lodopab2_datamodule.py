#pyright: reportGeneralTypeIssues=false
import typing

import omegaconf
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision

from fixed_noise_dataset import FixedNoiseDataset, Noise
from lodopab2_dataset import LoDoPaB2Dataset


class LoDoPaB2DataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig, noise: Noise) -> None:
        super().__init__()
        self.config = config
        self.noise = noise

    def train_dataloader(self) -> torch.utils.data.DataLoader[typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
        training_dataset = LoDoPaB2Dataset("/data/datasets/", LoDoPaB2Dataset.Subset.TEST, target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config.img_size, antialias=True),
            torchvision.transforms.CenterCrop(self.config.img_size)
        ]))
        training_dataset = FixedNoiseDataset(training_dataset, noise=self.noise, append_clean=True, append_noise=True)
        return torch.utils.data.DataLoader(training_dataset, drop_last=self.config.drop_last_training_batch, batch_size=self.config.training_batch_size, shuffle=self.config.shuffle_training_data, num_workers=self.config.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader[typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
        validation_dataset = LoDoPaB2Dataset("/data/datasets/", LoDoPaB2Dataset.Subset.TEST, target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config.img_size, antialias=True),
            torchvision.transforms.CenterCrop(self.config.img_size)
        ]))
        validation_dataset = FixedNoiseDataset(validation_dataset, noise=self.noise, append_clean=True, append_noise=True)
        return torch.utils.data.DataLoader(validation_dataset, drop_last=self.config.drop_last_validation_batch, batch_size=self.config.validation_batch_size, shuffle=self.config.shuffle_validation_data, num_workers=self.config.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader[typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
        test_dataset = LoDoPaB2Dataset("/data/datasets/", LoDoPaB2Dataset.Subset.TEST, target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config.img_size, antialias=True),
            torchvision.transforms.CenterCrop(self.config.img_size)
        ]))
        test_dataset = FixedNoiseDataset(test_dataset, noise=self.noise, append_clean=True, append_noise=True)
        return torch.utils.data.DataLoader(test_dataset, drop_last=self.config.drop_last_test_batch, batch_size=self.config.test_batch_size, shuffle=self.config.shuffle_test_data, num_workers=self.config.num_workers)