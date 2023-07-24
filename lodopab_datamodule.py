#pyright: reportGeneralTypeIssues=false
import typing

import omegaconf
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision

from lodopab_dataset import LoDoPaBDataset
from feature_mod_dataset import FeatureModDataset


class LoDoPaBDataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> torch.utils.data.DataLoader[typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
        TTT = torchvision.transforms.Resize((500,257))
        training_dataset = LoDoPaBDataset("/data/datasets/", LoDoPaBDataset.Subset.TEST, extracted=True, transform=TTT, target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config.img_size, antialias=True),
            torchvision.transforms.CenterCrop(self.config.img_size)
        ]))
        training_dataset = FeatureModDataset(training_dataset, append=(lambda x: torch.zeros_like(x[0]), lambda x: torch.zeros_like(x[0])), new_order=(0,1,3,4))
        return torch.utils.data.DataLoader(training_dataset, drop_last=self.config.drop_last_training_batch, batch_size=self.config.training_batch_size, shuffle=self.config.shuffle_training_data, num_workers=self.config.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader[typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
        TTT = torchvision.transforms.Resize((500,257))
        validation_dataset = LoDoPaBDataset("/data/datasets/", LoDoPaBDataset.Subset.TEST, extracted=True, transform=TTT, target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config.img_size, antialias=True),
            torchvision.transforms.CenterCrop(self.config.img_size)
        ]))
        validation_dataset = FeatureModDataset(validation_dataset, append=(lambda x: torch.zeros_like(x[0]), lambda x: torch.zeros_like(x[0])), new_order=(0,1,3,4))
        return torch.utils.data.DataLoader(validation_dataset, drop_last=self.config.drop_last_validation_batch, batch_size=self.config.validation_batch_size, shuffle=self.config.shuffle_validation_data, num_workers=self.config.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader[typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
        TTT = torchvision.transforms.Resize((500,257))
        test_dataset = LoDoPaBDataset("/data/datasets/", LoDoPaBDataset.Subset.TEST, extracted=True, transform=TTT, target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config.img_size, antialias=True),
            torchvision.transforms.CenterCrop(self.config.img_size)
        ]))
        test_dataset = FeatureModDataset(test_dataset, append=(lambda x: torch.zeros_like(x[0]), lambda x: torch.zeros_like(x[0])), new_order=(0,1,3,4))
        return torch.utils.data.DataLoader(test_dataset, drop_last=self.config.drop_last_test_batch, batch_size=self.config.test_batch_size, shuffle=self.config.shuffle_test_data, num_workers=self.config.num_workers)