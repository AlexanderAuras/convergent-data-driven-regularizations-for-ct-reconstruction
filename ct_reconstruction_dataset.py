import typing

import torch
import torch.utils.data

import radon


class CTReconstructionDataset(torch.utils.data.Dataset[typing.Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: torch.utils.data.Dataset[typing.Tuple[torch.Tensor, ...]], gt_idx: int = 0, angles: typing.Union[torch.Tensor,None] = None, positions: typing.Union[torch.Tensor,None] = None) -> None:
        super().__init__()
        self.__dataset = dataset
        self.__gt_idx = gt_idx
        self.__angles = angles
        self.__positions = positions

    def __len__(self) -> int:
        return len(typing.cast(typing.Sized, self.__dataset))
    
    def __getitem__(self, i: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        sample = self.__dataset[i]
        ground_truth = sample[self.__gt_idx]
        #sinogram = radon.radon_forward(ground_truth.contiguous().unsqueeze(0).to("cuda"), self.__angles, self.__positions)[0].to("cpu")
        sinogram = radon.radon_forward(ground_truth.contiguous().unsqueeze(0), self.__angles, self.__positions)[0]
        return (sinogram, ground_truth)