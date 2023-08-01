import enum
import pathlib
import typing

import torch.utils.data


class Ellipses2Dataset(torch.utils.data.Dataset[typing.Tuple[torch.Tensor,torch.Tensor]]):
    class Subset(enum.Enum):
        TRAIN = "train"
        VAL = "val"
        TEST = "test"

    def __init__(self, 
                 path: str|pathlib.Path, 
                 subset: Subset,
                 transform: typing.Callable[[torch.Tensor],torch.Tensor]=lambda x: x, 
                 target_transform: typing.Callable[[torch.Tensor],torch.Tensor]=lambda x: x) -> None:
        super().__init__()
        self.__transform = transform
        self.__target_transform = target_transform
        self.__subset = subset
        self.__path = pathlib.Path(path, "Ellipses")

    def __len__(self) -> int:
        return {
            Ellipses2Dataset.Subset.TRAIN: 10000,
            Ellipses2Dataset.Subset.VAL: 1000,
            Ellipses2Dataset.Subset.TEST: 1000,
        }[self.__subset]

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor,torch.Tensor]:
        observation = torch.load(self.__path.joinpath(self.__subset.value, f"sinogram_{idx}.pt"))
        ground_truth = torch.load(self.__path.joinpath(self.__subset.value, f"ground_truth_{idx}.pt"))
        return self.__transform(observation), self.__target_transform(ground_truth)