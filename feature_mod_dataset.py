import typing

import torch
import torch.utils.data


class FeatureModDataset(torch.utils.data.Dataset[typing.Tuple[torch.Tensor,...]]):
    def __init__(self, dataset: torch.utils.data.Dataset[typing.Tuple[torch.Tensor, ...]], append: typing.Union[typing.Sequence[typing.Callable[[typing.Tuple[torch.Tensor,...]],torch.Tensor]],None] = None, new_order: typing.Union[typing.Sequence[int],None] = None) -> None:
        super().__init__()
        self.__dataset = dataset
        self.__new_order = new_order
        self.__append_factories = append

    def __len__(self) -> int:
        return len(typing.cast(typing.Sized, self.__dataset))
    
    def __getitem__(self, i: int) -> typing.Tuple[torch.Tensor, ...]:
        sample = self.__dataset[i]
        
        if self.__append_factories is not None:
            new_features = []
            for factory in self.__append_factories:
                new_features.append(factory(sample))
            sample = list(sample)+new_features
        sample = list(sample)

        if self.__new_order is not None:
            new_sample = typing.cast(list[torch.Tensor], [None for _ in range(len(self.__new_order))])
            for i in range(len(self.__new_order)):
                new_sample[i] = sample[self.__new_order[i]]
            sample = new_sample
        
        return tuple(sample)