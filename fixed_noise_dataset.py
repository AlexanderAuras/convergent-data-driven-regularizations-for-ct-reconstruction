import abc
import typing

import torch
import torch.utils.data

from deterministic_multivariate_normal import DeterministicMultivariateNormal


class Noise(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self._generator = torch.Generator()
        self.__initial_generator_state = self._generator.get_state()

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        ...

    def reset(self, seed: typing.Union[int, None] = None) -> None:
        self._generator.set_state(self.__initial_generator_state)
        if seed is not None:
            self._generator.manual_seed(seed)

    def __getstate__(self) -> tuple[list[int], list[int]]:
        return (self.__initial_generator_state.tolist(), self._generator.get_state().tolist())
    
    def __setstate__(self, state: tuple[list[int], list[int]]) -> None:
        self._generator = torch.Generator()
        self.__initial_generator_state = torch.tensor(state[0], dtype=torch.uint8)
        self._generator.set_state(torch.tensor(state[1], dtype=torch.uint8))

class AdditiveElementwiseUniformNoise(Noise):
    def __init__(self, min_: float = 0.0, max_: float = 1.0) -> None:
        super().__init__()
        self.__min = min_
        self.__max = max_
    
    def __call__(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = self.__min+(self.__max-self.__min)*torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self._generator)
        return (x+noise, noise)

    def __getstate__(self) -> tuple[tuple[list[int], list[int]], float, float]:  # type: ignore
        return (super().__getstate__(), self.__min, self.__max)
    
    def __setstate__(self, state: tuple[tuple[list[int], list[int]], float, float]) -> None:  # type: ignore
        super().__setstate__(state[0])
        self.__min = state[1]
        self.__max = state[2]

class AdditiveElementwiseGaussianNoise(Noise):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.__mu = mu
        self.__sigma = sigma
    
    def __call__(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = self.__mu+self.__sigma*torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=self._generator)
        return (x+noise, noise)

    def __getstate__(self) -> tuple[tuple[list[int], list[int]], float, float]:  # type: ignore
        return (super().__getstate__(), self.__mu, self.__sigma)
    
    def __setstate__(self, state: tuple[tuple[list[int], list[int]], float, float]) -> None:  # type: ignore
        super().__setstate__(state[0])
        self.__mu = state[1]
        self.__sigma = state[2]

class AdditiveElementwisePoissonNoise(Noise):
    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        self.__rate = rate
    
    def __call__(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.poisson(torch.full(x.shape, self.__rate, dtype=x.dtype, device=x.device), generator=self._generator)
        return (x+noise, noise)

    def __getstate__(self) -> tuple[tuple[list[int], list[int]], float]:  # type: ignore
        return (super().__getstate__(), self.__rate)
    
    def __setstate__(self, state: tuple[tuple[list[int], list[int]], float]) -> None:  # type: ignore
        super().__setstate__(state[0])
        self.__rate = state[1]

class AdditiveTensorwiseGaussianNoise(Noise):
    def __init__(self, mu: torch.Tensor, sigma: typing.Union[torch.Tensor, None] = None) -> None:
        super().__init__()
        if sigma is None:
            sigma = torch.diag_embed(torch.ones((mu.numel())))
        self.__distribution = DeterministicMultivariateNormal(mu, sigma)
    
    def __call__(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = self.__distribution.rsample(x.shape, generator=self._generator).to(x.dtype).to(x.device)
        return (x+noise, noise)

    def __getstate__(self) -> tuple[tuple[list[int], list[int]], float, float]:  # type: ignore
        raise NotImplementedError()
    
    def __setstate__(self, state: tuple[tuple[list[int], list[int]], float, float]) -> None:  # type: ignore
        raise NotImplementedError()


class FixedNoiseDataset(torch.utils.data.Dataset[typing.Tuple[torch.Tensor, ...]]):
    def __init__(self, 
                 dataset: torch.utils.data.Dataset[typing.Tuple[torch.Tensor, ...]], 
                 noisy_features: typing.Sequence[int] = (0,), 
                 noise: typing.Union[Noise, None] = None,
                 append_clean: bool = False,
                 append_noise: bool = False) -> None:
        super().__init__()
        self.__dataset = dataset
        self.__noisy_features = noisy_features
        if noise is None:
            noise = AdditiveElementwiseGaussianNoise()
        self.__noise = noise
        self.__seeds = torch.randint(10000000,99999999, (len(typing.cast(typing.Sized, dataset)),len(noisy_features)))
        self.__append_clean = append_clean
        self.__append_noise = append_noise

    def __len__(self) -> int:
        return len(typing.cast(typing.Sized, self.__dataset))
    
    def __getitem__(self, i: int) -> typing.Tuple[torch.Tensor, ...]:
        sample = self.__dataset[i]
        transformed_features = []
        clean_features = []
        noise_list = []
        for j in range(len(sample)):
            if j in self.__noisy_features:
                self.__noise.reset(int(self.__seeds[i,j].item()))
                noisy_feature, noise = self.__noise(sample[j])
                transformed_features.append(noisy_feature)
                clean_features.append(sample[j])
                noise_list.append(noise)
            else:
                transformed_features.append(sample[j])
        result = [*transformed_features]
        if self.__append_clean:
            result += clean_features
        if self.__append_noise:
            result += noise_list
        return tuple(result)