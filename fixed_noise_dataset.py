import abc
import typing

import torch
import torch.utils.data

from deterministic_multivariate_normal import DeterministicMultivariateNormal


class Noise(torch.nn.Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self._generator = torch.Generator()
        self.__initial_generator_state = self._generator.get_state()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        ...

    def reset(self, seed: typing.Union[int, None] = None) -> None:
        self._generator.set_state(self.__initial_generator_state)
        if seed is not None:
            self._generator.manual_seed(seed)

class AdditiveElementwiseUniformNoise(Noise):
    def __init__(self, min_: float = 0.0, max_: float = 1.0) -> None:
        super().__init__()
        self.__min = min_
        self.__max = max_
    
    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = self.__min+(self.__max-self.__min)*torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self._generator)
        return (x+noise, noise)

class AdditiveElementwiseGaussianNoise(Noise):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.__mu = mu
        self.__sigma = sigma
    
    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = self.__mu+self.__sigma*torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=self._generator)
        return (x+noise, noise)

class AdditiveElementwisePoissonNoise(Noise):
    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        self.__rate = rate
    
    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.poisson(torch.full(x.shape, self.__rate, dtype=x.dtype, device=x.device), generator=self._generator)
        return (x+noise, noise)

class AdditiveTensorwiseGaussianNoise(Noise):
    def __init__(self, mu: torch.Tensor, sigma: typing.Union[torch.Tensor, None] = None) -> None:
        super().__init__()
        if sigma is None:
            sigma = torch.diag_embed(torch.ones((mu.numel())))
        self.__distribution = DeterministicMultivariateNormal(mu, sigma)
    
    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        noise = self.__distribution.rsample(x.shape, generator=self.__generator).to(x.dtype).to(x.device)
        return (x+noise, noise)


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