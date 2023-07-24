import inspect
from math import ceil
import typing
import warnings

import omegaconf

import pytorch_lightning as pl
import pytorch_lightning.loggers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard

import torchmetrics

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import radon as radon

from utils import log_img



class TikhonovModel(pl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.randn((1,1,len(self.config.sino_angles) if self.config.sino_angles != None else 256,len(self.config.sino_positions) if self.config.sino_positions != None else ceil((self.config.img_size*1.41421356237)/2.0)*2+1)) #Needed for pytorch lightning
        self.automatic_optimization = False

        self.angles = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_angles, dtype=torch.float32), requires_grad=False) if self.config.sino_angles != None else None
        self.positions = torch.nn.parameter.Parameter(torch.tensor(self.config.sino_positions, dtype=torch.float32), requires_grad=False) if self.config.sino_positions != None else None
        
        #print("Calculating radon matrix", flush=True)
        A = radon.radon_matrix(torch.zeros(self.config.img_size, self.config.img_size), thetas=self.angles, positions=self.positions)
        #print("Calculating regularized matrix", flush=True)
        alphaI = self.config.model.alpha*torch.eye(A.shape[1])
        #print(alphaI.shape, flush=True)
        ATA = A.mT@A
        #print(ATA.shape, flush=True)
        pre_inv = ATA + alphaI
        #print("Pre_inv calculated", flush=True)
        ######
        self.A = torch.nn.parameter.Parameter(pre_inv, requires_grad=False)
        ######
        #inv = torch.inverse(pre_inv)
        #print("Inverse calculated")
        #self.matrix = torch.nn.parameter.Parameter(inv@A.mT, requires_grad=False)
        ##self.matrix = (A.mT@A+self.config.alpha*torch.eye(A.shape[1])).inv()@A.mT
        #print("Regularized matrix calculated")

        #Setup metrics
        with warnings.catch_warnings():
            self.training_learned_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_learned_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_learned_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_learned_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_learned_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_learned_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_input_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_learned_output_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_analytic_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_analytic_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.training_analytic_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_analytic_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_analytic_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.validation_analytic_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_psnr_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_ssim_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_input_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")
            self.test_analytic_output_l2_metric = torchmetrics.MeanMetric(nan_strategy="ignore")



    #HACK Removes metrics from PyTorch Lightning overview
    def named_children(self) -> typing.Iterator[typing.Tuple[str, nn.Module]]:
        stack = inspect.stack()
        if stack[2].function == "summarize" and stack[2].filename.endswith("pytorch_lightning/utilities/model_summary/model_summary.py"):
            return filter(lambda x: not x[0].endswith("metric"), super().named_children())
        return super().named_children()



    def forward_analytic(self, x: torch.Tensor) -> torch.Tensor:
        #print("Forward analytic", flush=True)
        b = radon.radon_backward(x, self.config.img_size, self.angles, self.positions)
        b = b.reshape(*b.shape[:-2], -1, 1)
        z = torch.linalg.solve(self.A, b)
        return z.reshape(x.shape[0], 1, self.config.img_size, self.config.img_size)
        #return (self.matrix@x.reshape(*x.shape[:-2], -1, 1)).reshape(x.shape[0], 1, self.config.img_size, self.config.img_size)



    def forward_fbp(self, x: torch.Tensor) -> torch.Tensor:
        filtered_sinogram = radon.radon_filter(x, radon.ram_lak_filter)
        return radon.radon_backward(filtered_sinogram, self.config.img_size, self.angles, self.positions)



    #Apply model for n iterations
    def forward(self, sino: torch.Tensor) -> torch.Tensor: #type: ignore
        return self.forward_analytic(sino)
    

    def configure_optimizers(self) -> None:
        pass
    

    def training_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> None:#torch.Tensor: #type: ignore
        pass



    def validation_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor], batch_idx: int) -> typing.Dict[str,typing.Union[torch.Tensor,None]]: #type: ignore
        pass



    def validation_epoch_end(self, outputs: typing.List[typing.Dict[str,typing.Union[torch.Tensor,typing.List[torch.Tensor]]]]) -> None: #type: ignore
        pass



    def test_step(self, batch: typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor], batch_idx: int) -> typing.Dict[str,typing.Union[torch.Tensor,typing.List[torch.Tensor]]]: #type: ignore
        #Reset metrics
        self.test_learned_loss_metric.reset()
        self.test_learned_psnr_metric.reset()
        self.test_learned_ssim_metric.reset()
        self.test_analytic_loss_metric.reset()
        self.test_analytic_psnr_metric.reset()
        self.test_analytic_ssim_metric.reset()

        #Forward pass
        noisy_sinogram = batch[0]
        ground_truth   = batch[1]
        sinogram       = batch[2]
        #ground_truth = batch[0] if len(batch) == 2 else batch[1]
        #sinogram = radon.radon_forward(ground_truth, torch.tensor(self.config.sino_angles, device=ground_truth.device) if self.config.sino_angles != None else None, torch.tensor(self.config.sino_positions, device=ground_truth.device) if self.config.sino_positions != None else None)
        #noise = torch.zeros_like(sinogram)
        #if self.config.noise_type == "uniform":
        #    noise = self.config.noise_level*torch.rand_like(sinogram)
        #elif self.config.noise_type == "gaussian":
        #    noise = self.config.noise_level*torch.randn_like(sinogram)
        #elif self.config.noise_type == "poisson":
        #    noise = self.config.noise_level*torch.poisson(sinogram)
        #elif self.config.noise_type == "multivariate_gaussian":
        #    noise = torch.zeros_like(sinogram)
        #    mvn = torch.distributions.MultivariateNormal(torch.zeros((sinogram[0].numel(),)), torch.diag_embed(torch.linspace(1.0, 0.5, sinogram[0].numel())))
        #    for i in range(sinogram.shape[0]):
        #        noise[i] = mvn.sample().reshape(sinogram.shape).to(sinogram.dtype).to(sinogram.device)
        #    noise *= self.config.noise_level
        #noisy_sinogram = sinogram+noise
        analytic_reconstruction = self.forward_analytic(noisy_sinogram)
        self.test_analytic_loss_metric.update(F.mse_loss(analytic_reconstruction, ground_truth))
        self.test_analytic_psnr_metric.update(torchmetrics.functional.peak_signal_noise_ratio(analytic_reconstruction, ground_truth))
        self.test_analytic_ssim_metric.update(typing.cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(analytic_reconstruction, ground_truth)))
        self.test_analytic_input_l2_metric.update(torch.sqrt(torch.sum(ground_truth**2, 3).sum(2)).mean())
        self.test_analytic_output_l2_metric.update(torch.sqrt(torch.sum(analytic_reconstruction**2, 3).sum(2)).mean())
        fbp = self.forward_fbp(noisy_sinogram)

        #Return data for logging purposes
        if batch_idx < 10:
            return {
                "sinogram": sinogram, 
                "noisy_sinogram": noisy_sinogram, 
                "ground_truth": ground_truth,
                "analytic_reconstruction": analytic_reconstruction,
                "fbp": fbp
            }
        else:
            return {}



    def test_epoch_end(self, outputs: typing.List[typing.Dict[str,typing.Union[torch.Tensor,typing.List[torch.Tensor]]]]) -> None: #type: ignore
        if self.logger and self.trainer.is_global_zero:
            logger = typing.cast(pytorch_lightning.loggers.TensorBoardLogger, self.logger).experiment

            #Log mean test metrics
            logger.add_scalar("test/analytic_loss", self.test_analytic_loss_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_psnr", self.test_analytic_psnr_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_ssim", self.test_analytic_ssim_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_input_l2", self.test_analytic_input_l2_metric.compute().item(), 0)
            logger.add_scalar("test/analytic_output_l2", self.test_analytic_output_l2_metric.compute().item(), 0)

            #Log examples
            for i in range(10):
                sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["sinogram"][0,0]
                noisy_sinogram = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["noisy_sinogram"][0,0]
                ground_truth = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["ground_truth"][0,0]
                fbp = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["fbp"][0,0]
                log_img(logger, "test/sinogram", sinogram.mT, i)
                log_img(logger, "test/noisy_sinogram", noisy_sinogram.mT, i)
                log_img(logger, "test/ground_truth", ground_truth, i)
                log_img(logger, "test/fbp", fbp, i)
                analytic_reconstruction = typing.cast(typing.List[typing.Dict[str,torch.Tensor]], outputs)[i]["analytic_reconstruction"][0,0]
                log_img(logger, "test/analytic_reconstruction", analytic_reconstruction, i)
                if self.config.log_raw:
                    torch.save(sinogram, f"test_sinogram_{i}.pt")
                    torch.save(noisy_sinogram, f"test_noisy_sinogram_{i}.pt")
                    torch.save(fbp, f"test_fbp_{i}.pt")
                    torch.save(ground_truth, f"test_ground_truth_{i}.pt")
                    torch.save(analytic_reconstruction, f"test_analytic_reconstruction_{i}.pt")