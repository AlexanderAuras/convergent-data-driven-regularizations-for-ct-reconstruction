#RUNNING       Real-world data
#IMPLEMENTED   Thikonov
#IMPLEMENTED   Config add analytic vs learned flag
#IMPLEMENTED   Uniform Noise
#IMPLEMENTED   Poisson Noise
#IMPLEMENTED   Gaussian Noise (Cov = diag(linspace(1,0.5)))
#IMPLEMENTED   FBP

#IMPLEMENTED   Sinograms of LoDoPaB
#IMPLEMENTED   Noise in dataset (static, not dynamic)
#IMPLEMENTED   Raw logging

#pyright: reportPrivateImportUsage=false, reportGeneralTypeIssues=false
import functools
import logging
import os
import pathlib
import subprocess
import typing
import warnings
from math import ceil, sqrt

import hydra
import hydra.core.hydra_config
import omegaconf

#Register additional resolver for log path
omegaconf.OmegaConf.register_new_resolver("list_to_string", lambda o: functools.reduce(lambda acc, x: acc+", "+x.replace("\"","").replace("/"," "), o, "")[2:])
omegaconf.OmegaConf.register_new_resolver("eval", lambda c: eval(c))

import pytorch_lightning
import pytorch_lightning.accelerators
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import pytorch_lightning.utilities
import torch
import torch.utils.tensorboard
import torch.version

from apple_ct_datamodule import AppleCTDataModule
from apple_ct_dataset import AppleCTDataset
from ellipses2_datamodule import Ellipses2DataModule
from fbp_model import FBPModel
from filter_model import FilterModel
from fixed_noise_dataset import (AdditiveElementwiseGaussianNoise,
                                 AdditiveElementwisePoissonNoise,
                                 AdditiveElementwiseUniformNoise,
                                 AdditiveTensorwiseGaussianNoise)
from lodopab2_datamodule import LoDoPaB2DataModule
from mnist_datamodule import MNISTDataModule
from svd_model import SVDModel
from tikhonov_model import TikhonovModel


#Custom version of pytorch lightnings TensorBoardLogger, to allow manipulation of internal logging settings
class CustomTensorBoardLogger(pytorch_lightning.loggers.TensorBoardLogger):
    #Disables logging of epoch
    @pytorch_lightning.utilities.rank_zero_only
    def log_metrics(self, metrics: typing.Dict[str, typing.Union[torch.Tensor,float]], step: int) -> None:
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)
    
    #Disables creation of hparams.yaml
    @pytorch_lightning.utilities.rank_zero_only
    def save(self) -> None:
        dir_path = self.log_dir
        if not os.path.isdir(dir_path):
            dir_path = self.save_dir



@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: omegaconf.DictConfig) -> None:
    #multiprocessing.set_start_method("spawn")

    #Setup logging
    logger = logging.getLogger(__name__)
    logging.captureWarnings(True)
    logging.getLogger("pytorch_lightning").handlers.append(logger.root.handlers[1]) #Route pytorch lightning logging to hydra logger
    for old_log in os.listdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir):
        if old_log.startswith("events.out.tfevents"):
            os.remove(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, old_log))

    #Append current git commit hash to saved config
    with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,".hydra","config.yaml"), "a") as cfg_file:
        cfg_file.write(f"git_project: {subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], cwd=hydra.utils.get_original_cwd()).decode('ascii').strip()}\n")
        cfg_file.write(f"git_branch: {subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=hydra.utils.get_original_cwd()).decode('ascii').strip()}\n")
        cfg_file.write(f"git_commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=hydra.utils.get_original_cwd()).decode('ascii').strip()}\n")

    #Set num_workers to available CPU count
    if config.num_workers == -1:
        config.num_workers = os.cpu_count()
    
    #Initialize determinism
    if config.deterministic:
        pytorch_lightning.seed_everything(config.seed, workers=True)

    if config.noise_type == "uniform":
        trainval_noise = AdditiveElementwiseUniformNoise(min_=-sqrt(3.0)*config.noise_level, max_=sqrt(3.0)*config.noise_level)
    elif config.noise_type == "gaussian":
        trainval_noise = AdditiveElementwiseGaussianNoise(sigma=config.noise_level)
    elif config.noise_type == "poisson":
        trainval_noise = AdditiveElementwisePoissonNoise(rate=config.noise_level)
    elif config.noise_type == "multivariate_gaussian":
        sino_numel = (config.sino_angles.shape[0] if config.sino_angles is not None else 256)*(config.sino_positions.shape[0] if config.sino_positions is not None else 2*ceil(config.img_size*1.41421356237/2.0)+1)
        trainval_noise = AdditiveTensorwiseGaussianNoise(mu=torch.zeros((sino_numel,)), sigma=torch.diag_embed(torch.linspace(1.0, 0.5, sino_numel)))
    else:
        raise NotImplementedError()
        
    if config.trainval_dataset.name == "MNIST":
        trainval_datamodule = MNISTDataModule(config, trainval_noise)
    elif config.trainval_dataset.name == "Ellipses":
        #trainval_datamodule = EllipsesDataModule(config, trainval_noise)
        config.img_size = 64
        config.sino_angles = torch.linspace(0.0, torch.pi, 257)[:-1].tolist()
        config.sino_positions = torch.arange(-ceil(64*1.41421356237/2.0), ceil(64*1.41421356237/2.0)+1, dtype=torch.float).tolist()
        trainval_datamodule = Ellipses2DataModule(config, trainval_noise)
    elif config.trainval_dataset.name == "LoDoPaB":
        config.img_size = 64
        config.sino_angles = torch.linspace(0.0, torch.pi, 257)[:-1].tolist()
        config.sino_positions = torch.arange(-ceil(64*1.41421356237/2.0), ceil(64*1.41421356237/2.0)+1, dtype=torch.float).tolist()
        trainval_datamodule = LoDoPaB2DataModule(config, trainval_noise)
        #trainval_datamodule = LoDoPaBDataModule(config)
        #if config.sino_angles != None:
        #    raise RuntimeError("Incompatible sino_angles")
        #if config.sino_positions != None:
        #    raise RuntimeError("Incompatible sino_positions")
        #config.img_size = 362
        #config.sino_angles = torch.linspace(torch.pi*0.5, torch.pi*1.5, 501)[:-1].flip(-1).tolist()
        #config.sino_positions = (-torch.linspace(-362.0/sqrt(2.0), 362.0/sqrt(2.0), 257, dtype=torch.float)).tolist()
    elif config.trainval_dataset.name == "Apple-CT":
        trainval_datamodule = AppleCTDataModule(config)
        if config.sino_angles != None:
            raise RuntimeError("Incompatible sino_angles")
        if config.sino_positions != None:
            raise RuntimeError("Incompatible sino_positions")
        config.sino_angles = AppleCTDataset("/data/datasets/", AppleCTDataset.NoiseType.NONE, AppleCTDataset.Subset.TRAIN, extracted=False).angles.tolist()
        config.sino_positions = torch.arange(-688,689).tolist()
    else:
        raise NotImplementedError()
    
    trainval_datamodule.train_dataloader()

    if config.noise_type == "uniform":
        test_noise = AdditiveElementwiseUniformNoise(min_=-sqrt(3.0)*config.noise_level, max_=sqrt(3.0)*config.noise_level)
    elif config.noise_type == "gaussian":
        test_noise = AdditiveElementwiseGaussianNoise(sigma=config.noise_level)
    elif config.noise_type == "poisson":
        test_noise = AdditiveElementwisePoissonNoise(rate=config.noise_level)
    elif config.noise_type == "multivariate_gaussian":
        sino_numel = (config.sino_angles.shape[0] if config.sino_angles is not None else 256)*(config.sino_positions.shape[0] if config.sino_positions is not None else 2*ceil(config.img_size*1.41421356237/2.0)+1)
        test_noise = AdditiveTensorwiseGaussianNoise(mu=torch.zeros((sino_numel,)), sigma=torch.diag_embed(torch.linspace(1.0, 0.5, sino_numel)))
    else:
        raise NotImplementedError()
    
    if config.test_dataset.name == "MNIST":
        test_datamodule = MNISTDataModule(config, test_noise)
    elif config.test_dataset.name == "Ellipses":
        #test_datamodule = EllipsesDataModule(config, test_noise)
        config.img_size = 64
        config.sino_angles = torch.linspace(0.0, torch.pi, 257)[:-1].tolist()
        config.sino_positions = torch.arange(-ceil(64*1.41421356237/2.0), ceil(64*1.41421356237/2.0)+1, dtype=torch.float).tolist()
        test_datamodule = Ellipses2DataModule(config, test_noise)
    elif config.test_dataset.name == "LoDoPaB":
        config.img_size = 64
        config.sino_angles = torch.linspace(0.0, torch.pi, 257)[:-1].tolist()
        config.sino_positions = torch.arange(-ceil(64*1.41421356237/2.0), ceil(64*1.41421356237/2.0)+1, dtype=torch.float).tolist()
        test_datamodule = LoDoPaB2DataModule(config, test_noise)
        #test_datamodule = LoDoPaBDataModule(config)
        #if config.img_size != None and config.trainval_dataset.name != "LoDoPaB":
        #    raise RuntimeError("Incompatible img_size")
        #if config.sino_angles != None and config.trainval_dataset.name != "LoDoPaB":
        #    raise RuntimeError("Incompatible sino_angles")
        #if config.sino_positions != None and config.trainval_dataset.name != "LoDoPaB":
        #    raise RuntimeError("Incompatible sino_positions")
        #config.img_size = 362
        #config.sino_angles = torch.linspace(torch.pi*0.5, torch.pi*1.5, 501)[:-1].flip(-1).tolist()
        #config.sino_positions = (-torch.linspace(-362.0/sqrt(2.0), 362.0/sqrt(2.0), 257, dtype=torch.float)).tolist()
    elif config.test_dataset.name == "Apple-CT":
        test_datamodule = AppleCTDataModule(config)
        if config.sino_angles != None and config.trainval_dataset.name != "Apple-CT":
            raise RuntimeError("Incompatible sino_angles")
        if config.sino_positions != None and config.trainval_dataset.name != "Apple-CT":
            raise RuntimeError("Incompatible sino_positions")
        config.sino_angles = AppleCTDataset("/data/datasets/", AppleCTDataset.NoiseType.NONE, AppleCTDataset.Subset.TEST, extracted=False).angles.tolist()
        config.sino_positions = torch.arange(-688,689).tolist()
    else:
        raise NotImplementedError()

    #Create model and load data
    if config.model.name == "filter":
        modelClass = FilterModel
    elif config.model.name == "svd":
        modelClass = SVDModel
    elif config.model.name == "tikhonov":
        modelClass = TikhonovModel
    elif config.model.name == "fbp":
        modelClass = FBPModel
    else:
        raise NotImplementedError()
    ################################################################
    ################################################################
    ################################################################
    ##                                                            ##
    ##  ##### ##    ##   ####   ##    ##  #####  ######  ## ## ## ##
    ## ##     ##    ##  ##  ##  ###   ## ##      ##      ## ## ## ##
    ## ##     ######## ######## ## ## ## ## #### ######  ## ## ## ##
    ## ##     ##    ## ##    ## ##   ### ##   ## ##               ##
    ##  ##### ##    ## ##    ## ##    ##  #####  ######  ## ## ## ##
    ##                                                            ##
    ################################################################
    if config.checkpoint != None:
        model = modelClass.load_from_checkpoint(os.path.abspath(os.path.join("../../" if hydra.core.hydra_config.HydraConfig.get().mode == hydra.types.RunMode.MULTIRUN else "../", config.checkpoint)), config=config)
    else:
        if modelClass == SVDModel or modelClass == FBPModel:
            model = modelClass(config, str(pathlib.Path(__file__).parent.joinpath("cache")))
        elif modelClass == TikhonovModel:
            model = modelClass(config, str(pathlib.Path(__file__).parent.joinpath("cache", "A_"+config.noise_type[0]+{
                0.0: "n",
                0.005: "l",
                0.015: "m",
                0.03: "h",
                0.01: "x"
            }[config.noise_level]+config.trainval_dataset.name[0].lower()+".pt")))
        else:
            model = modelClass(config)

    #import radon
    #batch = next(iter(test_datamodule.test_dataloader()))
    #noisy_sino = batch[0].to(config.device)[0:1]
    #ground_truth = batch[1].to(config.device)[0:1]
    #alpha = 1.0
    #print(f"Alpha hyperparameter search (target_loss = {config.noise_level**2})", flush=True)
    #while True:
    #    config.model.alpha = alpha
    #    model = modelClass(config).to(config.device)
    #    recon = model.test_step((noisy_sino, ground_truth, torch.zeros_like(noisy_sino), torch.zeros_like(noisy_sino)), 0)["analytic_reconstruction"]
    #    recon_sino = radon.radon_forward(recon, thetas=model.angles, positions=model.positions)
    #    loss = torch.nn.functional.mse_loss(recon_sino, noisy_sino)
    #    if abs(loss.item()-config.noise_level**2) <= config.noise_level**2/100.0:
    #        print(f"Final alpha: {alpha} (loss={loss.item()})", flush=True)
    #        break
    #    if loss.item() > config.noise_level**2:
    #        alpha *= 0.9
    #        print(f"Loss > alpha^2: alpha <- {alpha} (loss={loss.item()})", flush=True)
    #    else:
    #        alpha *= 1.1
    #        print(f"Loss < alpha^2: alpha <- {alpha} (loss={loss.item()})", flush=True)
    #exit(0)
    ###############################################################
    ###############################################################
    ###############################################################

    #Execute training and testing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Checkpoint directory .+ exists and is not empty\.") #Needed thanks to the custom logger
        warnings.filterwarnings("ignore", r"The dataloader, ((train)|(val)|(test)) dataloader( \d+)?, does not have many workers which may be a bottleneck\. Consider increasing the value of the `num_workers` argument` \(try \d+ which is the number of cpus on this machine\) in the `DataLoader` init to improve performance\.") #BUG Contradictory warnings with num_workers on cluster and slow loading with LoDoPaB
        trainer = pytorch_lightning.Trainer(
            deterministic=config.deterministic, 
            callbacks=[pytorch_lightning.callbacks.ModelCheckpoint(dirpath=".")], 
            accelerator="gpu" if config.device == "cuda" else None, devices=1,
            max_epochs=config.epochs, 
            logger=CustomTensorBoardLogger(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, None, ""), 
            limit_train_batches=int(config.training_batch_count) if config.training_batch_count != -1 else len(trainval_datamodule.train_dataloader()), 
            limit_val_batches=int(config.validation_batch_count) if config.validation_batch_count != -1 else len(trainval_datamodule.val_dataloader()), 
            limit_test_batches=int(config.test_batch_count) if config.test_batch_count != -1 else len(test_datamodule.test_dataloader()))
        trainer.fit(model, trainval_datamodule)
        trainer.test(model, test_datamodule)


    logging.getLogger("pytorch_lightning").handlers.remove(logger.root.handlers[1]) #Stops pytorch lightning from writing to previous runs during multiruns



if __name__ == "__main__":
    main()