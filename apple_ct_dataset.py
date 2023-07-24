import csv
import enum
import hashlib
import logging
import pathlib
import random
import re
import typing
import urllib.request
import zipfile
from functools import reduce

import numpy as np
import PIL.Image
import torch.utils.data
import torchvision
import tqdm


class AppleCTDataset(torch.utils.data.Dataset[typing.Tuple[torch.Tensor,torch.Tensor,int,int]]):
    class Subset(enum.Enum):
        ALL = enum.auto()
        TRAIN = enum.auto()
        VAL = enum.auto()
        TEST = enum.auto()
    
    class NoiseType(enum.Enum):
        NONE = "A"
        GAUSSIAN = "B"
        SCATTER = "C"

    def __init__(self, 
                 path: str|pathlib.Path, 
                 noise_type: NoiseType=NoiseType.NONE,
                 subset: Subset=Subset.ALL,
                 randomize_validation: bool=False,
                 extracted: bool=False, 
                 transform: typing.Callable[[torch.Tensor],torch.Tensor]=lambda x: x, 
                 target_transform: typing.Callable[[torch.Tensor],torch.Tensor]=lambda x: x, 
                 download: bool=False) -> None:
        super().__init__()
        self.__transform = transform
        self.__target_transform = target_transform
        self.__subset = subset
        self.__noise_type = noise_type
        self.__extracted = extracted

        if isinstance(path, str):
            path = pathlib.Path(path)
        self.__base_path = path.joinpath("Apple_CT")

        logger = logging.getLogger(__name__)
        if path.exists():
            logger.debug(f"Found dataset directory at {self.__base_path}")
            logger.debug("Verifying dataset installation...")
            self.__verify_installation(True)
            logger.debug("Installation verified")
        elif download:
            logger.debug(f"Dataset not found at {self.__base_path}, downloading...")
            self.__download()
            logger.debug("Download complete")
            if extracted:
                logger.debug("Extracting dataset")
                self.__extract()
                logger.debug("Extractions complete")
            logger.debug("Verifying dataset installation...")
            self.__verify_installation(False)
            logger.debug("Installation verified")
        else:
            raise FileNotFoundError(f"Cannot find dataset at {path.resolve()} and download is disabled")
        
        logger.debug("Loading angles id...")
        with open(self.__base_path.joinpath("proj_angs.txt").resolve()) as file:
            self.angles = torch.from_numpy(np.fromiter(map(float, file.readlines()), np.float_))
        logger.debug("Angles loaded")

        if randomize_validation:
            self.__train_ids, self.__val_ids = self.__split_trainval()
        else:
            self.__train_ids = AppleCTDataset._TRAIN_IDS_PARTIAL if noise_type == AppleCTDataset.NoiseType.SCATTER else AppleCTDataset._TRAIN_IDS_FULL
            self.__val_ids   = AppleCTDataset._VAL_IDS_PARTIAL   if noise_type == AppleCTDataset.NoiseType.SCATTER else AppleCTDataset._VAL_IDS_FULL
        self.__test_ids   = AppleCTDataset._TEST_IDS_PARTIAL   if noise_type == AppleCTDataset.NoiseType.SCATTER else AppleCTDataset._TEST_IDS_FULL


    def __md5_hash(self, path: pathlib.Path) -> str:
        md5 = hashlib.md5()
        with open(path.resolve(), "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()

    
    def __verify_installation(self, skip_hash: bool) -> None:
        logger = logging.getLogger(__name__)
        file_infos = {} if self.__extracted else AppleCTDataset._FILE_INFOS
        for file_name, file_info in file_infos.items():
            if not self.__extracted:
                logger.debug(f"    Verifying {file_name}...")
            file_path = self.__base_path.joinpath(file_name)
            if not file_path.exists():
                raise FileNotFoundError(f"Corrupted dataset at {self.__base_path.resolve()}, {file_name} is missing")
            if not file_path.is_file():
                raise FileNotFoundError(f"Corrupted dataset at {self.__base_path.resolve()}, {file_name} is not a normal file")
            if not skip_hash and self.__md5_hash(file_path) != file_info.md5:
                raise FileNotFoundError(f"Corrupted dataset at {self.__base_path.resolve()}, {file_name} is corrupted")
            
        
    def __download_file(self, url: str, path: pathlib.Path) -> None:
        def update_progress(progress: typing.Any, current_size: int, _: int, file_size: int) -> None:
            if progress.total == -1:
                progress.total = file_size
            progress.update(current_size)
        if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
            with tqdm.tqdm(desc=f"Download", total=-1, unit="B", unit_scale=True) as progress:
                urllib.request.urlretrieve(url, path.resolve(), reporthook=lambda bc, bs, fs: update_progress(progress, bc, bs, fs))
        else:
            urllib.request.urlretrieve(url, path.resolve())
            
        
    def __download(self) -> None:
        logger = logging.getLogger(__name__)
        self.__base_path.mkdir()
        for file_name, file_info in AppleCTDataset._FILE_INFOS.items():
            logger.debug(f"    Downloading {file_name}...")
            file_path = self.__base_path.joinpath(file_name)
            self.__download_file(file_info.url, file_path)

    
    def __extract(self) -> None:
        logger = logging.getLogger(__name__)
        for file_name in AppleCTDataset._FILE_INFOS.keys():
            if not file_name.endswith(".zip"):
                continue
            logger.debug(f"    Extracting {file_name}...")
            file_path = self.__base_path.joinpath(file_name)
            with zipfile.ZipFile(file_path.resolve(), "r") as zip_file:
                for name in zip_file.namelist():
                    if name in ["Dataset_A/", "Dataset_B/", "Dataset_C/"]:
                        continue
                    match = typing.cast(re.Match[str], re.fullmatch(r"Dataset_"+self.__noise_type.value+r"/data_(?:noisy_)?(\d+)_(\d+).tif", name))
                    zip_file.getinfo(name).filename = f"data_{self.__noise_type.value}_{match.group(1)}_{match.group(2)}.tif"
                    zip_file.extract(name)
            file_path.unlink()


    def __len__(self) -> int:
        return {
            AppleCTDataset.NoiseType.NONE:     {AppleCTDataset.Subset.TRAIN: 768*63, AppleCTDataset.Subset.VAL: 768*11, AppleCTDataset.Subset.TEST: 768*20},
            AppleCTDataset.NoiseType.GAUSSIAN: {AppleCTDataset.Subset.TRAIN: 768*63, AppleCTDataset.Subset.VAL: 768*11, AppleCTDataset.Subset.TEST: 768*20},
            AppleCTDataset.NoiseType.SCATTER:  {AppleCTDataset.Subset.TRAIN: 768*63, AppleCTDataset.Subset.VAL: 768*11, AppleCTDataset.Subset.TEST: 768*20}
        }[self.__noise_type][self.__subset]
    

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor,torch.Tensor,int,int]:
        apple_id_idx = idx//80 if self.__noise_type == AppleCTDataset.NoiseType.SCATTER else idx//768
        apple_id = {AppleCTDataset.Subset.TRAIN: self.__train_ids, AppleCTDataset.Subset.VAL: self.__val_ids, AppleCTDataset.Subset.TEST: self.__test_ids}[self.__subset][apple_id_idx]
        slice_no = idx%80 if self.__noise_type == AppleCTDataset.NoiseType.SCATTER else idx%768
        with torch.no_grad():
            if not self.__extracted:
                zip_file = zipfile.ZipFile(self.__base_path.joinpath(f"Dataset_{self.__noise_type.value}.zip").resolve(), "r")
                file = zip_file.open(f"Dataset_{self.__noise_type.value}/data{'_noisy' if self.__noise_type == AppleCTDataset.NoiseType.GAUSSIAN else ''}_{apple_id}_{slice_no:03}.tif")
            else:
                file = open(self.__base_path.joinpath(f"data_{self.__noise_type.value}_{apple_id}_{slice_no:03}.tif").resolve(), "rb")
            observation = torchvision.transforms.ToTensor()(PIL.Image.open(file))
            file.close()
            if not self.__extracted:
                zip_file.close() # type: ignore

            if not self.__extracted:
                zip_file = zipfile.ZipFile(self.__base_path.joinpath(f"ground_truths_{AppleCTDataset._AppleIDZipMap[apple_id]}.zip").resolve(), "r")
                file = zip_file.open(f"{apple_id}_{slice_no:03}.tif")
            else:
                file = open(self.__base_path.joinpath(f"{apple_id}_{slice_no:03}.tif").resolve(), "rb")
            ground_truth = torchvision.transforms.ToTensor()(PIL.Image.open(file))
            file.close()
            if not self.__extracted:
                zip_file.close() # type: ignore
        
        return self.__transform(observation), self.__target_transform(ground_truth), apple_id, slice_no


    def __split_trainval(self, tries: int=10000) -> typing.Tuple[list[int],list[int]]:
        idx_no_map = {}
        no_idx_map = {}
        file_name = "apple_defect_partial.csv" if self.__noise_type == AppleCTDataset.NoiseType.SCATTER else "apple_defect_full.csv"
        defects = np.zeros((94,4))
        with open(self.__base_path.joinpath(file_name).resolve()) as file:
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                idx_no_map[int(row[0])] = i
                no_idx_map[i] = int(row[0])
                defects[i,0] = int(row[1])
                defects[i,1] = int(row[2])
                defects[i,2] = int(row[3])
                defects[i,3] = int(row[4])
        defects /= defects.sum(0)

        best_val = None
        best_val_loss = float("inf")
        test_ids = AppleCTDataset._TEST_IDS_PARTIAL if self.__noise_type == AppleCTDataset.NoiseType.SCATTER else AppleCTDataset._TEST_IDS_FULL
        allowed_ids = list(set(range(94)).difference(map(lambda x: idx_no_map[x], test_ids)))
        for _ in range(tries):
            perm = random.sample(allowed_ids, 11)
            loss = np.sum((defects[perm].sum(0)-0.2)**2)
            if best_val is None or best_val_loss > loss:
                best_val = perm
                best_val_loss = loss
        best_val = typing.cast(list[int], best_val)
        best_train = set(allowed_ids).difference()
        
        return sorted(map(lambda x: no_idx_map[x], best_train)), sorted(map(lambda x: no_idx_map[x], best_val))
    

    _TRAIN_IDS_PARTIAL = [31101, 31102, 31103, 31104, 31105, 31106, 31107, 31108, 31110, 31111, 31112, 31113, 31117, 31118, 31119, 31120, 31122, 31201, 31203, 31204, 31205, 31206, 31207, 31209, 31211, 31212, 31213, 31214, 31215, 31216, 31217, 31218, 31220, 31303, 31305, 31308, 31309, 31310, 31311, 31312, 31314, 31316, 31317, 31319, 31322, 32101, 32104, 32105, 32107, 32108, 32109, 32111, 32113, 32115, 32117, 32118, 32120, 32121, 32122, 32202, 32203, 32205, 32206]
    _TRAIN_IDS_FULL    = [31102, 31103, 31104, 31105, 31106, 31110, 31111, 31112, 31113, 31114, 31115, 31116, 31117, 31118, 31119, 31120, 31202, 31203, 31205, 31208, 31209, 31210, 31211, 31212, 31213, 31215, 31216, 31217, 31218, 31219, 31220, 31221, 31222, 31301, 31303, 31304, 31306, 31307, 31308, 31309, 31311, 31312, 31314, 31318, 31320, 31321, 31322, 32101, 32102, 32104, 32105, 32106, 32107, 32109, 32110, 32111, 32112, 32113, 32116, 32118, 32120, 32122, 32202]
    _VAL_IDS_PARTIAL   = [31109, 31116, 31121, 31210, 31306, 31313, 32103, 32106, 32112, 32114, 32116]
    _VAL_IDS_FULL      = [31121, 31206, 31207, 31310, 31316, 31319, 32103, 32114, 32117, 32203, 32206]
    _TEST_IDS_PARTIAL  = [31114, 31115, 31202, 31208, 31219, 31221, 31222, 31301, 31302, 31304, 31307, 31315, 31318, 31320, 31321, 32102, 32110, 32119, 32201, 32204]
    _TEST_IDS_FULL     = [31101, 31107, 31108, 31109, 31122, 31201, 31204, 31214, 31302, 31305, 31313, 31315, 31317, 32108, 32115, 32119, 32121, 32201, 32204, 32205]
    _AppleCTFileInfo = typing.NamedTuple("AppleCTFileInfo", md5=str, url=str)
    _FILE_INFOS = {
        "apple_defect_full.csv":    _AppleCTFileInfo(md5="e2fd7d2f5eeb3ab88602c9d95a7a12d3", url="https://zenodo.org/record/4212301/files/apple_defect_full.csv?download=1"),
        "apple_defect_partial.csv": _AppleCTFileInfo(md5="4fee01b076920f85fc92e0de774dc277", url="https://zenodo.org/record/4212301/files/apple_defect_partial.csv?download=1"),
        "proj_angs.txt":            _AppleCTFileInfo(md5="e001b52ec7384fdcfa77a4026ee7b4d2", url="https://zenodo.org/record/4212301/files/proj_angs.txt?download=1"),
        "Dataset_A.zip":            _AppleCTFileInfo(md5="b1764054100c5d6273820db9fa0f38bd", url="https://zenodo.org/record/4212301/files/Dataset_A.zip?download=1"),
        "Dataset_B.zip":            _AppleCTFileInfo(md5="fa85eea7301bba5d5936caaa0ab202ef", url="https://zenodo.org/record/4212301/files/Dataset_B.zip?download=1"),
        "Dataset_C.zip":            _AppleCTFileInfo(md5="044e64693f88f9b4a63d29a539f6791f", url="https://zenodo.org/record/4212301/files/Dataset_C.zip?download=1"),
        "ground_truths_1.zip":       _AppleCTFileInfo(md5="c1e364e5b32cd35f2caea3d253f4baec", url="https://zenodo.org/record/4550729/files/ground_truths_1.zip?download=1"),
        "ground_truths_2.zip":       _AppleCTFileInfo(md5="752e0c59e400ceea9d72741cdb028427", url="https://zenodo.org/record/4575904/files/ground_truths_2.zip?download=1"),
        "ground_truths_3.zip":       _AppleCTFileInfo(md5="a5ba432eaf8a14b68c2333e76d48b0b2", url="https://zenodo.org/record/4576078/files/ground_truths_3.zip?download=1"),
        "ground_truths_4.zip":       _AppleCTFileInfo(md5="2b0ce4e8167b952f042f781687b2acde", url="https://zenodo.org/record/4576122/files/ground_truths_4.zip?download=1"),
        "ground_truths_5.zip":       _AppleCTFileInfo(md5="63f3622c121c63b72ab455b2f6b03d8f", url="https://zenodo.org/record/4576202/files/ground_truths_5.zip?download=1"),
        "ground_truths_6.zip":       _AppleCTFileInfo(md5="13f3da41bdfbb07ec7b13898bdc700ab", url="https://zenodo.org/record/4576260/files/ground_truths_6.zip?download=1")
    }
    _AppleIDZipMap = {
        31101: 1, 31102: 1, 31103: 1, 31104: 1, 31105: 1, 31106: 1, 31107: 1, 31108: 1, 31109: 1, 31110: 1, 31111: 1, 31112: 1, 31113: 1, 31114: 1, 31115: 1,
        31116: 2, 31117: 2, 31118: 2, 31119: 2, 31120: 2, 31121: 2, 31122: 2, 31201: 2, 31202: 2, 31203: 2, 31204: 2, 31205: 2, 31206: 2, 31207: 2, 31208: 2,
        31209: 3, 31210: 3, 31211: 3, 31212: 3, 31213: 3, 31214: 3, 31215: 3, 31216: 3, 31217: 3, 31218: 3, 31219: 3, 31220: 3, 31221: 3, 31222: 3, 31301: 3, 31302: 3, 
        31303: 4, 31304: 4, 31305: 4, 31306: 4, 31307: 4, 31308: 4, 31309: 4, 31310: 4, 31311: 4, 31312: 4, 31313: 4, 31314: 4, 31315: 4, 31316: 4, 31317: 4, 31318: 4, 
        31319: 5, 31320: 5, 31321: 5, 31322: 5, 32101: 5, 32102: 5, 32103: 5, 32104: 5, 32105: 5, 32106: 5, 32107: 5, 32108: 5, 32109: 5, 32110: 5, 32111: 5, 32112: 5, 
        32113: 6, 32114: 6, 32115: 6, 32116: 6, 32117: 6, 32118: 6, 32119: 6, 32120: 6, 32121: 6, 32122: 6, 32201: 6, 32202: 6, 32203: 6, 32204: 6, 32205: 6, 32206: 6
    }