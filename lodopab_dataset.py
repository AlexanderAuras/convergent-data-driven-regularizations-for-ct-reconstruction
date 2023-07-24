import enum
import hashlib
import logging
import pathlib
import typing
import urllib.request
import zipfile

import h5py
import numpy as np
import numpy.typing
import torch.utils.data
import tqdm


class LoDoPaBDataset(torch.utils.data.Dataset[typing.Tuple[torch.Tensor,torch.Tensor,int]]):
    class Subset(enum.Enum):
        TRAIN = "train"
        VAL = "validation"
        TEST = "test"

    def __init__(self, 
                 path: str|pathlib.Path, 
                 subset: Subset, 
                 extracted: bool=False, 
                 transform: typing.Callable[[torch.Tensor],torch.Tensor]=lambda x: x, 
                 target_transform: typing.Callable[[torch.Tensor],torch.Tensor]=lambda x: x, 
                 download: bool=False) -> None:
        super().__init__()
        self.__transform = transform
        self.__target_transform = target_transform
        self.__subset = subset
        self.__extracted = extracted

        if isinstance(path, str):
            path = pathlib.Path(path)
        self.__base_path = path.joinpath("LoDoPaB")

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
        
        logger.debug("Loading patient ids...")
        with open(self.__base_path.joinpath("patient_ids_rand_train.csv").resolve()) as file:
            self.__patient_ids_train = torch.from_numpy(np.fromiter(map(int, file.readlines()), np.int_))
        with open(self.__base_path.joinpath("patient_ids_rand_validation.csv").resolve()) as file:
            self.__patient_ids_validation = torch.from_numpy(np.fromiter(map(int, file.readlines()), np.int_))
        with open(self.__base_path.joinpath("patient_ids_rand_test.csv").resolve()) as file:
            self.__patient_ids_test = torch.from_numpy(np.fromiter(map(int, file.readlines()), np.int_))
        logger.debug("Patient ids loaded")
        

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
        file_infos = {} if self.__extracted else LoDoPaBDataset._FILE_INFOS
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
        for file_name, file_info in LoDoPaBDataset._FILE_INFOS.items():
            logger.debug(f"    Downloading {file_name}...")
            file_path = self.__base_path.joinpath(file_name)
            self.__download_file(file_info.url, file_path)

    
    def __extract(self) -> None:
        logger = logging.getLogger(__name__)
        #Unzip
        logger.debug(f"    Unzipping files...")
        for file_name in LoDoPaBDataset._FILE_INFOS.keys():
            if not file_name.endswith(".zip"):
                continue
            logger.debug(f"    Extracting {file_name}...")
            file_path = self.__base_path.joinpath(file_name)
            with zipfile.ZipFile(file_path.resolve(), "r") as zip_file:
                zip_file.extractall(self.__base_path)
            file_path.unlink()
        logger.debug(f"    Files unzipped")
        #Extracting ".npy-frames" from HDF5 files to increase access speed
        logger.debug(f"    Converting HDF5 files...")
        for file in self.__base_path.iterdir():
            if file.suffix == ".hdf5":
                with h5py.File(file.resolve(), "r") as hdf5_file:
                    for i in range(typing.cast(dict[str,numpy.typing.NDArray[np.float_]], hdf5_file)["data"].shape[0]):
                        np.save(file.parent.joinpath(file.stem+"-"+str(i)), typing.cast(dict[str,numpy.typing.NDArray[np.float_]], hdf5_file)["data"][i,...])
                file.unlink()
        logger.debug(f"    Converting done")


    def __len__(self) -> int:
        return {
            LoDoPaBDataset.Subset.TRAIN: 35820,
            LoDoPaBDataset.Subset.VAL: 3522,
            LoDoPaBDataset.Subset.TEST: 3553,
        }[self.__subset]
    

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor,torch.Tensor,int]:
        with torch.no_grad():
            if not self.__extracted:
                with zipfile.ZipFile(self.__base_path.joinpath(f"observation_{self.__subset.value}.zip").resolve(), "r") as zip_file:
                    with zip_file.open(f"observation_{self.__subset.value}_{idx//128:03}.hdf5") as file:
                        with h5py.File(file, "r") as hdf5_file:
                            observation = torch.from_numpy(typing.cast(dict[str, numpy.typing.NDArray[np.float_]], hdf5_file)["data"][idx%128,...]).unsqueeze(0)
            else:
                observation = torch.from_numpy(np.load(self.__base_path.joinpath("loose", f"observation_{self.__subset.value}_{idx//128:03}-{idx%128}.npy"))).unsqueeze(0)

            if not self.__extracted:
                with zipfile.ZipFile(self.__base_path.joinpath(f"ground_truth_{self.__subset.value}.zip").resolve(), "r") as zip_file:
                    with zip_file.open(f"ground_truth_{self.__subset.value}_{idx//128:03}.hdf5") as file:
                        with h5py.File(file, "r") as hdf5_file:
                            ground_truth = torch.from_numpy(typing.cast(dict[str, numpy.typing.NDArray[np.float_]], hdf5_file)["data"][idx%128,...]).unsqueeze(0)
            else:
                ground_truth = torch.from_numpy(np.load(self.__base_path.joinpath("loose", f"ground_truth_{self.__subset.value}_{idx//128:03}-{idx%128}.npy"))).unsqueeze(0)
            
        patient_id = {
            LoDoPaBDataset.Subset.TRAIN: self.__patient_ids_train,
            LoDoPaBDataset.Subset.VAL: self.__patient_ids_validation,
            LoDoPaBDataset.Subset.TEST: self.__patient_ids_test,
        }[self.__subset][idx]
        return self.__transform(observation), self.__target_transform(ground_truth), int(patient_id.item())
    

    _LoDoPaBFileInfo = typing.NamedTuple("LoDoPaBFileInfo", md5=str, url=str)
    _FILE_INFOS = {
        "patient_ids_rand_train.csv":      _LoDoPaBFileInfo(md5="4fee01b076920f85fc92e0de774dc277", url="https://zenodo.org/record/3384092/files/patient_ids_rand_train.csv?download=1"),
        "patient_ids_rand_validation.csv": _LoDoPaBFileInfo(md5="a387e619074f49573ae376c60a948db4", url="https://zenodo.org/record/3384092/files/patient_ids_rand_validation.csv?download=1"),
        "patient_ids_rand_test.csv":       _LoDoPaBFileInfo(md5="e86068312ad8a039e03f7f929352f7fd", url="https://zenodo.org/record/3384092/files/patient_ids_rand_test.csv?download=1"),
        "ground_truth_train.zip":          _LoDoPaBFileInfo(md5="f06829ccf2b9bb817abd093ce490b2c7", url="https://zenodo.org/record/3384092/files/ground_truth_train.zip?download=1"),
        "ground_truth_validation.zip":     _LoDoPaBFileInfo(md5="666c36f403734842f14ca8811a63b8f7", url="https://zenodo.org/record/3384092/files/ground_truth_validation.zip?download=1"),
        "ground_truth_test.zip":           _LoDoPaBFileInfo(md5="ecc655767fbe3d40908ca823921f4c7f", url="https://zenodo.org/record/3384092/files/ground_truth_test.zip?download=1"),
        "observation_train.zip":           _LoDoPaBFileInfo(md5="c4fde721ac5862469812de49f98fb0a3", url="https://zenodo.org/record/3384092/files/observation_train.zip?download=1"),
        "observation_validation.zip":      _LoDoPaBFileInfo(md5="3cff406e09c59774912655eb7a72cfcf", url="https://zenodo.org/record/3384092/files/observation_validation.zip?download=1"),
        "observation_test.zip":            _LoDoPaBFileInfo(md5="9ae6b053bb1faa94d573311af8ec67b2", url="https://zenodo.org/record/3384092/files/observation_test.zip?download=1")
    }