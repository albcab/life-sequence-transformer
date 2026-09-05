import json as json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Sequence

import h5py
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from .tasks.base import Task

from .types import Background, PersonDocument, EncodedDocument

log = logging.getLogger(__name__)

T1 = TypeVar("T1", bound=Any)
T2 = TypeVar("T2", bound=Any)
_T_co = TypeVar("_T_co", covariant=True)


class HDF5Dataset(Dataset, Generic[T1, T2]):
    """
    Stores and loads compressed records of arbitrary json data using the hdf5 format.
    The data is stored in a zero-padded numpy string array on disk. Since we are using
    compression, the zero-padding does not increase the storage reguired significantly.

    This class should be subclasses and the serialize/deserialize methods implenented.

    We allow for a transform argument, mapping the records stored in the dataset to
    be transformed to something else, for instance in the case of augmentation.

    """

    def __init__(
        self,
        file: Path,
        encoding: str = "utf-8",
        transform: Optional[Callable[[T1], T2]] = None,
    ):

        self.file = Path(file)
        assert self.file.suffix == ".hdf5"
        self.encoding = encoding
        self.transform = transform

    def _transform(self, x: T1) -> T2:
        """Apply the transform if defined."""
        if self.transform is None:
            return x
        else:
            return self.transform(x)

    def deserialize(self, x: str) -> T1:
        """Instantiate the record from the json string"""
        raise NotImplementedError

    def serialize(self, x: T1) -> str:
        """Serialize the instantiated record to a json string"""
        return json.dumps(x, separators=(",", ":"))

    def __len__(self) -> int:
        """Return the number of records"""
        with h5py.File(self.file, "r") as f:  # TODO: set "r" attribute
            return len(f["data"])

    def __getitem__(self, idx: int) -> T2:
        """Retrieve a record string from the hdf5 array, apply the deserialization,
        then finally apply the transform.
        """

        # Sometimes, the OS would throw an error on file access. This is solved by
        # retrying a few times
        retries = 5
        for i in range(retries):
            try:
                with h5py.File(self.file, "r") as f:
                    content = self.deserialize(
                        f["data"][idx].decode(self.encoding))
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(
                    f"Encountered {e} while opening {self.file} at index {i}")
                if i < retries - 1:
                    print("Retrying...")
                    time.sleep(1 + i * 5)
                else:
                    print("Aborting...")
                    raise e

        content = self._transform(content)
        return content

    def save_data(self, data: Iterable[T1]) -> None:
        """Saves records to the dataset"""

        out_array = np.array(
            [self.serialize(x).encode(self.encoding) for x in data])
        self.file.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(self.file, "w") as f:
            f.create_dataset(
                "data", data=out_array, compression="gzip", compression_opts=9
            )


TaskT = TypeVar("TaskT", bound=Task)


class DocumentDataset(HDF5Dataset[PersonDocument, EncodedDocument[TaskT]]):
    """Dataset implementation for storing PersonDocuments as records. The augmentation
    and encoding is done using the transform parameter. Returns the encoded documents.


    :param file:
    :param encoding:
    :param transform:
    """

    def deserialize(self, x: str) -> PersonDocument:
        """Loads the person document from the json data."""
        data = json.loads(x)
        data["background"] = Background(**data["background"])
        return PersonDocument(**data)

    def serialize(self, x: PersonDocument) -> str:
        """Dumps the person document to json data"""
        return json.dumps(asdict(x), separators=(",", ":"))


class ShardedDocumentDataset(ConcatDataset, Generic[TaskT]):
    """Wrapper around :class:`torch.utils.data.ConcatDataset` for combining multiple
    document datasets in the case of sharded data.
    """

    def __init__(
        self,
        directory: Path,
        encoding: str = "utf-8",
        transform: Optional[Callable[[PersonDocument],
                                     EncodedDocument[TaskT]]] = None,
    ):

        self.directory = Path(directory)
        self.encoding = encoding
        self.transform = transform

        datasets = []
        for file_ in sorted(self.directory.glob("*.hdf5")):
            datasets.append(
                DocumentDataset(
                    file_,
                    encoding=self.encoding,
                    transform=self.transform,
                )
            )

        super().__init__(datasets)


class TruncSubset(Dataset[_T_co]):
    r"""
    Subset of a dataset at specified indices with applied truncation.

    WORKS ONLY WITH ONE SINGLE INDEX!!!

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[_T_co]
    idx: int
    reps: int
    trunc_years: int
    sample: _T_co

    def __init__(self, dataset: Dataset[_T_co], idx: int, reps: int, trunc_years: int) -> None:
        self.dataset = dataset
        self.idx = idx
        self.reps = reps
        self.trunc_years = trunc_years
        # if idx > 5e6:
        #     for i in range(len(dataset) - 1, -1, -1):
        #         sample = dataset[i]
        #         if sample.sequence_id == idx:
        #             break
        # else:
        #     for sample in dataset:
        #         if sample.sequence_id == idx:
        #             break

        directory = getattr(dataset, "directory", None).name
        cache_path = directory + "_idx_cache.json"

        try:
            with open(cache_path, "r") as f:
                index = json.load(f)

        except FileNotFoundError:
            index = {}
            for i, sample in enumerate(dataset):
                index[str(sample.sequence_id)] = i

            with open(cache_path, "w") as f:
                json.dump(index, f)

        if str(idx) not in index:
            raise ValueError(f"Sequence ID {idx} not found in dataset.")
            
        sample_idx = index[str(idx)]
        sample = dataset[sample_idx]

        self.sample = self.truncate_fn(sample)

    def __getitem__(self, idx: int):
        if isinstance(idx, list):
            raise "Why is idx a list?"
        return self.sample

    def __len__(self):
        return self.reps
    
    def truncate_fn(self, _user_data):
        return truncate_sample(_user_data, self.trunc_years)


def truncate_sample(_user_data, trunc_years: int):
    user_data = asdict(_user_data)

    input_ids = user_data.pop('input_ids').copy()
    padding_mask = user_data.pop('padding_mask').copy()

    last_idx = int(padding_mask.sum().item()) - 1
    last_year = input_ids[1, last_idx].item()

    mask = (input_ids[1, :] > (last_year - trunc_years)) & (padding_mask == 1)
    input_ids[:, mask] = 0
    padding_mask[mask] = 0

    return replace(_user_data, input_ids=input_ids, padding_mask=padding_mask)


class MultiTruncSubset(Dataset[_T_co]):
    """Lazily load and truncate several sequence IDs.

    Samples are interleaved by repetition so a sequential DataLoader batch contains
    different people instead of repeated copies of one person.
    """

    def __init__(
        self,
        dataset: Dataset[_T_co],
        idxs: Sequence[int],
        reps: int,
        trunc_years: Sequence[int],
    ) -> None:
        if len(idxs) != len(trunc_years):
            raise ValueError("idxs and trunc_years must have the same length")
        if reps < 1:
            raise ValueError("reps must be at least 1")

        self.dataset = dataset
        self.items = [
            (int(idx), int(years))
            for _ in range(reps)
            for idx, years in zip(idxs, trunc_years)
        ]
        self._sample_cache = {}

        directory = getattr(dataset, "directory", None)
        if directory is None:
            raise ValueError("dataset must expose its source directory")
        cache_path = f"{directory.name}_idx_cache.json"

        try:
            with open(cache_path, "r") as f:
                self.index = json.load(f)
        except FileNotFoundError:
            self.index = {}
            for i, sample in enumerate(dataset):
                self.index[str(sample.sequence_id)] = i
            with open(cache_path, "w") as f:
                json.dump(self.index, f)

        missing = [idx for idx in idxs if str(idx) not in self.index]
        if missing:
            raise ValueError(f"Sequence IDs not found in dataset: {missing[:10]}")

    def __getitem__(self, item: int):
        idx, trunc_years = self.items[item]
        cache_key = (idx, trunc_years)
        if cache_key not in self._sample_cache:
            sample = self.dataset[self.index[str(idx)]]
            self._sample_cache[cache_key] = truncate_sample(sample, trunc_years)
        return self._sample_cache[cache_key]

    def __len__(self):
        return len(self.items)
