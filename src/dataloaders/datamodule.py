import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pandas.tseries.offsets import MonthEnd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .tasks.base import Task, collate_encoded_documents
from .sampler import FixedSampler
from .dataset import DocumentDataset, MultiTruncSubset, ShardedDocumentDataset, TruncSubset
from .decorators import save_parquet, save_pickle
from .ops import concat_columns_dask, concat_sorted
from .populations.base import Population
from .serialize import DATA_ROOT, ValidationError, _jsonify
from .sources.base import Field, TokenSource
from .vocabulary import Vocabulary

log = logging.getLogger(__name__)


def compute_age(date: pd.Series, birthday: pd.Series) -> pd.Series:
    age = date.dt.year - birthday.year
    # type: ignore
    # age -= (date + MonthEnd(1)).dt.day_of_year < birthday.dt.day_of_year
    # Still leaves some inconstistent birthdays (mainly due to 1-off errors i think)
    return age


# Not quite true, but it is not trivial to do recursive types i think
JSONSerializable = Any


@dataclass
class Corpus:
    """
    Provides a corpus for the specified population with tokens for the specified
    sources. Splits the data into training, validation and testing partition according
    to the population splits.

    .. todo::
        consider renaming the reference_date and threshold parameters to something more
        meaningful

    :param sources: List of token sources from which to generate sentences
    :param population: Cohort to generate sentences for.

    :param reference_date: the day from which we can calculate the ABSOLUTE POSITION
    :param threshold: the day at which we want to stop or cut the sequence.

    """

    name: str

    sources: List[TokenSource]
    population: Population

    reference_year: int = (1986 - 1) ##SHOULD PROBABLY CALCULATE USING THE OBSERVED DATA, SO THAT THEY ARE ALL > 0
    reference_month: int = 0
    threshold_year: int = 2019
    threshold_month: int = 12

    # def __post_init__(self) -> None:

        # self._reference_date = pd.to_datetime(self.reference_date)
        # self._threshold = pd.to_datetime(self.threshold)

    @save_parquet(
        DATA_ROOT / "processed/corpus/{self.name}/sentences/{split}",
        on_validation_error="recompute",
        verify_index=False,
    )
    def combined_sentences(self, split: str) -> dd.DataFrame:
        """Combines the sentences from each source. Filters the data to only consist
        of sentences for the given :obj:`split`.

        :param split: Data split to return sentences for.

        :return:

            A :class:`dask.dataframe.DataFrame` object with the following columns

            * user_id (Index column) - The user ids.

            * start_date - Date of sentence as number of days since
              :attr:`self.reference_date`

            * sentence - The sentence / life-sentence.

            * age - Is calculated bases on the birthday of each person. If the sentences
              already have an AGE columns, this is used instead.

            * sex - The sex as specified in the population data

            * after_threshold - a boolean column, indicating whether an event is efter
              :attr:`self.threshold`.

            * Any additional columns from the population data is carried over as well.

        """
        assert self.reference_month == 0

        population: pd.DataFrame = self.population.population()
        # self.reference_year = int(population.BIRTHDAY_YEAR.min() - 1)
        data_split = getattr(self.population.data_split(), split)
        ### RESETTING INDEX TO AVOID ERRORS OF SORTED DIVISIONS
        sentences_parts = [self.sentences(s).reset_index().set_index("USER_ID", sort=False, sorted=True) 
                           for s in self.sources]
        combined_sentences = concat_sorted(
            [sp.loc[lambda x: x.index.isin(data_split)] for sp in sentences_parts],
            columns=["START_MONTH"],
        ).join(population)

        # combined_sentences["BIRTHDAY_MONTH"] = combined_sentences["BIRTHDAY_MONTH"].apply(
        #     lambda x: pd.to_datetime(x, errors='coerce'), meta=('BIRTHDAY_MONTH', 'datetime64[ns]'))

        combined_sentences["AGE"] = combined_sentences["START_MONTH"].dt.year - combined_sentences["BIRTHDAY_YEAR"]
        #BIRTHDAY_YEAR should not be standardized
        # combined_sentences["BIRTHDAY_YEAR"] = combined_sentences["BIRTHDAY_MONTH"].dt.year
        combined_sentences["BIRTHDAY_MONTH"] = combined_sentences["BIRTHDAY_MONTH"] - self.reference_month

        combined_sentences["START_YEAR"] = combined_sentences["START_MONTH"].dt.year - self.reference_year
        combined_sentences["START_MONTH"] = combined_sentences["START_MONTH"].dt.month - self.reference_month

        combined_sentences["END_YEAR"] = combined_sentences["END_MONTH"].dt.year - self.reference_year
        combined_sentences["END_MONTH"] = combined_sentences["END_MONTH"].dt.month - self.reference_month

        combined_sentences["ALIVE"] = combined_sentences["aa_morte"].isna()

        combined_sentences["THRESHOLD_YEAR"] = combined_sentences["aa_morte"].astype("Int16") - self.reference_year
        combined_sentences["THRESHOLD_YEAR"] = combined_sentences["THRESHOLD_YEAR"].where(
            ~combined_sentences["ALIVE"], self.threshold_year - self.reference_year)
        
        combined_sentences["THRESHOLD_MONTH"] = 12 - self.reference_month
        combined_sentences["THRESHOLD_MONTH"] = combined_sentences["THRESHOLD_MONTH"].where(
            ~combined_sentences["ALIVE"], self.threshold_month - self.reference_month)

        combined_sentences = combined_sentences.drop(columns=['aa_morte'])

        # # Date as days from reference date <- maybe move into task

        # combined_sentences["RECORD_DATE"] = (
        #     combined_sentences.RECORD_DATE - self._reference_date
        # ).dt.days.astype(int)

        # DASK SPECIFIC
        combined_sentences = combined_sentences.reset_index().set_index(
            "USER_ID", sorted=True)#.repartition(partition_size='50MB')

        assert isinstance(combined_sentences, dd.DataFrame)

        return combined_sentences

    # for big datasets you might record intermediate steps
    @save_parquet(
       DATA_ROOT / "interim/corpus/{self.name}/sentences_{source.name}",
       on_validation_error="recompute",
       verify_index=False,
    )
    def sentences(self, source: TokenSource) -> dd.DataFrame:
        """Returns the sentences from :obj:`source`, ie all the fields in
        :attr:`source.fields` in the transformed tokenized data concatenated as strings.
        """
        tokenized = self.tokenized_and_transformed(source)
        # tokenized['name'] = source.name.upper()
        field_labels = source.field_labels()

        import pandas.api.types as ptypes

        for field in field_labels:
            is_string = ptypes.is_string_dtype(tokenized[field].dtype)
            is_known_cat = (
                ptypes.is_categorical_dtype(tokenized[field].dtype)
                and tokenized[field].cat.known
            )
            assert is_string or is_known_cat

        cols = ["START_MONTH", "END_MONTH", "SENTENCE"]
        # if "AGE" in tokenized.columns:
        #     cols.append("AGE")

        # It is a bit akwkard that we join, then split right after.
        # However it is easier to deal with strings, I think
        sentences = tokenized.astype({x: "string" for x in field_labels}).assign(
            SENTENCE=concat_columns_dask(tokenized, columns=list(field_labels))
        )[cols]

        assert isinstance(sentences, dd.DataFrame)

        return sentences

    @save_parquet(
        DATA_ROOT
        / "interim/corpus/{self.name}/tokenized_and_transformed/{source.name}",
        on_validation_error="recompute",
        verify_index=False,
    )
    def tokenized_and_transformed(self, source: TokenSource) -> dd.DataFrame:
        """Returns the tokenized data for :obj:`source`, with any
        :class:`src.sources.base.Field` tranformations applied.
        """

        fields_to_transform = self.fitted_fields(source)
        tokenized = source.tokenized().reset_index().set_index("USER_ID", sort=False, sorted=True)
        for field in fields_to_transform:
            tokenized[field.field_label] = field.transform(
                tokenized[field.field_label])
        assert isinstance(tokenized, dd.DataFrame)
        return tokenized

    @save_pickle(
        DATA_ROOT / "interim/corpus/{self.name}/fitted_fields/{source.name}",
        on_validation_error="recompute",
    )
    def fitted_fields(self, source: TokenSource) -> List[Field]:
        """Fits any :class:`src.sources.base.Field` using the :meth:`fit` method on the
        training data, and saves their state using pickle.
        """
        ids = self.population.data_split().train
        tokenized = source.tokenized().reset_index().set_index("USER_ID", sort=False, sorted=True)
        fields = source.fields
        fields_to_fit = [field for field in fields if isinstance(field, Field)]
        for field in fields_to_fit:
            field.fit(tokenized.loc[lambda x: x.index.isin(ids)])
        return fields_to_fit

    def prepare(self) -> None:
        """Prepares each dataset split"""

        self.combined_sentences("train")
        self.combined_sentences("val")
        self.combined_sentences("test")


@dataclass
class L2VDataModule(pl.LightningDataModule):
    """
    Main life2vec data processing pipeline. The data is generated based on a corpus
    and a task. The generated data is stored in /processed/<corpus>/<task>, with
    subfolders corresponding to each data split. The remaining parameters are given
    to :class:`torch.utils.data.DataLoader`.

    :param corpus: The corpus to generate data from.
    :param vocabulary: Vocabulary to use.
    :param task: Task to generate data for.

    :param batch_size: Batch size
    :param num_workers: Number of data loading workers
    :param persisten_worksers: Whether to persist workers
    :param pin_memory: Whether to pin memory

    """

    # Data components
    corpus: Corpus
    vocabulary: Vocabulary
    task: Task

    # Data loading params
    batch_size: int = 8
    num_workers: int = 2
    persistent_workers: bool = False
    pin_memory: bool = False
    subset: bool = False
    subset_id: bool = 0  # max 2

    def __post_init__(self) -> None:
        super().__init__()
        assert self.name != ""
        self.task.register(self)
        self.processor = None

    @property
    def dataset_root(self) -> Path:
        """Return the dataset root according to the vocabulary and task names"""
        return DATA_ROOT / "processed" / "datasets" / self.vocabulary.name / self.task.name

    def prepare(self) -> None:
        """Calls :meth:`prepare_data` to prepare the data."""
        self.prepare_data()
        self.setup()

    def _arguments(self) -> Dict[str, Any]:
        """Since we dont want to include the data loading parameters, when validating
        the saved datasets with the current parameters, we instead supply the arguments
        of the corpus and task from here."""

        return {
            "corpus": _jsonify(self.corpus),
            "vocabulary": _jsonify(self.vocabulary),
            "task": _jsonify(self.task),
        }

    def prepare_data(self) -> None:
        """Checks whether the data already exists.
        If not, then prepares the corpus and each data split using
        :meth:`prepare_data_split`
        """
        arg_path = self.dataset_root / "_arguments"
        try:
            with open(arg_path, "rb") as f:
                arguments = pickle.load(f)
                print(arg_path)
            if arguments == self._arguments():
                return
            else:
                log.warning("Arguments do not correspond to the recorded ones")
                raise ValidationError
        except (EOFError, FileNotFoundError):
            pass

        log.info("Preparing corpus...")
        self.corpus.prepare()
        log.info("Prepared corpus.")

        log.info("Preparing vocabulary...")
        self.vocabulary.prepare()
        log.info("Prepared vocabulary.")
        log.info("\tVocabulary size: %s" % self.vocabulary.size())

        log.info("Preparing datasets...")
        self.dataset_root.mkdir(exist_ok=True, parents=True)
        dask.compute(
            self.prepare_data_split("train"),
            self.prepare_data_split("val"),
            self.prepare_data_split("test"),
        )
        log.info("Prepared datasets.")

        with open(self.dataset_root / "_arguments", "wb") as f:
            pickle.dump(self._arguments(), f)

    def prepare_data_split(self, split: str) -> dd.Series:
        """Prepares the dataset for some split (train/val/test).

        Loads the combined sentences of the corpus, then for each parquet partition,
        filters according to the split and using pandas group_by, for each USER_ID
        calls the :meth:`get_document` method of :attr:`task` to get the
        user documents. The resulting list of documents then gets saved using
        :class:`src.data_new.dataset.DocumentDataset`.

        :return: Returns a :class:`dask.dataframe.Series` with a single :code:`True`
            for each partition. This is only meant for use with :func:`dask.compute`,
            so that we can apply this step for all splits in parallel.
        """

        data = self.corpus.combined_sentences(split)
        N = data.npartitions

        def process_partition(
            partition: pd.DataFrame, partition_info: Optional[Dict[str, int]] = None
        ) -> bool:
            """Process a single sentence data partition into a
            :class:`src.data_new.dataset.DocumentDataset`
            """

            assert partition_info is not None

            from math import log10

            file_name = str(partition_info["number"]).zfill(
                int(log10(N)) + 1) + ".hdf5"
            path = self.dataset_root / split / file_name
            log.debug(
                "Processing partition %s_%d to %s",
                split,
                partition_info["number"],
                path,
            )

            records = (
                partition.groupby(level="USER_ID")
                .apply(self.task.get_document)
                .to_list()
            )

            DocumentDataset(file=path).save_data(records)

            try:
                if N < 2:
                    pass
                elif partition_info["number"] % (N//10) == 1:
                    log.info(
                        "\t%s out of %s %s partitions completed",
                        partition_info["number"],
                        N,
                        split,
                    )
            except:
                log.warning("Partitioning complited: %s" % split)

            return True

        result = data.map_partitions(process_partition, meta=(None, bool))
        assert isinstance(result, dd.Series)

        return result

    def get_dataset(self, split: str, train_preprocessor: bool = True) -> Dataset:
        """Instantiates the dataset for the split in question using the preprocessor
        from :attr:`task`
        """
        if train_preprocessor:
            preprocessor = self.task.get_preprocessor(
                is_train=split == "train")
        else:
            preprocessor = self.task.get_preprocessor(is_train=split == "val")
        dataset = ShardedDocumentDataset(
            directory=self.dataset_root / split,
            transform=preprocessor,
        )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiates the datasets relevant to the given stage"""
        if stage == "fit" or stage is None:
            self.train = self.get_dataset("train")
            self.val = self.get_dataset("val")
        if stage == "test" or stage is None:
            self.test = self.get_dataset("test")

    # TODO: vvv Consider moving this stuff to the task instead vvv

    def get_dataloader(
        self, dataset: Dataset, shuffle: bool = True, drop_last: bool = True
    ) -> DataLoader:
        """Instantiaties and return a dataloader for the given dataset using the
        parameters of the module"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_encoded_documents,
            generator=torch.Generator(),
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persistent_workers,
            multiprocessing_context='fork' if torch.backends.mps.is_available() else None,
            # "fork" in case you run it on MPS cluster, otherwise, None
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader"""
        if self.subset:
            assert self.subset_id < 3
            log.info("Subset %s" % self.subset_id)
            idx = [i for i in range(len(self.train)) if i %
                   3 == self.subset_id]
            log.info("First ID: %s Total records: %s" % (idx[0], len(idx)))
            self.train = torch.utils.data.Subset(self.train, idx)
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader"""
        return self.get_dataloader(self.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader"""
        return self.get_dataloader(self.test, shuffle=False)

    def single_idx_dataloader(self, idx, reps, trunc_years=1, split="test"):
        if reps % self.batch_size != 0:
            log.warning(f"Reps of idx not divisible by batch_size, last batch will be smaller for idx={idx}.")
        sdataset = TruncSubset(self.get_dataset(split), idx, reps, trunc_years)
        return self.get_dataloader(sdataset, shuffle=False, drop_last=False)

    def multi_idx_dataloader(self, idxs, reps, trunc_years, split="test"):
        dataset = MultiTruncSubset(
            self.get_dataset(split), idxs, reps, trunc_years
        )
        return self.get_dataloader(dataset, shuffle=False, drop_last=False)


class CLSDataModule(L2VDataModule):

    def get_train_weights(self) -> torch.Tensor:
        ids = self.corpus.population.data_split().train
        population = self.corpus.population.population()
        targets = population.loc[ids]["TARGET"].values

        class_weights = 1. / np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        class_weights = [i/sum(class_weights) for i in class_weights]
        return torch.tensor([class_weights[t] for t in targets]).reshape(-1)

    def get_ordered_indexes(self, split: str) -> torch.LongTensor:
        if split == "val":
            ids = self.corpus.population.data_split().val
        elif split == "test": 
            ids = self.corpus.population.data_split().test
        else:
            raise ValueError()

        population = self.corpus.population.population()
        targets = population.loc[ids]["TARGET"].values
        pos_idx = np.where(targets == 1)[0]
        neg_idx = np.where(targets == 0)[0]
        np.random.seed(0)
        np.random.shuffle(neg_idx)
        return torch.LongTensor(np.hstack([pos_idx, neg_idx]))

    def get_fixed_dataloader(self, dataset):
        indices = self.__ordered_subset_indexes__(dataset=dataset)
        sampler = FixedSampler(indices = indices)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler = sampler,
            shuffle = False,
            persistent_workers=self.persistent_workers
        )   

    def get_weighted_dataloader(self, dataset, weights, replacement: bool) -> DataLoader:
        """Weighted Random Sampling (mostly to upsample the minority class)"""
        sampler = WeightedRandomSampler(
            weights=weights, 
            num_samples = weights.shape[0],
            replacement = replacement,
            generator = torch.Generator())

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_documents,
            sampler=sampler, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def get_fixed_dataloader(self, dataset, indices) -> DataLoader:
        sampler = FixedSampler(indices = indices)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_encoded_documents,
            sampler = sampler,
            shuffle = False,
            persistent_workers=self.persistent_workers
        )   

    def get_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Instantiaties and return a dataloader for the given dataset using the
        parameters of the module"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_encoded_documents,
            generator=torch.Generator(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader"""
        sample_weights = self.get_train_weights()
        return self.get_weighted_dataloader(self.train, weights=sample_weights, replacement=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader"""
        indices = self.get_ordered_indexes(split = "val")
        return self.get_fixed_dataloader(self.val, indices)

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader"""
        indices = self.get_ordered_indexes(split = "test")
        return self.get_fixed_dataloader(self.test, indices)


from transformers import PreTrainedTokenizerBase

@dataclass
class HFCausalL2VDataModule(L2VDataModule):
    tokenizer: PreTrainedTokenizerBase = None

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.tokenizer is not None

        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                # "[CLS]",
                # "[SEP]",
                "[BOL]",
                "[EOL]",
                "[EOY]",
                "[PLCH0]",
                "[UNK]",
            ]
        })

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _decode_l2v_ids(self, ids) -> str:
        index2token = self.vocabulary.index2token

        # return " [SEP] ".join(   #we could use separation token
        return " ".join(
            index2token[int(idx)]
            for idx in ids
            if int(idx) != 0 #remove padding
        )

    def collate_hf(self, documents):
        texts = [
            self._decode_l2v_ids(doc.original_sequence)
            for doc in documents
        ]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels

        return batch

    def get_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.collate_hf,
            generator=torch.Generator(),
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )
