import random
from dataclasses import asdict, dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Dict, List, TypeVar

import numpy as np
import pandas as pd
import torch
from functools import partial


from src.dataloaders.augment import (
    shuffle_events,
    shuffle_tokens,
    drop_tokens,
)
from src.dataloaders.types import Background, PersonDocument

if TYPE_CHECKING:
    from src.dataloaders.datamodule import L2VDataModule
    from src.dataloaders.types import EncodedDocument


def collate_encoded_documents(
    batch: List["EncodedDocument"],
) -> Dict[str, torch.Tensor]:
    dicts = [asdict(x) for x in batch]
    return torch.utils.data.default_collate(dicts)  # type: ignore


_TaskT = TypeVar("_TaskT", bound="Task")


def preprocessor(task: _TaskT, x: PersonDocument, is_train: bool) -> "EncodedDocument[_TaskT]":
    x = task.augment_document(x, is_train=is_train)
    x = task.clip_document(x)
    return task.encode_document(x)


@dataclass
class Task:
    """
    Base class for processing :class:`src.data.types.PersonDocument` objects
    for various ML tasks. Includes default implementations for clipping and augmenting
    documents.
    In order to implement a new task, we have to implement the :meth:`encode_sequence`,
    which defines how the documents should be encoded for a specific task, ie. what
    input the forward method of the model expects for the task in question.

    Also defines a default implementation for pulling :class:`PersonDocument` objects
    out of the sentence data provided by the :class:`src.data.Corpus`. Task
    implementations can extend this method and save task-specific information in the
    :attr:`task_info` field

    :param name: The name of the task.
    :param max_length: The maximum length of the encoded documents
    :param no_sep: If True, don't include the [SEP] token between sentences.

    :param p_timecut:
    :param p_resample:
    :param p_abspos_noise:
    :param p_hide_background:
    :param p_shuffle_sentences:

    :param shuffle_within_sentences:


    .. todo::
        Confirm this is what no_sep is actually for since this has always been False as
        far as I (Søren) know

    """

    # General
    name: str
    max_length: int

    # Augmentation
    p_sequence_reverse_events: float = 0.0
    p_sequence_shuffle_events: float = 0.0
    p_sequence_shuffle_tokens: float = 0.0
    p_sentence_drop_tokens: float = 0.0

    # Task specific
    ...

    def register(self, datamodule: "L2VDataModule") -> None:
        self.datamodule = datamodule

    def get_preprocessor(
        self: _TaskT, is_train: bool
    ) -> Callable[[PersonDocument], "EncodedDocument[_TaskT]"]:

        return partial(preprocessor, self, is_train=is_train)

    def augment_document(
        self, document: PersonDocument, is_train: bool
    ) -> PersonDocument:

        if is_train:

            # AUGMENTATION WITH NOISE
            p = np.random.uniform(low=0.0, high=1.0, size=[4])

            # 1. REVERSE EVENTS
            if p[0] < self.p_sequence_reverse_events:
                document = shuffle_events(document, reverse=True)
            # 2. SHUFFLE EVENTS
            if p[1] < self.p_sequence_shuffle_events:
                document = shuffle_events(document, reverse=False)
            # 3. SHUFFLE CHARACTERISTICS
            if p[2] < self.p_sequence_shuffle_tokens:
                document = shuffle_tokens(document)
            # 4. DROP TOKENS FROM THE SEQUENCE
            if p[3] < self.p_sentence_drop_tokens:
                document = drop_tokens(document, p=self.p_sentence_drop_tokens)

        return document

    def clip_document(self, document: PersonDocument) -> PersonDocument:

        lengths = [sum([len(events) for events in year]) for year in document.lifeseq]
        clip_idx = None
        for i, x in enumerate(accumulate(reversed(lengths))): #delete whole years only, from last to first
            if x >= self.max_length:
                clip_idx = i
                break

        if clip_idx is not None:
            document.lifeseq = document.lifeseq[-clip_idx:]
            document.yearseq = document.yearseq[-clip_idx:]
            document.ageseq = document.ageseq[-clip_idx:]
            document.monthseq = document.monthseq[-clip_idx:]

        return document

    def encode_document(
        self: _TaskT, document: PersonDocument
    ) -> "EncodedDocument[_TaskT]":
        raise NotImplementedError

    def get_document(self, person_sentences: pd.DataFrame) -> PersonDocument:

        person_id = person_sentences.name
        sentences = [x.split(" ") for x in person_sentences.SENTENCE]
        start_month = person_sentences.START_MONTH.to_list()
        start_year = person_sentences.START_YEAR.to_list()
        end_month = person_sentences.END_MONTH.to_list()
        end_year = person_sentences.END_YEAR.to_list()

        threshold_year = person_sentences.THRESHOLD_YEAR.iloc[0]
        threshold_month = person_sentences.THRESHOLD_MONTH.iloc[0]
        start_age = person_sentences.AGE.iloc[0]

        first_year = start_year[0]
        yearseq = [year for year in range(first_year, threshold_year + 1)]
        ageseq = [int(start_age + i) for i in range(len(yearseq))]
        
        monthseq = [[] for year in yearseq]
        lifeseq = [[] for year in range(first_year, threshold_year + 1)]
        for sm, sy, em, ey, sentence in zip(start_month, start_year, end_month, end_year, sentences):
            if sy != ey:
                print("One instance of start year != end year for ID=", person_id)
                continue

            if sy > threshold_year:
                continue
            elif sy == threshold_year and sm > threshold_month:
                continue
            elif ey == threshold_year:
                em = min(em, threshold_month)

            lifeseq[ey - first_year].append([f"MONTH_{sm}"] + sentence + [f"DUR_{em - sm + 1}"])
            monthseq[ey - first_year].append(sm)
        for life_year in lifeseq:
            life_year.append(["[EOY]"])
        for month_year in monthseq:
            month_year.append(0)

        # print(person_id)
        # print(sentences)
        # print(years)
        # print(months)
        # print("++++++++++++++++++++++++++++++")

        alive = person_sentences.ALIVE.iloc[0]
        if not alive:
            lifeseq[-1][-1][0] = "[EOL]"

            # print(person_id)
            # print(lifeseq)
            # print(yearseq)
            # print(monthseq)
            # print("-------------------------------")

        if threshold_month < 12: #no EOY when the ending is between year
            lifeseq[-1] = lifeseq[-1][:-1]
            monthseq[-1] = monthseq[-1][:-1]

        # birthday_month = f"MONTH_{person_sentences.BIRTHDAY_MONTH.iloc[0].month}"
        birthday_month = person_sentences.BIRTHDAY_MONTH.iloc[0] #if we do time2vec month in background
        birthday_year = person_sentences.BIRTHDAY_YEAR.iloc[0]
        sex = person_sentences.SEX.iloc[0]
        area = person_sentences.AREA.iloc[0]

        # print(birthday_month)
        # print(type(birthday_month))
        # print(birthday_year)
        # print(type(birthday_year))
        # print(sex)
        # print(area)
        # print("===================================")

        background = Background(
            gender=sex,
            birth_month=int(birthday_month),
            birth_year=int(birthday_year),
            area=area,
        )

        return PersonDocument(
            person_id=person_id,
            lifeseq=lifeseq,
            yearseq=yearseq,
            ageseq=ageseq,
            monthseq=monthseq,
            background=background,
        )
