from dataclasses import dataclass
from doctest import ELLIPSIS_MARKER
from itertools import chain
from typing import List, TypeVar, cast

import numpy as np
import pandas as pd

from src.dataloaders.types import Background, PersonDocument, EncodedDocument, JSONSerializable
from src.dataloaders.tasks.base import Task

T = TypeVar("T")


@dataclass
class CLS(Task):
    """
    Pulls data from somewhere and uses it for classification?

    .. todo::
        Describe CLS
    """
    # CLS Specific params
    pooled: bool = False
    num_pooled_sep: int = 0


    def __post_init__(self) -> None:
        import warnings
        if self.pooled:
            raise NotImplementedError("Pooled version is not implemented")

    # CLS Specific params
    def get_document(self, person_sentences: pd.DataFrame) -> PersonDocument:
        document = super().get_document(person_sentences)
        target = int(person_sentences.TARGET.iloc[0])
        document.task_info = cast(JSONSerializable, target)  # makes mypy happy

        return document

    def encode_document(self, document: PersonDocument) -> "CLSEncodedDocument":

        prefix_sentence = (
            ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]
        )
        sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = [len(x) for x in sentences]

        def expand(x: List[T]) -> List[T]:
            assert len(x) == len(sentence_lengths)
            return list(
                chain.from_iterable(
                    length * [i] for length, i in zip(sentence_lengths, x)
                )
            )

        abspos_expanded = expand([0] + document.abspos)
        age_expanded = expand([0.0] + document.age)  # todo abs_age vs age?
        assert document.segment is not None
        segment_expanded = expand([1] + document.segment)

        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]

        flat_sentences = np.concatenate(sentences)
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])

        length = len(token_ids)

        input_ids = np.zeros((4, self.max_length))
        input_ids[0, :length] = token_ids
        input_ids[1, :length] = abspos_expanded
        input_ids[2, :length] = age_expanded
        input_ids[3, :length] = segment_expanded

        padding_mask = np.repeat(False, self.max_length)
        padding_mask[:length] = True

        original_sequence = np.zeros(self.max_length)
        original_sequence[:length] = token_ids

        target = np.array(document.task_info).astype(np.float32)

        sequence_id = np.array(document.person_id)

        if self.pooled:
            sep_pos = self.extract_sep_positions(token_ids)
        else:
            sep_pos = np.array([0])


        return CLSEncodedDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            target=target,
            sep_pos=sep_pos,
            original_sequence=original_sequence,
        )

    def extract_sep_positions(self, token_ids: np.ndarray) -> np.ndarray:

        token2index = self.datamodule.vocabulary.token2index
        sep_id = token2index["[SEP]"]

        MAX_LEN = self.num_pooled_sep 
        _sep_pos = np.where(token_ids == sep_id)[0]
        sep_pos = np.zeros(MAX_LEN)

        if len(_sep_pos) >= MAX_LEN:
            offset = len(_sep_pos) - MAX_LEN
            _sep_pos = _sep_pos[offset:]

        sep_pos[: len(_sep_pos)] = _sep_pos
        return sep_pos


@dataclass
class CLSEncodedDocument(EncodedDocument[CLS]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    target: np.ndarray
    sep_pos: np.ndarray
    original_sequence: np.ndarray
