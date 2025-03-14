import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import List, Tuple, TypeVar, cast, Dict

import numpy as np
import torch
from random import shuffle

from src.dataloaders.types import Background, PersonDocument, EncodedDocument
from src.dataloaders.tasks.base import Task

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DecodeOnly(Task):

    def encode_document(self, document: PersonDocument) -> "DecoderDocument":

        prefix_year = [
            ["PLCH0"],
            Background.get_sentence(document.background),
            ["[BOL]"]
        ]

        lifeseq = [prefix_year] + document.lifeseq
        year_lengths = [sum([len(event) for event in year]) for year in lifeseq]
        # event_lengths = [[len(event) for event in year] for year in lifeseq]

        yearseq = [0] + document.yearseq
        assert len(yearseq) == len(year_lengths), document.person_id
        yearseq = list(chain.from_iterable(l * [i] for l, i in zip(year_lengths, yearseq)))

        ageseq = [0] + document.ageseq
        assert len(ageseq) == len(year_lengths), document.person_id
        ageseq = list(chain.from_iterable(l * [i] for l, i in zip(year_lengths, ageseq)))

        # monthseq = [[0]] + document.monthseq
        # for ms, el in zip(monthseq, event_lengths):
        #     assert len(ms) == len(el), document.person_id
        # monthseq = list(chain.from_iterable(l * [i] for e, m in zip(event_lengths, monthseq) for l, i in zip(e, m)))

        # print(lifeseq)
        # print(yearseq)
        # print(monthseq)
        # print("-----------------------------")

        try:
            flat_lifeseq = np.concatenate([np.concatenate([np.array(event) for event in year]) for year in lifeseq])
        except:
            print(document.person_id)
            print( lifeseq)
        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]

        # print(flat_lifeseq[500:550])
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_lifeseq])
        input_sentences = token_ids[:-1]
        yearseq = yearseq[:-1]
        ageseq = ageseq[:-1]

        length = len(input_sentences)

        target_tokens = np.zeros(self.max_length)
        target_tokens[:length] = token_ids[1:].copy()

        # print(flat_lifeseq)
        # print(token_ids)
        # print(length)
        # print("++++++++++++++++++++++++++++++++++++++")

        input_ids = np.zeros((3, self.max_length))
        input_ids[0, :length] = input_sentences
        input_ids[1, :length] = yearseq
        input_ids[2, :length] = ageseq
        # input_ids[3, :length] = segment_expanded

        padding_mask = np.repeat(False, self.max_length)
        padding_mask[:length] = True

        original_sequence = np.zeros(self.max_length)
        original_sequence[:(length + 1)] = token_ids

        sequence_id = np.array(document.person_id)

        return DecoderDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            target_tokens=target_tokens,
            original_sequence=original_sequence,
        )


@dataclass
class DecoderDocument(EncodedDocument[DecodeOnly]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    target_tokens: np.ndarray
    original_sequence: np.ndarray
