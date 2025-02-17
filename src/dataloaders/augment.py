"""
This script contains the data augmentation functions that alter the input data during the training procedure.
"""
import random
from collections import defaultdict
import numpy as np
from .types import PersonDocument


def drop_tokens(document: PersonDocument, p) -> PersonDocument:
    """Randomly drop tokens from a sentence, only characteristics of the event"""
    for year in document.lifeseq:
        for event in year:
            if len(event) < 3:
                continue
            if np.random.rand(1) < p:
                event.pop(random.randrange(2, len(event)))

    return document

def shuffle_tokens(document: PersonDocument) -> PersonDocument:
    """Shuffle the characteristics of the event"""
    for year in document.lifeseq:
        for event in year:
            start = event[:2]
            end = event[-1:]
            middle = event[2:-1]
            random.shuffle(middle)
            event = start + middle + end

    return document

def shuffle_events(document: PersonDocument, reverse=False) -> PersonDocument:
    """Shuffle/reverse events that happen congrunetly"""
    for year, month in zip(document.lifeseq, document.monthseq):

        grouped_indices = defaultdict(list)
        for idx, m in enumerate(month):
            grouped_indices[m].append(idx)

        for indices in grouped_indices.values():
            original = indices.copy()
            if reverse:
                indices.reverse()
            else:
                random.shuffle(indices)
            temp = [year[i] for i in indices]
            for original_idx, new_idx in zip(original, range(len(indices))):
                year[original_idx] = temp[new_idx]

    return document
