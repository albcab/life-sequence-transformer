from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar, Dict

import numpy as np

JSONSerializable = NewType("JSONSerializable", object)

if TYPE_CHECKING:
    from src.dataloaders.tasks.base import Task

_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass
class PersonDocument:
    """Dataclass for defining the complete person document in a structured fashion"""

    person_id: int
    lifeseq: List[List[List[str]]]
    yearseq: List[int]
    ageseq: List[int]
    monthseq: List[List[int]]
    background: Optional["Background"] = None
    shuffled: bool = False
    task_info: Optional[JSONSerializable] = None


@dataclass
class Background:
    """Defines the background information about a person"""

    gender: str
    birth_month: int
    birth_year: int
    area: str

    @staticmethod
    def get_background(x: Optional["Background"], token2index) -> Dict[str, np.ndarray]:
        """Return sequence of tokens corresponding to this person. Implemented as
        classmethod since we can null the background in PersonDocument in case of
        unknown background.
        """

        if x is None:
            tokens = 4 * ["[UNK]"]
            # month = "[UNK]"
            # month = np.array(0.)
            # year = np.array(0.)
        else:
            tokens = [x.gender, x.birth_month, x.birth_year, x.area]
            # month = x.birth_month
            # year = np.array(x.birth_year)

        return {"tokens": np.array([token2index.get(t) for t in tokens])}#, "month": token2index.get(month), "year": year}

    @staticmethod
    def get_sentence(x: Optional["Background"]) -> List[str]:
        if x is None:
            return 4 * ["[UNK]"]
        else:
            return [x.gender, x.birth_month, x.birth_year, x.area]
        

class EncodedDocument(Generic[_TaskT]):
    """Generic class for encoded documents. Each task can then type-hint their
    specific encoding using a dataclass like

    .. code-block ::

        class MyTask:
            def encode_document(x: PersonDocument) -> "MyTaskEncodedDocument":
                return MyTaskEncodedDocument(target=1)

        @dataclass
        class MyTaskEncodedDocument(EncodedDocument[MyTask]):
            target: int

    """
