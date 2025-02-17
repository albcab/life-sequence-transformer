from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, set_monthly_intensity


@dataclass
class LaborMonthSource(TokenSource):
    """This generates tokens based on information from the monthly Labor dataset.

    :param name: The name of the dataset/source.
    :param fields: The columns to include in the dataset (in case you need to bin the continue variables use 'Binned' class).
    :param input_csv: CSV file from which to load the Synthetic Labor Dataset.
    """

    name: str = "labor"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "OCCUPATION",
            "ATECO",
            "TIPOEPISODIO",
            "INTENSITY"]
    )
    subsidized: bool = False  
    input_csv: Path = DATA_ROOT / "rawdata" / "labor_month.csv"
    latest_end_date: str = "01/01/2100"

    def __post_init__(self) -> None:
        self._latest_end_date = pd.to_datetime(self.latest_end_date, dayfirst=True)
        if not self.subsidized:
            self.fields = ["OCCUPATION", "ATECO", "INTENSITY"]
        else:
            self.fields = ["TIPOEPISODIO", "INTENSITY"]
            self.name = "sub" + self.name

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
        # You might want to run the verification (aka that indexes are indeed sorted)
        verify_index=True,
    )
    def tokenized(self) -> dd.DataFrame:
        """
        Loads the indexed data, then tokenizes it.
        Do some preprocessing on the raw data
        """

        if not self.subsidized:
            result = (
                self.indexed()
                .assign(
                    ATECO=lambda x: "ATE_" + x.ATECO,
                    OCCUPATION=lambda x: "OCC_" + x.cp2011,
                    INTENSITY=lambda x: x.spell_duration.apply(set_monthly_intensity, meta=('INTENSITY', 'string')),
                )
                .drop(columns=[
                    "cp2011",
                    "spell_duration",
                ])
                # This is important for performance
                .reset_index().set_index("USER_ID", sorted=True, sort=False, npartition="auto")
                # CHOOSE WHAT IS BEST FOR YOUR DATA
                # .repartition(npartitions=1)#partition_size="50MB")
            )
        else:
            result = (
                self.indexed()
                .assign(
                    TIPOEPISODIO=lambda x: "TIPO_" + x.tipo_episodio,
                    INTENSITY=lambda x: x.spell_duration.apply(set_monthly_intensity, meta=('INTENSITY', 'string')),
                )
                .drop(columns=[
                    "tipo_episodio",
                    "spell_duration",
                ])
                .reset_index().set_index("USER_ID", sorted=True, sort=False, npartition="auto")
            )

        assert isinstance(result, dd.DataFrame)
        return result

    # FOR BIG DATA, it is often useful to save intermediate results
    # @save_parquet(
    #    DATA_ROOT / "interim/sources/{self.name}/indexed",
    #    on_validation_error="recompute",
    # )
    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = (self.parsed()
                  .pipe(lambda x: x.categorize(x.select_dtypes("string").columns))
                  .set_index("USER_ID")
                  .pipe(sort_partitions, columns=["START_MONTH"])
                  .pipe(lambda x: x.astype(
                      {k: "string" for k in x.select_dtypes(
                          "category").columns}
                  )
        )
        )

        assert isinstance(result, dd.DataFrame)
        return result

    # @save_parquet(
    #    DATA_ROOT / "interim/sources/{self.name}/parsed",
    #    on_validation_error="error",
    #    verify_index=False,
    # )
    def parsed(self) -> dd.DataFrame:
        """
        Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps
        """
        
        if not self.subsidized:
            columns = [
                "USER_ID",
                "cp2011",
                "ATECO",
                "START_MONTH",
                "END_MONTH",
                "spell_duration",
            ]
            dtypes = {
                "USER_ID": int,  # Deal with missing values
                "cp2011": "string",
                "ATECO": "string",
                "spell_duration": int,
            }
            drop = ["cp2011", "ATECO"]

        else:
            columns = [
                "USER_ID",
                "tipo_episodio",
                "START_MONTH",
                "END_MONTH",
                "spell_duration",
            ]
            dtypes = {
                "USER_ID": int,
                "tipo_episodio": "string",
                "spell_duration": int,
            }
            drop = ["tipo_episodio"]

        ddf = dd.read_csv(
            self.input_csv,
            low_memory=False,
            usecols=columns,
            on_bad_lines="error",
            assume_missing=True,
            dtype=dtypes,
            blocksize="256MB",
            # blocksize=None,
        )

        ddf = (
            ddf.dropna(subset=drop)
            .assign(
                USER_ID=lambda x: x.USER_ID.astype(int),
                START_MONTH=lambda x: dd.to_datetime(
                    x.START_MONTH,
                    format="%Y-%m",
                    errors="coerce",
                ),
                END_MONTH=lambda x: dd.to_datetime(
                    x.END_MONTH,
                    format="%Y-%m",
                    errors="coerce",
                ),
            )
            .loc[lambda x: x.START_MONTH <= self._latest_end_date]
        )
        assert isinstance(ddf, dd.DataFrame)
        return ddf
