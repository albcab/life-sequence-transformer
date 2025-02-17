from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, set_puf_intensity, Binned


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
            # Binned("INCOME_MONTH", prefix="INCOME", n_bins=300),
            Binned("INCOME_MONTH_ADJ", prefix="INCOME", n_bins=300),
            "FIRM_SIZE",
            "PTIME",
            "WRK_PROVINCE",
            "WRK_TITLE",
            "ATECO",
            "TIPOEPISODIO",
            "INTENSITY_WORK",
            "INTENSITY_MATERNITY",
            "INTENSITY_SICK"]
    )
    input_csv: Path = DATA_ROOT / "rawdata" / "labour_puf.csv"
    latest_end_date: str = "01/01/2100"

    def __post_init__(self) -> None:
        self._latest_end_date = pd.to_datetime(self.latest_end_date, dayfirst=True)

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

        result = (
            self.indexed()
            .assign(
                TIPOEPISODIO=lambda x: "TIPO_" + x.tipo_episodio,
                ATECO=lambda x: "ATE_" + x.ATECO_SEC,
                FIRM_SIZE=lambda x: "FSIZE_" + x.FIRM_SIZE,
                PTIME=lambda x: x.PTIME.map_partitions(lambda s: s.map(lambda x: "PART_TIME" if x else "FULL_TIME", na_action="ignore"), meta=('PTIME', 'string')),
                WRK_PROVINCE=lambda x: "WRKP_" + x.WRK_PROVINCE,
                WRK_TITLE=lambda x: "WRKT_" + x.WRK_TITLE,
                INTENSITY_WORK=lambda x: x.INTENSITY_WORK.map_partitions(lambda s: s.map(
                    lambda x: set_puf_intensity(x, prefix="WRK"), na_action="ignore"), meta=('INTENSITY_WORK', 'string')),
                INTENSITY_MATERNITY=lambda x: x.INTENSITY_MATERNITY.map_partitions(lambda s: s.map(
                    lambda x: set_puf_intensity(x, prefix="MAT"), na_action="ignore"), meta=('INTENSITY_MATERNITY', 'string')),
                INTENSITY_SICK=lambda x: x.INTENSITY_SICK.map_partitions(lambda s: s.map(
                    lambda x: set_puf_intensity(x, prefix="SIK"), na_action="ignore"), meta=('INTENSITY_SICK', 'string')),
                # INCOME_MONTH=lambda x: x.INCOME_MONTH.where(lambda x: x > 0, pd.NA),
                INCOME_MONTH_ADJ=lambda x: x.INCOME_MONTH_ADJ.where(lambda x: x > 0, pd.NA),
            )
            .drop(columns=["ATECO_SEC", "tipo_episodio"])
            # This is important for performance
            .reset_index().set_index("USER_ID", sorted=True, sort=False, npartition="auto")
            # CHOOSE WHAT IS BEST FOR YOUR DATA
            # .repartition(npartitions=1)#partition_size="50MB")
        )

        assert isinstance(result, dd.DataFrame)
        return result

    # FOR BIG DATA, it is often useful to save intermediate results
    @save_parquet(
       DATA_ROOT / "interim/sources/{self.name}/indexed",
       on_validation_error="recompute",
    )
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

    @save_parquet(
       DATA_ROOT / "interim/sources/{self.name}/parsed",
       on_validation_error="error",
       verify_index=False,
    )
    def parsed(self) -> dd.DataFrame:
        """
        Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps
        """
        
        columns = [
            "USER_ID",
            # "INCOME_MONTH",
            "INCOME_MONTH_ADJ",
            "FIRM_SIZE",
            "PTIME",
            "WRK_PROVINCE",
            "WRK_TITLE",
            "ATECO_SEC",
            "tipo_episodio",
            "START_MONTH",
            "END_MONTH",
            "INTENSITY_WORK",
            "INTENSITY_MATERNITY",
            "INTENSITY_SICK",
        ]
        dtypes = {
            "USER_ID": int,  # Deal with missing values
            # "INCOME_MONTH": float,
            "INCOME_MONTH_ADJ": float,
            "FIRM_SIZE": "string",
            "PTIME": "boolean",
            "WRK_PROVINCE": "string",
            "WRK_TITLE": "string",
            "ATECO_SEC": "string",
            "tipo_episodio": "string",
            "INTENSITY_WORK": float,
            "INTENSITY_MATERNITY": float,
            "INTENSITY_SICK": float,
        }
        drop = ["USER_ID"]#, "ATECO"]

        ddf = dd.read_csv(
            self.input_csv,
            low_memory=False,
            usecols=columns,
            on_bad_lines="error",
            assume_missing=True,
            dtype=dtypes,
            blocksize="512MB",
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
