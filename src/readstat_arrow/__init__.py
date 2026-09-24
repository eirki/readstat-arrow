"""readstat-arrow: read and write SPSS (.sav) and Stata (.dta) files as Apache Arrow tables.

A thin, typed wrapper around the ReadStat C library. Every ``read_*`` function
returns ``(pyarrow.Table, Metadata)``; ``write_*`` takes the same pair.

>>> import readstat_arrow
>>> table, metadata = readstat_arrow.read_sav("survey.sav")
>>> table.num_rows, metadata.variable_labels["q1"], metadata.value_labels["q1"]
"""

from importlib.metadata import version as _version

from readstat_arrow.errors import ReadstatError, ReadstatWarning
from readstat_arrow.metadata import (
    Alignment,
    Code,
    Measure,
    Metadata,
    Missingness,
    MissingRange,
    MissingValues,
    Value,
)
from readstat_arrow.reader import (
    DtaStreamingReader,
    SavStreamingReader,
    open_dta,
    open_sav,
    read_dta,
    read_dta_metadata,
    read_sav,
    read_sav_metadata,
)
from readstat_arrow.writer import (
    DtaWriter,
    SavWriter,
    TextLimitPolicy,
    write_dta,
    write_sav,
)

__all__ = [
    "Alignment",
    "Code",
    "DtaStreamingReader",
    "DtaWriter",
    "Measure",
    "Metadata",
    "MissingRange",
    "MissingValues",
    "Missingness",
    "ReadstatError",
    "ReadstatWarning",
    "SavStreamingReader",
    "SavWriter",
    "TextLimitPolicy",
    "Value",
    "open_dta",
    "open_sav",
    "read_dta",
    "read_dta_metadata",
    "read_sav",
    "read_sav_metadata",
    "write_dta",
    "write_sav",
]

__version__ = _version("readstat-arrow")  # single source of truth: pyproject.toml
