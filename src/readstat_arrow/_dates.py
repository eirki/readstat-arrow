"""Vectorised conversion of SPSS/Stata date-like columns to Arrow temporal types.

ReadStat hands us raw numbers; what they mean depends on the file format and the
variable's display format:

===========  ==================  =====================================
format       unit                epoch
===========  ==================  =====================================
sav (SPSS)   seconds (always)    1582-10-14 (start of Gregorian calendar)
dta (Stata)  days / milliseconds 1960-01-01
===========  ==================  =====================================

"""

from __future__ import annotations

import re
import typing as t
from datetime import date

import pyarrow as pa
import pyarrow.compute as pc

from readstat_arrow._formats import FileFormat

TemporalKind = t.Literal["date", "datetime", "time", "duration"]

# The Arrow type :func:`convert` produces for each kind.
TYPE_OF_KIND: dict[TemporalKind, pa.DataType] = {
    "date": pa.date32(),
    "datetime": pa.timestamp("us"),
    "time": pa.time64("us"),
    "duration": pa.duration("us"),
}

_UNIX_EPOCH = date(1970, 1, 1)


def _micros_per_unit(file_format: FileFormat) -> int:
    """Microseconds per unit of the format's raw numbers: Stata counts milliseconds, SPSS seconds.

    Multiply by this when reading, divide by it when writing.
    """
    if file_format == "dta":
        return 1_000
    elif file_format == "sav":
        return 1_000_000
    else:
        t.assert_never(file_format)


def _epoch_days(file_format: FileFormat) -> int:
    """Days from the format's epoch to the Unix epoch."""
    if file_format == "dta":
        return (_UNIX_EPOCH - date(1960, 1, 1)).days
    elif file_format == "sav":
        return (_UNIX_EPOCH - date(1582, 10, 14)).days
    else:
        t.assert_never(file_format)


# SPSS format *names* (width/decimals stripped) that denote temporal values.
_SPSS_DATE = {"DATE", "ADATE", "EDATE", "JDATE", "SDATE"}
_SPSS_DATETIME = {"DATETIME", "YMDHMS"}
_SPSS_TIME = {"TIME"}
_SPSS_DTIME = {"DTIME"}

_FORMAT_NAME = re.compile(r"^[A-Z][A-Z0-9]*[A-Z]")
# Suffix of a Stata %tc format that shows only hours/minutes/seconds (+ optional am/pm).
_STATA_TIME_ONLY = re.compile(r"[Hh]{1,2}(:[Mm]{2})?(:[Ss]{2}(\.s+)?)?(\s*[aApP]\.?[mM]\.?)?")


def classify(file_format: FileFormat, fmt: str | None) -> TemporalKind | None:
    """Return the temporal kind a display format denotes, or ``None``."""
    if not fmt:
        return None
    if file_format == "dta":
        if fmt.startswith(("%tc", "%tC")):
            # Stata has no time type: a %tc value shown with a time-only
            # display format (e.g. %tcHH:MM:SS) is a time of day.
            return "time" if _STATA_TIME_ONLY.fullmatch(fmt[3:]) else "datetime"
        if fmt.startswith(("%td", "%d")):
            return "date"
        return None
    elif file_format == "sav":
        m = _FORMAT_NAME.match(fmt.upper())
        if not m:
            return None
        name = m.group(0)
        if name in _SPSS_DATETIME:
            return "datetime"
        if name in _SPSS_DATE:
            return "date"
        if name in _SPSS_TIME:
            return "time"
        if name in _SPSS_DTIME:
            return "duration"
        return None
    else:
        t.assert_never(file_format)


def _to_int64(arr: pa.Array | pa.ChunkedArray, scale: float) -> pa.Array | pa.ChunkedArray:
    """``round(arr * scale)`` as int64, propagating nulls."""
    if scale != 1:
        arr = pc.multiply(pc.cast(arr, pa.float64()), scale)
    if pa.types.is_floating(arr.type):
        arr = pc.round(arr)
    return pc.cast(arr, pa.int64(), safe=False)


def convert(
    arr: pa.Array | pa.ChunkedArray, file_format: FileFormat, kind: TemporalKind
) -> pa.Array | pa.ChunkedArray:
    """Convert a raw numeric column to the Arrow temporal type for ``kind``."""
    epoch_days = _epoch_days(file_format)
    scale = _micros_per_unit(file_format)  # -> microseconds

    if kind == "date":
        if file_format == "sav":
            days = pc.floor(pc.divide(pc.cast(arr, pa.float64()), 86400.0))
            days = pc.cast(days, pa.int64(), safe=False)
        elif file_format == "dta":
            days = _to_int64(arr, 1)
        else:
            t.assert_never(file_format)
        days = pc.subtract(days, epoch_days)
        return pc.cast(pc.cast(days, pa.int32()), pa.date32())

    elif kind == "datetime":
        micros = _to_int64(arr, scale)
        micros = pc.subtract(micros, epoch_days * 86_400_000_000)
        return pc.cast(micros, pa.timestamp("us"))

    elif kind == "duration":
        # SPSS DTIME: duration that may exceed 24h
        micros = _to_int64(arr, scale)
        return pc.cast(micros, pa.duration("us"))

    elif kind == "time":
        # time of day (TIME format)
        micros = _to_int64(arr, scale)
        limit = 86_400_000_000
        # Extract fractional part (time-of-day) by modulo 24h to match pyreadstat behavior.
        # For SPSS TIME format, this extracts the time-of-day regardless of day component.
        micros = pc.subtract(micros, pc.multiply(pc.floor(pc.divide(micros, limit)), limit))
        micros = pc.cast(micros, pa.int64(), safe=False)
        return pc.cast(micros, pa.time64("us"))

    else:
        t.assert_never(kind)


# ---------------------------------------------------------------------------
# Writing: Arrow temporal types back to the raw numbers each format expects
# ---------------------------------------------------------------------------

# The display format each kind is written with. ``None`` means no format: Stata has nothing that
# denotes elapsed time
DEFAULT_FORMAT: dict[FileFormat, dict[TemporalKind, str | None]] = {
    "sav": {"date": "DATE11", "datetime": "DATETIME20", "time": "TIME8", "duration": "DTIME11"},
    "dta": {"date": "%td", "datetime": "%tc", "time": "%tcHH:MM:SS", "duration": None},
}


def kind_of_type(typ: pa.DataType) -> TemporalKind | None:
    """The temporal kind an Arrow type maps to when writing, or ``None`` for non-temporal types."""
    if pa.types.is_date(typ):
        return "date"
    if pa.types.is_timestamp(typ):
        return "datetime"
    if pa.types.is_duration(typ):
        return "duration"
    if pa.types.is_time(typ):
        return "time"
    return None


def to_raw(arr: pa.Array, file_format: FileFormat, kind: TemporalKind) -> pa.Array:
    """Inverse of :func:`convert`: a temporal array as the format's raw numeric representation.

    Returns ``int32`` for Stata dates (days) and ``float64`` for everything else.
    Timezone-aware timestamps are written as their UTC instant; neither format
    stores a timezone.
    """
    epoch_days = _epoch_days(file_format)
    divisor = _micros_per_unit(file_format)

    if kind == "date":
        days = pc.cast(pc.cast(arr, pa.date32()), pa.int32())
        if file_format == "dta":
            return pc.cast(pc.add(days, epoch_days), pa.int32())
        elif file_format == "sav":
            return pc.multiply(pc.cast(pc.add(days, epoch_days), pa.float64()), 86400.0)
        else:
            t.assert_never(file_format)

    if kind == "datetime":
        unit_type = (
            pa.timestamp("us", tz=arr.type.tz) if pa.types.is_timestamp(arr.type) else pa.timestamp("us")
        )
        micros = pc.cast(pc.cast(arr, unit_type), pa.int64())
        micros = pc.add(micros, epoch_days * 86_400_000_000)
        return pc.divide(pc.cast(micros, pa.float64(), safe=False), divisor)

    if kind == "duration":
        micros = pc.cast(pc.cast(arr, pa.duration("us")), pa.int64())
        # milliseconds for Stata, as %tc and %tC also count
        return pc.divide(pc.cast(micros, pa.float64(), safe=False), divisor)

    if kind == "time":
        micros = pc.cast(pc.cast(arr, pa.time64("us")), pa.int64())
        return pc.divide(pc.cast(micros, pa.float64(), safe=False), divisor)

    t.assert_never(kind)
