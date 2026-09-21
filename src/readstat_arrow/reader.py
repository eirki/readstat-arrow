"""Public reading API: ``read_sav`` / ``read_dta`` and their ``*_metadata`` variants."""

from __future__ import annotations

import os
import re
import typing as t
import warnings
from collections.abc import Iterable
from dataclasses import replace

import pyarrow as pa
import pyarrow.compute as pc

from readstat_arrow import _dates
from readstat_arrow._cython import parser as _parser
from readstat_arrow._formats import SUPPORTS_TAGGED_MISSING, FileFormat
from readstat_arrow.errors import ReadstatWarning
from readstat_arrow.metadata import Metadata

__all__ = ["read_dta", "read_dta_metadata", "read_sav", "read_sav_metadata"]

PathLike = str | os.PathLike[str] | bytes

# Type of the ``tag`` field when Stata tagged missings are preserved.
TAG_TYPE = pa.dictionary(pa.int8(), pa.string())

# The signed integer types a column can be narrowed to, narrowest first, with the
# range each one holds. There is no int64: a double already holds every integer
# up to 2^53 in the same 8 bytes, so nothing is saved below that and nothing is
# exact above it. Unsigned types are left out too - they would buy one bit at the
# cost of a type most consumers of the table handle less well.
_INT_TYPES: tuple[tuple[pa.DataType, int, int], ...] = (
    (pa.int8(), -(2**7), 2**7 - 1),
    (pa.int16(), -(2**15), 2**15 - 1),
    (pa.int32(), -(2**31), 2**31 - 1),
)

_READ_DOC = """
    Parameters
    ----------
    where:
        Path to the file, or a binary file object to read it from. A file object
        must be seekable - both formats jump around the file - and is read from
        its current position and left open; its position afterwards is wherever
        reading stopped.
    columns:
        Names of the variables to read; ``None`` (default) reads all of them.
    row_limit, row_offset:
        Read at most ``row_limit`` rows, starting ``row_offset`` rows in. ``0`` means no limit.
    encoding:
        Override the character encoding declared in (or inferred from) the file.
    scan_and_narrow_types:
        Scan the values first, then read every numeric column at the narrowest
        type that holds the ones the scan found, rather than at the type the file
        stores the column as.

        Where memory is the constraint, holding the whole table in it is the
        expensive part of a read, and the stored type is often wider than the
        values need: a .sav is the worst of it, every numeric column being a
        64-bit double whatever it holds, and a .dta variable is only as narrow as
        whoever wrote the file declared it. So a column of one-digit codes can
        cost 8 bytes a cell; with ``True`` it comes back as ``int8``.

        The scan keeps a few scalars per column and no values at all, so the file
        is parsed twice but never held twice: the trade is time for memory, at
        about twice the wall clock of a plain read. The widths it settles on are
        ``int8``/``int16``/``int32`` where every value was a whole number,
        ``float32`` where every value survives one, else ``float64``; strings are
        untouched, and a column of nothing but nulls comes back as ``int8``. A
        value that does not fit the width measured for it raises
        :class:`~readstat_arrow.ReadstatError` rather than wrapping around, which
        no file that holds still between the two passes can provoke.
    preserve_user_missing:
        Keep the file's user-level missing information instead of collapsing it
        to null. By default every kind of missing is an Arrow null. With ``True``:

        * SPSS: values declared missing (``MISSING VALUES x (-1)``) stay in the
          data; ``Metadata.missing_values`` says which they are.
        * Stata: every numeric column becomes
          ``struct<value: <numeric>, tag: dictionary<int8, string>>`` so the
          tagged missings ``.a``-``.z`` survive: ``.a`` is ``{value: null, tag: "a"}``,
          a plain ``.`` is a null struct, a real number has a null tag. The
          writers accept the same structs back.

        System-missing is always null.

    Returns
    -------
    (table, metadata):
        The data as a ``pyarrow.Table`` and a :class:`~readstat_arrow.Metadata`
        describing the file.
"""


def read_sav(
    where: PathLike | t.IO[bytes],
    *,
    columns: Iterable[str] | None = None,
    row_limit: int = 0,
    row_offset: int = 0,
    encoding: str | None = None,
    scan_and_narrow_types: bool = False,
    preserve_user_missing: bool = False,
) -> tuple[pa.Table, Metadata]:
    """Read an SPSS ``.sav`` file."""
    return _read_data(
        where,
        "sav",
        columns,
        row_limit,
        row_offset,
        encoding,
        scan_and_narrow_types,
        preserve_user_missing,
    )


def read_dta(
    where: PathLike | t.IO[bytes],
    *,
    columns: Iterable[str] | None = None,
    row_limit: int = 0,
    row_offset: int = 0,
    encoding: str | None = None,
    scan_and_narrow_types: bool = False,
    preserve_user_missing: bool = False,
) -> tuple[pa.Table, Metadata]:
    """Read a Stata ``.dta`` file."""
    return _read_data(
        where,
        "dta",
        columns,
        row_limit,
        row_offset,
        encoding,
        scan_and_narrow_types,
        preserve_user_missing,
    )


for _fn in (read_sav, read_dta):
    _fn.__doc__ = (_fn.__doc__ or "") + _READ_DOC


def read_sav_metadata(
    where: PathLike | t.IO[bytes], *, encoding: str | None = None
) -> tuple[int | None, pa.Schema, Metadata]:
    """Read only the metadata of an SPSS ``.sav`` file; no data rows are decoded.

    Takes a path or a seekable binary file object, as :func:`read_sav` does.

    Returns ``(row_count, schema, metadata)``: the ``pyarrow.Schema`` a full read
    would give the table - variable names in file order, with the type each column
    would come back as - and everything the file declares about the variables. The
    row count comes from the file header and is ``None`` when the file does not
    record it (some non-SPSS writers omit it).

    The schema describes a ``preserve_user_missing=False`` read; that option changes
    .dta column types (see :func:`read_dta`) and is not reflected here.
    """
    return _read_metadata(where, "sav", encoding)


def read_dta_metadata(
    where: PathLike | t.IO[bytes], *, encoding: str | None = None
) -> tuple[int | None, pa.Schema, Metadata]:
    """Read only the metadata of a Stata ``.dta`` file; no data rows are decoded.

    Returns ``(row_count, schema, metadata)`` as :func:`read_sav_metadata` does;
    Stata files always record the row count.
    """
    return _read_metadata(where, "dta", encoding)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _read_data(
    where: PathLike | t.IO[bytes],
    file_format: FileFormat,
    columns: Iterable[str] | None,
    row_limit: int,
    row_offset: int,
    encoding: str | None,
    scan_and_narrow_types: bool,
    preserve_user_missing: bool,
) -> tuple[pa.Table, Metadata]:
    path, file = _source(where)
    types = None
    if scan_and_narrow_types:
        types = _scanned_types(
            path, file, file_format, columns, row_limit, row_offset, encoding, preserve_user_missing
        )
    arrays, tags, table_schema, metadata, _, messages, _ = _parser.parse(
        path,
        file_format,
        file=file,
        metadata_only=False,
        columns=None if columns is None else list(columns),
        types=types,
        row_limit=row_limit,
        row_offset=row_offset,
        encoding=encoding,
        preserve_user_missing=preserve_user_missing,
    )
    _emit_warnings(messages)
    metadata = _normalise_widths(metadata, file_format)
    table = pa.Table.from_arrays(arrays, schema=table_schema)

    table = _convert_dates(table, metadata, file_format)
    if preserve_user_missing and SUPPORTS_TAGGED_MISSING[file_format]:
        table = _with_tag_structs(table, tags)

    return table, metadata


def _read_metadata(
    where: PathLike | t.IO[bytes], file_format: FileFormat, encoding: str | None
) -> tuple[int | None, pa.Schema, Metadata]:
    path, file = _source(where)
    _, _, schema, metadata, row_count, messages, _ = _parser.parse(
        path, file_format, file=file, metadata_only=True, encoding=encoding
    )
    _emit_warnings(messages)
    metadata = _normalise_widths(metadata, file_format)
    schema = _convert_date_types(schema, metadata, file_format)
    return row_count, schema, metadata


def _scanned_types(
    path: bytes | None,
    file: t.IO[bytes] | None,
    file_format: FileFormat,
    columns: Iterable[str] | None,
    row_limit: int,
    row_offset: int,
    encoding: str | None,
    preserve_user_missing: bool,
) -> dict[str, pa.DataType]:
    """The narrowest type each column fits in, from a scan that stores no values.

    Names only the columns that narrow; the rest are read as the file stores them.
    Date-like columns narrow like any other - what they are converted to afterwards
    does not depend on the width they were read at, and a narrower one is less to
    convert.

    A file object is left where it began, so the read that follows sees the same
    bytes - it may well have started partway into a larger stream. ReadStat's
    recoverable-problem messages are dropped here rather than warned about: the
    read parses the same file the same way and reports them itself, and one
    warning per problem is enough.
    """
    start = file.tell() if file is not None else 0
    *_, summaries = _parser.parse(
        path,
        file_format,
        file=file,
        scan=True,
        columns=None if columns is None else list(columns),
        row_limit=row_limit,
        row_offset=row_offset,
        encoding=encoding,
        preserve_user_missing=preserve_user_missing,
    )
    if file is not None:
        file.seek(start)
    narrowed = ((summary, _narrow_type(summary)) for summary in summaries)
    return {summary["name"]: narrow for summary, narrow in narrowed if narrow != summary["type"]}


def _narrow_type(summary: dict[str, t.Any]) -> pa.DataType:
    """The narrowest Arrow type that holds a scanned column without losing anything.

    Integer types when every value was a whole number, then ``float32`` when every
    value survives a round trip through it, else ``float64`` - and the stored type
    unchanged for strings, whose size is their bytes rather than their type. A
    column of nothing but nulls comes back as ``int8``: one byte a row is as small
    as a column with a validity bitmap gets.
    """
    stored: pa.DataType = summary["type"]
    if pa.types.is_string(stored) or pa.types.is_large_string(stored):
        return stored
    if summary["all_integral"]:
        low, high = summary["min"], summary["max"]
        if low is None or high is None:  # nothing but nulls
            return pa.int8()
        for typ, lo, hi in _INT_TYPES:
            if lo <= low and high <= hi:
                return typ
    if summary["float32_exact"]:
        return pa.float32()
    return pa.float64()


_SPSS_STRING_FORMAT = re.compile(r"A(\d+)$")  # "A20"; AHEX counts hex digits, so it is left alone
_STATA_STRING_FORMAT = re.compile(r"%-?\d+s$")  # "%20s", "%-20s"


def _normalise_widths(metadata: Metadata, file_format: FileFormat) -> Metadata:
    """Report each storage width as the width that was declared, not the space it occupies.

    ReadStat hands back each format's own storage accounting, which is wider than
    the declared width for strings: SPSS rounds up to whole 8-byte cells (``A20``
    -> 24, ``A3`` -> 8), and Stata adds a byte for a possible NUL (``str20`` ->
    21). Left alone, those inflated numbers grow every time metadata from a file
    is handed to :class:`~readstat_arrow.SavWriter` / :class:`~readstat_arrow.DtaWriter`
    to write the file again. Numeric variables already report their real size.
    """
    widths: dict[str, int | None] = {
        name: None if width is None else _declared_width(metadata.formats.get(name), width, file_format)
        for name, width in metadata.storage_widths.items()
    }
    return replace(metadata, storage_widths=widths)


def _declared_width(fmt: str | None, width: int, file_format: FileFormat) -> int:
    """The declared byte width behind ReadStat's storage width, if this is a string variable."""
    if fmt is None:
        return width
    if file_format == "sav":
        match = _SPSS_STRING_FORMAT.match(fmt)
        return int(match.group(1)) if match else width
    elif file_format == "dta":
        return width - 1 if _STATA_STRING_FORMAT.match(fmt) else width
    else:
        t.assert_never(file_format)


def _with_tag_structs(table: pa.Table, tags: list[pa.Array | None]) -> pa.Table:
    """Wrap every numeric column as ``struct<value, tag>`` carrying its tagged-missing letters."""
    for i, tag_array in enumerate(tags):
        column = table.column(i)
        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            continue  # Stata strings cannot be missing, tagged or otherwise
        values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
        if tag_array is None:
            tag_array = pa.nulls(len(values), TAG_TYPE)
        # The struct itself is null only for the plain '.', so null_count keeps meaning "missing".
        struct = pa.StructArray.from_arrays(
            [values, tag_array], names=["value", "tag"], mask=pc.and_(values.is_null(), tag_array.is_null())
        )
        table = table.set_column(i, table.column_names[i], struct)
    return table


def _emit_warnings(messages: list[str]) -> None:
    """Re-emit ReadStat's recoverable-problem messages as Python warnings at the caller's line."""
    for message in messages:
        # stacklevel: _emit_warnings -> _read_data/_read_metadata -> read_* -> caller
        warnings.warn(message, ReadstatWarning, stacklevel=4)


def _convert_date_types(schema: pa.Schema, metadata: Metadata, file_format: FileFormat) -> pa.Schema:
    """Retype the fields :func:`_convert_dates` would convert, without touching any data."""
    for i, field_ in enumerate(schema):
        if not pa.types.is_floating(field_.type) and not pa.types.is_integer(field_.type):
            continue
        kind = _dates.classify(file_format, metadata.formats.get(field_.name))
        if kind is not None:
            schema = schema.set(i, field_.with_type(_dates.TYPE_OF_KIND[kind]))
    return schema


def _convert_dates(table: pa.Table, metadata: Metadata, file_format: FileFormat) -> pa.Table:
    for i, name in enumerate(table.column_names):
        if not pa.types.is_floating(table.column(i).type) and not pa.types.is_integer(table.column(i).type):
            continue
        kind = _dates.classify(file_format, metadata.formats.get(name))
        if kind is not None:
            table = table.set_column(i, name, _dates.convert(table.column(i), file_format, kind))
    return table


def _source(where: PathLike | t.IO[bytes]) -> tuple[bytes | None, t.IO[bytes] | None]:
    """Split ``where`` into the ``(path, file)`` pair the compiled parser takes.

    Exactly one of the two is not ``None``. A file object has to be seekable:
    both formats read their headers and then jump to the data, and ReadStat asks
    for absolute offsets, so there is no reading either format in one pass.
    """
    if isinstance(where, str | bytes | os.PathLike):
        return _fs_path(where), None
    seekable = getattr(where, "seekable", None)
    if seekable is not None and not seekable():
        raise ValueError("file object is not seekable; read it into io.BytesIO, or pass a path instead")
    return None, where


def _fs_path(path: PathLike) -> bytes:
    if isinstance(path, bytes):
        return path
    return os.fsencode(os.path.expanduser(os.fspath(path)))
