"""Public reading API: ``read_sav`` / ``read_dta``, their ``*_metadata`` variants,
and the incremental ``open_sav`` / ``open_dta``."""

from __future__ import annotations

import contextlib
import os
import re
import typing as t
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import replace

import pyarrow as pa
import pyarrow.compute as pc

from readstat_arrow import _dates
from readstat_arrow._cython import parser as _parser
from readstat_arrow._formats import SUPPORTS_TAGGED_MISSING, FileFormat
from readstat_arrow.errors import ReadstatWarning
from readstat_arrow.metadata import PER_VARIABLE, Metadata

__all__ = [
    "DtaStreamingReader",
    "SavStreamingReader",
    "open_dta",
    "open_sav",
    "read_dta",
    "read_dta_metadata",
    "read_sav",
    "read_sav_metadata",
]

PathLike = str | os.PathLike[str] | bytes

# Type of the ``tag`` field when Stata tagged missings are preserved.
TAG_TYPE = pa.dictionary(pa.int8(), pa.string())

DEFAULT_BATCH_ROWS = 65_536

# The signed integer types a column can be narrowed to, narrowest first, with the
# range each one holds. int64 saves no space over the double a file stores - both
# are 8 bytes - but it is the cleaner type for a column of whole numbers, and it
# keeps a table of integer columns from having one of them stand out as a float.
# Unsigned types are left out - they would buy one bit at the cost of a type most
# consumers of the table handle less well.
_INT_TYPES: tuple[tuple[pa.DataType, int, int], ...] = (
    (pa.int8(), -(2**7), 2**7 - 1),
    (pa.int16(), -(2**15), 2**15 - 1),
    (pa.int32(), -(2**31), 2**31 - 1),
    (pa.int64(), -(2**63), 2**63 - 1),
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
        ``int8``/``int16``/``int32``/``int64`` where every value was a whole number,
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
) -> tuple[pa.Schema, int | None, Metadata]:
    """Read only the metadata of an SPSS ``.sav`` file; no data rows are decoded.

    Takes a path or a seekable binary file object, as :func:`read_sav` does.

    Returns ``(schema, num_rows, metadata)``: the ``pyarrow.Schema`` a full read
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
) -> tuple[pa.Schema, int | None, Metadata]:
    """Read only the metadata of a Stata ``.dta`` file; no data rows are decoded.

    Returns ``(schema, num_rows, metadata)`` as :func:`read_sav_metadata` does;
    Stata files always record the row count.
    """
    return _read_metadata(where, "dta", encoding)


def open_sav(
    where: PathLike | t.IO[bytes],
    *,
    columns: Iterable[str] | None = None,
    row_limit: int = 0,
    row_offset: int = 0,
    encoding: str | None = None,
    scan_and_narrow_types: bool = False,
    preserve_user_missing: bool = False,
) -> SavStreamingReader:
    """Open an SPSS ``.sav`` file for incremental reading."""
    return t.cast(
        SavStreamingReader,
        _open(
            where,
            "sav",
            columns,
            row_limit,
            row_offset,
            encoding,
            scan_and_narrow_types,
            preserve_user_missing,
        ),
    )


def open_dta(
    where: PathLike | t.IO[bytes],
    *,
    columns: Iterable[str] | None = None,
    row_limit: int = 0,
    row_offset: int = 0,
    encoding: str | None = None,
    scan_and_narrow_types: bool = False,
    preserve_user_missing: bool = False,
) -> DtaStreamingReader:
    """Open a Stata ``.dta`` file for incremental reading."""
    return t.cast(
        DtaStreamingReader,
        _open(
            where,
            "dta",
            columns,
            row_limit,
            row_offset,
            encoding,
            scan_and_narrow_types,
            preserve_user_missing,
        ),
    )


_OPEN_DOC = """
    Reads a batch of rows at a time instead of the whole file at once, so a file
    larger than memory can be converted to something column-oriented - Parquet,
    say - a batch at a time:

    >>> import pyarrow.parquet as pq
    >>> reader = readstat_arrow.open_sav("big.sav")
    >>> with pq.ParquetWriter("big.parquet", reader.schema) as writer:
    ...     reader.read_batches(writer.write_batch)

    The parse drives:
    :meth:`~readstat_arrow.SavStreamingReader.read_batches` calls ``callback``
    with each batch in turn, which is the shape ReadStat's own callbacks give
    and so costs no buffering and no thread. There is nothing to close, and
    nothing stopping a second read.

    Parameters
    ----------
    where, columns, row_limit, row_offset, encoding, preserve_user_missing:
        As :func:`read_sav`, except that a file object is left where it began
        rather than where reading stopped, so that reading again reads the same
        bytes. It must be left alone while the reader is using it.
    scan_and_narrow_types:
        As :func:`read_sav`, and worth rather more here: the scan keeps no values,
        so the narrow types it settles on are the ones every batch is read into,
        and the pass it costs is the only thing between a file and a narrowly
        typed copy of it that neither side ever holds whole.

    Returns
    -------
    reader:
        A :class:`~readstat_arrow.SavStreamingReader` /
        :class:`~readstat_arrow.DtaStreamingReader`, which also carries the
        ``schema``, ``num_rows`` and :class:`~readstat_arrow.Metadata` the file
        declares. Opening reads that metadata and nothing else; not a row is
        touched until :meth:`~readstat_arrow.SavStreamingReader.read_batches` or
        :meth:`~readstat_arrow.SavStreamingReader.read_all` is called.
"""

for _open_fn in (open_sav, open_dta):
    _open_fn.__doc__ = (_open_fn.__doc__ or "") + _OPEN_DOC


class _StreamingReader:
    """Shared implementation of :class:`SavStreamingReader` and :class:`DtaStreamingReader`.

    Holds what the file declares - :attr:`schema`, :attr:`metadata`,
    :attr:`num_rows`, all read before a single row is touched - and hands the
    rows over a batch at a time through :meth:`read_batches`.

    ReadStat parses a whole file in one call, pushing a value at a time at us, so
    that is the shape the rows come in: the parse drives and the caller supplies
    a callback. Nothing is buffered, nothing is held between opening and reading,
    and no thread is involved.
    """

    _file_format: FileFormat

    #: The schema every batch has.
    schema: pa.Schema

    #: What the file declares about the variables being read.
    metadata: Metadata

    #: Rows in the file as its header records them, or ``None`` where it does not
    #: (some non-SPSS writers omit it); what ``row_limit``/``row_offset`` will
    #: actually yield is not taken off it. The writers in this package need it up
    #: front, which is why it is here rather than in :attr:`metadata`.
    num_rows: int | None

    def __init__(
        self,
        path: bytes | None,
        file: t.IO[bytes] | None,
        *,
        schema: pa.Schema,
        read_schema: pa.Schema,
        metadata: Metadata,
        num_rows: int | None,
        tagged: bool,
        parse_kwargs: dict[str, t.Any],
    ) -> None:
        self.schema = schema
        self.metadata = metadata
        self.num_rows = num_rows
        self._read_schema = read_schema  # what a batch's arrays are, before converting
        self._tagged = tagged
        self._path = path
        self._file = file
        self._parse_kwargs = parse_kwargs

    def read_batches(
        self,
        callback: t.Callable[[pa.RecordBatch], object],
        *,
        batch_rows: int = DEFAULT_BATCH_ROWS,
    ) -> int:
        """Read the file, handing each batch to ``callback``; returns the rows read.

        Every batch has :attr:`schema`, and holds the same number of rows bar the
        last, which holds the remainder. A file with no rows calls back never and
        returns ``0``.

        Raise from ``callback`` to stop early: the parse is abandoned and the
        exception comes back out of here. Reading twice reads the file twice.

        Parameters
        ----------
        callback:
            Called with each ``pyarrow.RecordBatch`` in turn. What it returns is
            ignored; what it raises stops the read.
        batch_rows:
            Rows per batch. What a batch costs is rows times columns, so the
            right number depends on how wide the file is: the default is modest
            for a few dozen columns and far too much for a thousand.
            ``len(reader.schema)`` is known by the time this is called, which is
            why the choice lives here rather than on :func:`open_sav`.
        """
        if batch_rows <= 0:
            raise ValueError(f"batch_rows must be positive, got {batch_rows}")
        rows = 0

        def on_batch(arrays: list[pa.Array], tags: list[pa.Array | None]) -> None:
            nonlocal rows
            batch = self._to_batch(arrays, tags)
            rows += batch.num_rows
            callback(batch)

        # Left where it began, so reading again reads the same bytes.
        with _rewound(self._file):
            _parser.parse(
                self._path,
                self._file_format,
                on_batch=on_batch,
                batch_rows=batch_rows,
                **self._parse_kwargs,
            )
        return rows

    def read_all(self) -> pa.Table:
        """The whole file as one table - what the matching ``read_*`` would give."""
        batches: list[pa.RecordBatch] = []
        self.read_batches(batches.append)
        return pa.Table.from_batches(batches, self.schema)

    def _to_batch(self, arrays: list[pa.Array], tags: list[pa.Array | None]) -> pa.RecordBatch:
        """One batch's raw columns, converted exactly as :func:`_read_data` converts a table.

        ``from_arrays`` against ``_read_schema`` is also the check that the
        metadata pass and this one saw the same file: a file rewritten between
        them fails here rather than quietly producing something else.
        """
        table = pa.Table.from_arrays(arrays, schema=self._read_schema)
        table = _convert_dates(table, self.metadata, self._file_format)
        if self._tagged:
            table = _with_tag_structs(table, tags)
        columns = [column.combine_chunks() for column in table.columns]
        return pa.RecordBatch.from_arrays(columns, schema=self.schema)


class SavStreamingReader(_StreamingReader):
    """Reads an SPSS ``.sav`` file a batch at a time; see :func:`open_sav`."""

    _file_format: FileFormat = "sav"


class DtaStreamingReader(_StreamingReader):
    """Reads a Stata ``.dta`` file a batch at a time; see :func:`open_dta`."""

    _file_format: FileFormat = "dta"


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

_READER_OF: dict[FileFormat, type[_StreamingReader]] = {
    "sav": SavStreamingReader,
    "dta": DtaStreamingReader,
}


def _open(
    where: PathLike | t.IO[bytes],
    file_format: FileFormat,
    columns: Iterable[str] | None,
    row_limit: int,
    row_offset: int,
    encoding: str | None,
    scan_and_narrow_types: bool,
    preserve_user_missing: bool,
) -> _StreamingReader:
    path, file = _source(where)
    names = None if columns is None else list(columns)

    with _rewound(file):
        stored_schema, metadata, num_rows, messages = _metadata_pass(path, file, file_format, encoding)
    _emit_warnings(messages)
    metadata = _for_columns(metadata, names)

    types = None
    if scan_and_narrow_types:
        # Another pass of its own: the types every batch is read into have to be
        # settled before the first of them is.
        types = _scanned_types(
            path, file, file_format, columns, row_limit, row_offset, encoding, preserve_user_missing
        )

    read_schema = _selected_schema(stored_schema, names, types)
    schema = _convert_date_types(read_schema, metadata, file_format)
    tagged = preserve_user_missing and SUPPORTS_TAGGED_MISSING[file_format]
    if tagged:
        schema = _tag_struct_schema(schema)

    return _READER_OF[file_format](
        path,
        file,
        schema=schema,
        read_schema=read_schema,
        metadata=metadata,
        num_rows=num_rows,
        tagged=tagged,
        parse_kwargs={
            "file": file,
            "columns": names,
            "types": types,
            "row_limit": row_limit,
            "row_offset": row_offset,
            "encoding": encoding,
            "preserve_user_missing": preserve_user_missing,
        },
    )


def _selected_schema(
    stored: pa.Schema, columns: list[str] | None, types: dict[str, pa.DataType] | None
) -> pa.Schema:
    """The schema the parse will hand back: the file's variables, selected and retyped.

    The other half of what :func:`_metadata_pass` cannot know - which variables
    were asked for, and at which types - applied the way the variable callback
    applies it, in file order. Only the types a scan settled on ever get here, so
    there is nothing to validate that ``parse`` will not validate again.
    """
    fields = [field_ for field_ in stored if columns is None or field_.name in columns]
    if types is not None:
        fields = [field_.with_type(types.get(field_.name, field_.type)) for field_ in fields]
    return pa.schema(fields)


def _for_columns(metadata: Metadata, columns: list[str] | None) -> Metadata:
    """Narrow ``metadata`` to ``columns``, as a read of only those columns reports it.

    The metadata pass sees every variable; a read that skipped some declares
    nothing about them. File-level fields stay as they are - including
    ``multiple_response_sets``, which a filtered read does not trim either.
    """
    if columns is None:
        return metadata
    selected = frozenset(columns)
    kept: dict[str, t.Any] = {
        field_name: {name: value for name, value in getattr(metadata, field_name).items() if name in selected}
        for field_name in PER_VARIABLE
    }
    return replace(metadata, **kept)


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
) -> tuple[pa.Schema, int | None, Metadata]:
    path, file = _source(where)
    schema, metadata, num_rows, messages = _metadata_pass(path, file, file_format, encoding)
    _emit_warnings(messages)
    return _convert_date_types(schema, metadata, file_format), num_rows, metadata


def _metadata_pass(
    path: bytes | None, file: t.IO[bytes] | None, file_format: FileFormat, encoding: str | None
) -> tuple[pa.Schema, Metadata, int | None, list[str]]:
    """Returns the variables as the file stores them - before any column selection,
    narrowing or date conversion - along with the metadata, the header's row
    count, and ReadStat's messages for the caller to warn about at its own line.
    """
    _, _, schema, metadata, num_rows, messages, _ = _parser.parse(
        path, file_format, file=file, metadata_only=True, encoding=encoding
    )
    return schema, _normalise_widths(metadata, file_format), num_rows, messages


@contextlib.contextmanager
def _rewound(file: t.IO[bytes] | None) -> Iterator[None]:
    """Leave ``file`` where it was, so the pass that follows sees the same bytes.

    A file object may well have started partway into a larger stream, so where it
    began is not where rewinding it would put it. Nothing to do for a path.
    """
    start = file.tell() if file is not None else 0
    yield
    if file is not None:
        file.seek(start)


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

    ReadStat's recoverable-problem messages are dropped here rather than warned
    about: the read parses the same file the same way and reports them itself,
    and one warning per problem is enough.
    """
    with _rewound(file):
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


def _tag_struct_schema(schema: pa.Schema) -> pa.Schema:
    """The schema :func:`_with_tag_structs` produces, without any data to look at.

    Every batch has to have the same type whether or not it happens to hold a
    tagged missing, so this wraps what that function wraps - every column a Stata
    numeric, which is every column that is not a string.
    """
    fields = [
        field_
        if pa.types.is_string(field_.type) or pa.types.is_large_string(field_.type)
        else field_.with_type(pa.struct([("value", field_.type), ("tag", TAG_TYPE)]))
        for field_ in schema
    ]
    return pa.schema(fields)


def _emit_warnings(messages: list[str]) -> None:
    """Re-emit ReadStat's recoverable-problem messages as Python warnings at the caller's line."""
    for message in messages:
        # stacklevel: _emit_warnings -> _read_data/_read_metadata/_open -> read_*/open_* -> caller
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
