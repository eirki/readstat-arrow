# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True, embedsignature=True
# mypy: ignore-errors
"""Cython (pure Python mode) bridge between ReadStat's callback API and Arrow buffers.

ReadStat parses a file and calls back into us once for the file metadata, once
per variable, and once per (row, variable) cell.  Each :class:`ColumnBuilder`
writes straight into contiguous byte buffers laid out exactly as Arrow expects
them, so the final ``pyarrow.Array`` is created with zero copies via
``pa.Array.from_buffers``.

This file is syntactically valid Python but only meaningful once compiled by
Cython; do not import it directly - use :mod:`readstat_arrow.reader`.

It is typed for *Cython*, not for mypy: the annotations are C types
(``cython.int``, ``cython.pointer(...)``) and the decorators have no stubs, so
mypy is told to skip the file (``# mypy: ignore-errors`` above, mirrored by the
``exclude`` in pyproject.toml). Cython itself type-checks it at compile time.
"""

from __future__ import annotations

import typing as t

import cython
import pyarrow as pa
import pyarrow.compute as pc
from cython.cimports.cpython.buffer import PyBUF_WRITE
from cython.cimports.cpython.memoryview import PyMemoryView_FromMemory
from cython.cimports.libc.math import floor, isfinite
from cython.cimports.libc.string import memcpy, strlen
from cython.cimports.readstat_arrow._cython.readstat import (
    READSTAT_HANDLER_ABORT,
    READSTAT_HANDLER_OK,
    READSTAT_HANDLER_SKIP_VARIABLE,
    READSTAT_OK,
    READSTAT_SEEK_CUR,
    READSTAT_SEEK_END,
    READSTAT_SEEK_SET,
    READSTAT_TYPE_DOUBLE,
    READSTAT_TYPE_FLOAT,
    READSTAT_TYPE_INT8,
    READSTAT_TYPE_INT16,
    READSTAT_TYPE_INT32,
    READSTAT_TYPE_STRING,
    READSTAT_TYPE_STRING_REF,
    mr_set_t,
    readstat_double_value,
    readstat_error_message,
    readstat_error_t,
    readstat_float_value,
    readstat_get_file_label,
    readstat_get_multiple_response_sets,
    readstat_get_multiple_response_sets_length,
    readstat_get_row_count,
    readstat_int8_value,
    readstat_int16_value,
    readstat_int32_value,
    readstat_io_flags_t,
    readstat_metadata_t,
    readstat_off_t,
    readstat_parse_dta,
    readstat_parse_sav,
    readstat_parser_free,
    readstat_parser_init,
    readstat_parser_t,
    readstat_progress_handler,
    readstat_set_close_handler,
    readstat_set_error_handler,
    readstat_set_file_character_encoding,
    readstat_set_io_ctx,
    readstat_set_metadata_handler,
    readstat_set_note_handler,
    readstat_set_open_handler,
    readstat_set_read_handler,
    readstat_set_row_limit,
    readstat_set_row_offset,
    readstat_set_seek_handler,
    readstat_set_update_handler,
    readstat_set_value_handler,
    readstat_set_value_label_handler,
    readstat_set_variable_handler,
    readstat_string_value,
    readstat_type_t,
    readstat_value_is_defined_missing,
    readstat_value_is_missing,
    readstat_value_is_tagged_missing,
    readstat_value_t,
    readstat_value_tag,
    readstat_value_type,
    readstat_variable_get_alignment,
    readstat_variable_get_display_width,
    readstat_variable_get_format,
    readstat_variable_get_index_after_skipping,
    readstat_variable_get_label,
    readstat_variable_get_measure,
    readstat_variable_get_missing_range_hi,
    readstat_variable_get_missing_range_lo,
    readstat_variable_get_missing_ranges_count,
    readstat_variable_get_name,
    readstat_variable_get_storage_width,
    readstat_variable_get_type,
    readstat_variable_t,
)

from readstat_arrow._formats import FileFormat
from readstat_arrow.errors import ReadstatError
from readstat_arrow.metadata import Metadata

# ---------------------------------------------------------------------------
# Column kinds (a compact int so the hot loop can switch on it)
# ---------------------------------------------------------------------------

K_STRING = cython.declare(cython.int, 0)
K_INT8 = cython.declare(cython.int, 1)
K_INT16 = cython.declare(cython.int, 2)
K_INT32 = cython.declare(cython.int, 3)
K_FLOAT = cython.declare(cython.int, 4)
K_DOUBLE = cython.declare(cython.int, 5)
# Not an Arrow type but a seventh arm of the value handler's dispatch: the column
# was asked for in a narrower type than the file stores, so its values arrive as
# doubles and are range-checked on the way in. Kept out of the tables below,
# which are keyed by the type a column actually has.
K_NARROWED = cython.declare(cython.int, 6)

_ITEM_SIZE = {K_INT8: 1, K_INT16: 2, K_INT32: 4, K_FLOAT: 4, K_DOUBLE: 8}
_STRUCT_FMT = {K_INT8: "b", K_INT16: "h", K_INT32: "i", K_FLOAT: "f", K_DOUBLE: "d"}
_ARROW_TYPE = {
    K_STRING: pa.large_string(),
    K_INT8: pa.int8(),
    K_INT16: pa.int16(),
    K_INT32: pa.int32(),
    K_FLOAT: pa.float32(),
    K_DOUBLE: pa.float64(),
}

# The kind a caller asks for by naming an Arrow type in ``parse(types=...)``.
_KIND_OF_ARROW = {typ: kind for kind, typ in _ARROW_TYPE.items()}

_FLOAT_MAX = cython.declare(cython.double, 3.4028234663852886e38)
_INT8_MIN = cython.declare(cython.double, -128.0)
_INT8_MAX = cython.declare(cython.double, 127.0)
_INT16_MIN = cython.declare(cython.double, -32768.0)
_INT16_MAX = cython.declare(cython.double, 32767.0)
_INT32_MIN = cython.declare(cython.double, -2147483648.0)
_INT32_MAX = cython.declare(cython.double, 2147483647.0)

# set_checked outcomes; it cannot raise, so the caller turns these into errors.
FITS = cython.declare(cython.int, 0)
NOT_INTEGRAL = cython.declare(cython.int, 1)
OUT_OF_RANGE = cython.declare(cython.int, 2)

_MEASURE_NAMES = {0: "unknown", 1: "nominal", 2: "ordinal", 3: "scale"}
_ALIGNMENT_NAMES = {0: "unknown", 1: "left", 2: "center", 3: "right"}

_INITIAL_CAPACITY_UNKNOWN_ROWS = cython.declare(cython.Py_ssize_t, 4096)

# Dictionary for tag columns: index 0 is 'a', 25 is 'z'.
TAG_LETTERS = pa.array([chr(c) for c in range(ord("a"), ord("z") + 1)], pa.string())


@cython.cfunc
@cython.inline
def _decode(s: cython.p_const_char) -> object:
    """Decode a NUL-terminated C string to ``str`` (``None`` for NULL / empty)."""
    if s is cython.NULL or s[0] == 0:
        return None
    b: bytes = s
    return b.decode("utf-8", "replace")


@cython.cfunc
def _kind_for_type(t: readstat_type_t) -> cython.int:
    if t == READSTAT_TYPE_STRING or t == READSTAT_TYPE_STRING_REF:
        return K_STRING
    if t == READSTAT_TYPE_INT8:
        return K_INT8
    if t == READSTAT_TYPE_INT16:
        return K_INT16
    if t == READSTAT_TYPE_INT32:
        return K_INT32
    if t == READSTAT_TYPE_FLOAT:
        return K_FLOAT
    if t == READSTAT_TYPE_DOUBLE:
        return K_DOUBLE
    raise ReadstatError(f"unknown readstat type {int(t)}")


@cython.cfunc
def _value_to_python(value: readstat_value_t) -> object:
    """Convert a scalar ``readstat_value_t`` to a Python object (used for labels & ranges)."""
    if readstat_value_is_tagged_missing(value):
        return chr(readstat_value_tag(value))
    t: readstat_type_t = readstat_value_type(value)
    if t == READSTAT_TYPE_STRING or t == READSTAT_TYPE_STRING_REF:
        return _decode(readstat_string_value(value)) or ""
    if t == READSTAT_TYPE_INT8:
        return int(readstat_int8_value(value))
    if t == READSTAT_TYPE_INT16:
        return int(readstat_int16_value(value))
    if t == READSTAT_TYPE_INT32:
        return int(readstat_int32_value(value))
    if t == READSTAT_TYPE_FLOAT:
        return float(readstat_float_value(value))
    return readstat_double_value(value)


# ---------------------------------------------------------------------------
# ColumnBuilder: Arrow-layout buffers for one variable
# ---------------------------------------------------------------------------


@cython.cclass
class ColumnBuilder:
    """Accumulates one column in Arrow physical layout.

    * ``validity`` is an Arrow validity bitmap (LSB bit order).
    * Numeric kinds write into ``data`` through a typed memoryview.
    * Strings write UTF-8 bytes into ``data`` and int64 offsets into ``offsets``
      (``large_string`` layout, so >2 GiB of text per column is fine).

    Buffers are ``bytearray`` so they can grow when the row count is unknown
    up front (.sav files written by non-conforming software report -1 rows).  Typed views are released
    before growing and re-acquired afterwards because a ``bytearray`` cannot
    be resized while a buffer export is alive.
    """

    name: str
    kind: cython.int
    capacity: cython.Py_ssize_t
    length: cython.Py_ssize_t
    null_count: cython.Py_ssize_t

    validity: bytearray
    data: bytearray
    offsets: bytearray

    _valid: cython.uchar[::1]
    _f64: cython.double[::1]
    _f32: cython.float[::1]
    _i8: cython.schar[::1]
    _i16: cython.short[::1]
    _i32: cython.int[::1]
    _off: cython.longlong[::1]

    # Stata tagged missing values (.a-.z): one byte per row, 0 = no tag, 1..26 = a..z.
    # Allocated on the first tag seen, so untagged columns pay nothing.
    tag_codes: bytearray
    _tags: cython.uchar[::1]
    has_tags: cython.bint

    # Which arm of the value handler's dispatch this column takes: its ``kind``,
    # or ``K_NARROWED`` when that kind is not what the file stores (see
    # ``parse(types=...)``). Reading one field and switching on it costs the
    # handler nothing over switching on ``kind`` alone, which is why the narrowing
    # is a kind here rather than a flag to test first.
    store: cython.int

    def __cinit__(self, name: str, kind: cython.int, capacity: cython.Py_ssize_t):
        self.name = name
        self.kind = kind
        self.capacity = 0
        self.length = 0
        self.null_count = 0
        self.tag_codes = bytearray()
        self.has_tags = False
        self.store = kind
        self.validity = bytearray()
        self.data = bytearray()
        self.offsets = bytearray()
        if kind == K_STRING:
            self.offsets = bytearray(8)  # offsets[0] == 0
        self._reserve(max(capacity, 1))

    # -- buffer management --------------------------------------------------

    @cython.cfunc
    def _release_views(self) -> cython.void:
        self._valid = None
        self._tags = None
        self._f64 = None
        self._f32 = None
        self._i8 = None
        self._i16 = None
        self._i32 = None
        self._off = None

    @cython.cfunc
    def _acquire_views(self) -> cython.void:
        self._valid = memoryview(self.validity).cast("B")
        if self.has_tags:
            self._tags = memoryview(self.tag_codes).cast("B")
        k: cython.int = self.kind
        if k == K_STRING:
            self._off = memoryview(self.offsets).cast("q")
        elif k == K_DOUBLE:
            self._f64 = memoryview(self.data).cast("d")
        elif k == K_FLOAT:
            self._f32 = memoryview(self.data).cast("f")
        elif k == K_INT8:
            self._i8 = memoryview(self.data).cast("b")
        elif k == K_INT16:
            self._i16 = memoryview(self.data).cast("h")
        elif k == K_INT32:
            self._i32 = memoryview(self.data).cast("i")

    @cython.cfunc
    def _reserve(self, capacity: cython.Py_ssize_t) -> cython.void:
        """Grow buffers so that ``capacity`` rows fit."""
        if capacity <= self.capacity:
            return
        self._release_views()
        old: cython.Py_ssize_t = self.capacity
        self.validity.extend(bytes((capacity + 7) // 8 - len(self.validity)))
        if self.has_tags:
            self.tag_codes.extend(bytes(capacity - old))
        if self.kind == K_STRING:
            self.offsets.extend(bytes(8 * (capacity - old)))
        else:
            item: cython.Py_ssize_t = _ITEM_SIZE[self.kind]
            self.data.extend(bytes(item * (capacity - old)))
        self.capacity = capacity
        self._acquire_views()

    @cython.cfunc
    @cython.inline
    def ensure_row(self, row: cython.Py_ssize_t) -> cython.void:
        if row >= self.capacity:
            new_cap: cython.Py_ssize_t = self.capacity * 2
            if new_cap <= row:
                new_cap = row + 1
            self._reserve(new_cap)
        if row + 1 > self.length:
            self.length = row + 1

    # -- appending ----------------------------------------------------------

    @cython.cfunc
    @cython.inline
    def _set_valid(self, row: cython.Py_ssize_t) -> cython.void:
        self._valid[row >> 3] |= cython.cast(cython.uchar, 1 << (row & 7))

    @cython.cfunc
    @cython.inline
    def set_null(self, row: cython.Py_ssize_t) -> cython.void:
        self.ensure_row(row)
        self.null_count += 1
        if self.kind == K_STRING:
            self._off[row + 1] = self._off[row]
        # numeric buffers are zero-initialised; the bit stays 0

    @cython.cfunc
    def set_tagged_null(self, row: cython.Py_ssize_t, tag: cython.char) -> cython.void:
        self.set_null(row)
        if not self.has_tags:
            self._release_views()
            self.tag_codes = bytearray(self.capacity)
            self.has_tags = True
            self._acquire_views()
        self._tags[row] = cython.cast(cython.uchar, tag - 96)  # 'a' -> 1

    @cython.cfunc
    @cython.inline
    def set_double(self, row: cython.Py_ssize_t, v: cython.double) -> cython.void:
        self.ensure_row(row)
        self._f64[row] = v
        self._set_valid(row)

    @cython.cfunc
    @cython.inline
    def set_float(self, row: cython.Py_ssize_t, v: cython.float) -> cython.void:
        self.ensure_row(row)
        self._f32[row] = v
        self._set_valid(row)

    @cython.cfunc
    @cython.inline
    def set_int8(self, row: cython.Py_ssize_t, v: cython.schar) -> cython.void:
        self.ensure_row(row)
        self._i8[row] = v
        self._set_valid(row)

    @cython.cfunc
    @cython.inline
    def set_int16(self, row: cython.Py_ssize_t, v: cython.short) -> cython.void:
        self.ensure_row(row)
        self._i16[row] = v
        self._set_valid(row)

    @cython.cfunc
    @cython.inline
    def set_int32(self, row: cython.Py_ssize_t, v: cython.int) -> cython.void:
        self.ensure_row(row)
        self._i32[row] = v
        self._set_valid(row)

    @cython.cfunc
    def set_string(self, row: cython.Py_ssize_t, s: cython.p_const_char) -> cython.void:
        self.ensure_row(row)
        n: cython.size_t = 0 if s is cython.NULL else strlen(s)
        if n > 0:
            chunk: bytes = s[:n]
            self.data += chunk
        self._off[row + 1] = self._off[row] + cython.cast(cython.longlong, n)
        self._set_valid(row)

    @cython.cfunc
    def set_checked(self, row: cython.Py_ssize_t, v: cython.double) -> cython.int:
        """Store ``v`` in a type narrower than the file's, refusing to corrupt it.

        The C conversions ReadStat offers (``readstat_int8_value`` and friends)
        cast without checking, so a value outside the requested range would wrap
        silently. Returns ``FITS``, ``NOT_INTEGRAL`` or ``OUT_OF_RANGE``; the
        caller raises, because a ``cfunc`` cannot propagate an exception here.
        """
        k: cython.int = self.kind
        if k == K_DOUBLE:
            self.set_double(row, v)
            return FITS
        if k == K_FLOAT:
            # NaN and infinity are their own float32 values; only finite ones can overflow.
            if isfinite(v) and (v > _FLOAT_MAX or v < -_FLOAT_MAX):
                return OUT_OF_RANGE
            self.set_float(row, cython.cast(cython.float, v))
            return FITS
        if not isfinite(v) or floor(v) != v:
            return NOT_INTEGRAL
        if k == K_INT8:
            if v < _INT8_MIN or v > _INT8_MAX:
                return OUT_OF_RANGE
            self.set_int8(row, cython.cast(cython.schar, v))
        elif k == K_INT16:
            if v < _INT16_MIN or v > _INT16_MAX:
                return OUT_OF_RANGE
            self.set_int16(row, cython.cast(cython.short, v))
        elif k == K_INT32:
            if v < _INT32_MIN or v > _INT32_MAX:
                return OUT_OF_RANGE
            self.set_int32(row, cython.cast(cython.int, v))
        else:
            return OUT_OF_RANGE  # K_STRING, which _requested_kind never allows here
        return FITS

    # -- export -------------------------------------------------------------

    def to_arrow(self) -> pa.Array:
        """Wrap the accumulated buffers as a ``pyarrow.Array`` (no copy)."""
        n = self.length
        self._release_views()
        validity = None
        if self.null_count > 0:
            validity = pa.py_buffer(self.validity).slice(0, (n + 7) // 8)
        typ = _ARROW_TYPE[self.kind]
        if self.kind == K_STRING:
            offsets = pa.py_buffer(self.offsets).slice(0, 8 * (n + 1))
            data = pa.py_buffer(self.data)
            return pa.Array.from_buffers(typ, n, [validity, offsets, data], self.null_count)
        item = _ITEM_SIZE[self.kind]
        data = pa.py_buffer(self.data).slice(0, item * n)
        return pa.Array.from_buffers(typ, n, [validity, data], self.null_count)

    def tags_to_arrow(self) -> pa.Array | None:
        """Tag letters as ``dictionary<int8, string>``, null = untagged; ``None`` when there are no tags."""
        if not self.has_tags:
            return None
        n = self.length
        self._release_views()
        codes = pa.Array.from_buffers(pa.uint8(), n, [None, pa.py_buffer(self.tag_codes).slice(0, n)])
        indices = pc.subtract(codes, 1).cast(pa.int8())  # 0 -> -1, masked out below
        indices = pc.if_else(pc.equal(codes, 0), pa.scalar(None, pa.int8()), indices)
        return pa.DictionaryArray.from_arrays(indices, TAG_LETTERS)


# ---------------------------------------------------------------------------
# StatsAccumulator: one column summarised in constant memory
# ---------------------------------------------------------------------------


@cython.cclass
class StatsAccumulator:
    """What a scanning pass keeps of a column instead of its values.

    Five scalars per column: the value range, whether
    every value was a whole number, and whether every value survives float32.
    Together they say what the narrowest lossless Arrow type is, which a second
    pass can then read the column into (see ``parse(types=...)``). Nothing else
    about a column is worth a pass over the file, so nothing else is kept.
    """

    name: str
    kind: cython.int
    min_v: cython.double
    max_v: cython.double
    has_value: cython.bint  # whether min_v/max_v mean anything yet
    all_integral: cython.bint
    float32_exact: cython.bint

    def __cinit__(self, name: str, kind: cython.int):
        self.name = name
        self.kind = kind
        self.min_v = 0.0
        self.max_v = 0.0
        self.has_value = False
        self.all_integral = True
        self.float32_exact = True

    @cython.cfunc
    def observe(self, value: readstat_value_t) -> cython.void:
        # Every numeric kind reads as a double without loss (int32 included: 2^31
        # is well inside the 53-bit mantissa), so one accumulator covers them all.
        v: cython.double = readstat_double_value(value)
        if not isfinite(v):
            # No integer type holds a NaN or an infinity, and neither bounds it.
            self.all_integral = False
            return
        if not self.has_value:
            self.has_value = True
            self.min_v = v
            self.max_v = v
        else:
            if v < self.min_v:
                self.min_v = v
            if v > self.max_v:
                self.max_v = v
        if self.all_integral and floor(v) != v:
            self.all_integral = False
        if self.float32_exact and cython.cast(cython.double, cython.cast(cython.float, v)) != v:
            self.float32_exact = False

    def to_dict(self) -> dict:
        """The summary as plain Python, for :mod:`readstat_arrow.reader` to pick a type from."""
        return {
            "name": self.name,
            "type": _ARROW_TYPE[self.kind],
            "min": self.min_v if self.has_value else None,
            "max": self.max_v if self.has_value else None,
            "all_integral": bool(self.all_integral),
            "float32_exact": bool(self.float32_exact),
        }


def requested_kind(name: str, requested: object, stored: cython.int) -> object:
    """The column kind for the Arrow type ``requested`` for variable ``name``.

    A plain ``def``, not a ``cfunc``: it runs once per variable and is allowed to
    raise, which the variable callback turns into an aborted parse.
    """
    kind = _KIND_OF_ARROW.get(requested)
    if kind is None:
        raise ValueError(
            f"types[{name!r}]: {requested} is not a type a column can be read into; "
            f"choose one of {', '.join(str(t_) for t_ in _KIND_OF_ARROW)}"
        )
    if (kind == K_STRING) != (stored == K_STRING):
        raise ValueError(
            f"types[{name!r}]: {requested} cannot hold this variable, which the file "
            f"stores as {_ARROW_TYPE[stored]}"
        )
    return kind


# ---------------------------------------------------------------------------
# ParseContext: everything the callbacks need, passed to ReadStat as void*
# ---------------------------------------------------------------------------


@cython.cclass
class ParseContext:
    builders: list  # list[ColumnBuilder], indexed by index_after_skipping
    stats: list  # list[StatsAccumulator] instead of builders, when scanning
    variables: list  # variable names, same order
    column_types: list  # pyarrow type per variable, same order
    # One dict per Metadata mapping, filled only where the file declares something.
    variable_labels: dict
    formats: dict
    storage_widths: dict
    display_widths: dict
    measures: dict
    alignments: dict
    missing_values: dict
    label_set_of: dict  # variable name -> ReadStat label-set name
    columns: object  # frozenset[str] | None - selection filter
    types: object  # dict[str, pa.DataType] | None - per-variable type request
    metadata_only: cython.bint
    scan: cython.bint  # summarise the values instead of storing them
    preserve_user_missing: cython.bint
    num_rows: cython.Py_ssize_t  # -1 when unknown
    rows_seen: cython.Py_ssize_t
    # Batching: hand each ``batch_rows`` rows to ``on_batch`` and start over, so
    # the buffers never hold more than one batch. 0 means accumulate the lot.
    batch_rows: cython.Py_ssize_t
    batch_start: cython.Py_ssize_t  # absolute row index the builders start at
    on_batch: object  # callable(arrays, tags) | None
    file_label: object
    multiple_response_sets: list
    # ReadStat label-set name -> list[Code], created on first mention by either the
    # variable or the value-label callback (ReadStat does not fix which comes first).
    # Variables sharing a set get their own copy of it at the end.
    label_sets: dict
    notes: list
    warnings: list
    error: object  # exception raised inside a callback, re-raised after parse

    def __cinit__(self):
        self.builders = []
        self.stats = []
        self.variables = []
        self.column_types = []
        self.variable_labels = {}
        self.formats = {}
        self.storage_widths = {}
        self.display_widths = {}
        self.measures = {}
        self.alignments = {}
        self.missing_values = {}
        self.label_set_of = {}
        self.columns = None
        self.types = None
        self.metadata_only = False
        self.scan = False
        self.preserve_user_missing = False
        self.num_rows = -1
        self.rows_seen = 0
        self.batch_rows = 0
        self.batch_start = 0
        self.on_batch = None
        self.file_label = None
        self.multiple_response_sets = []
        self.label_sets = {}
        self.notes = []
        self.warnings = []
        self.error = None


@cython.cfunc
def _record(mapping: dict, name: str, value: object) -> object:
    """Store ``value`` under ``name`` unless it declares nothing (None, 0, "unknown")."""
    if value is None or value == 0 or value == "unknown":
        return None
    mapping[name] = value
    return None


@cython.cfunc
def _label_set(ctx: ParseContext, name: str) -> list:
    """The Code list of ReadStat label set ``name``, created empty on first mention."""
    codes = ctx.label_sets.get(name)
    if codes is None:
        codes = ctx.label_sets[name] = []
    return codes


# ---------------------------------------------------------------------------
# ReadStat callbacks.  They are ``noexcept`` so their signatures match the C
# function-pointer types exactly; any Python exception is stashed on the
# context and turned into READSTAT_HANDLER_ABORT.
# ---------------------------------------------------------------------------


@cython.cfunc
@cython.exceptval(check=False)
def _handle_metadata(meta: cython.pointer(readstat_metadata_t), vctx: cython.p_void) -> cython.int:
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        ctx.num_rows = readstat_get_row_count(meta)
        mr_sets = []
        n_mr: cython.size_t = readstat_get_multiple_response_sets_length(meta)
        mr: cython.pointer(mr_set_t) = cython.cast(
            cython.pointer(mr_set_t), readstat_get_multiple_response_sets(meta)
        )
        i: cython.size_t
        j: cython.int
        for i in range(n_mr):
            mr_sets.append(
                {
                    "name": _decode(mr[i].name),
                    "label": _decode(mr[i].label),
                    "type": chr(mr[i].type),
                    "is_dichotomy": bool(mr[i].is_dichotomy),
                    "counted_value": None if mr[i].counted_value == -1 else mr[i].counted_value,
                    "variables": [_decode(mr[i].subvariables[j]) for j in range(mr[i].num_subvars)],
                }
            )
        ctx.multiple_response_sets = mr_sets
        ctx.file_label = _decode(readstat_get_file_label(meta))
        return READSTAT_HANDLER_OK
    except BaseException as exc:  # must not leak into C
        ctx.error = exc
        return READSTAT_HANDLER_ABORT


@cython.cfunc
@cython.exceptval(check=False)
def _handle_variable(
    index: cython.int,
    variable: cython.pointer(readstat_variable_t),
    val_labels: cython.p_const_char,
    vctx: cython.p_void,
) -> cython.int:
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        name = _decode(readstat_variable_get_name(variable)) or f"V{index}"
        if ctx.columns is not None and name not in ctx.columns:
            return READSTAT_HANDLER_SKIP_VARIABLE

        kind: cython.int = _kind_for_type(readstat_variable_get_type(variable))
        narrowed: cython.bint = False
        if ctx.types is not None:
            wanted = ctx.types.get(name)
            if wanted is not None:
                requested: cython.int = requested_kind(name, wanted, kind)
                narrowed = requested != kind
                kind = requested
        n_ranges: cython.int = readstat_variable_get_missing_ranges_count(variable)
        i: cython.int
        missing_ranges = [
            (
                _value_to_python(readstat_variable_get_missing_range_lo(variable, i)),
                _value_to_python(readstat_variable_get_missing_range_hi(variable, i)),
            )
            for i in range(n_ranges)
        ]
        # ReadStat reports a discrete missing value as a range with lo == hi; a .sav
        # declares either up to three of those, or one that spans plus one discrete.
        discrete = [lo for lo, hi in missing_ranges if lo == hi]
        span = next(((lo, hi) for lo, hi in missing_ranges if lo != hi), None)
        if span is not None:
            missing = {"lo": span[0], "hi": span[1]}
            if discrete:
                missing["value"] = discrete[0]
        elif discrete:
            missing = {"values": discrete}
        else:
            missing = None

        # Every mapping records only what the file actually declares.
        ctx.variables.append(name)
        ctx.column_types.append(_ARROW_TYPE[kind])
        _record(ctx.variable_labels, name, _decode(readstat_variable_get_label(variable)))
        _record(ctx.formats, name, _decode(readstat_variable_get_format(variable)))
        _record(ctx.storage_widths, name, int(readstat_variable_get_storage_width(variable)))
        _record(ctx.display_widths, name, readstat_variable_get_display_width(variable))
        _record(ctx.measures, name, _MEASURE_NAMES.get(readstat_variable_get_measure(variable)))
        _record(ctx.alignments, name, _ALIGNMENT_NAMES.get(readstat_variable_get_alignment(variable)))
        _record(ctx.missing_values, name, missing)
        set_name = _decode(val_labels)
        if set_name is not None:
            ctx.label_set_of[name] = set_name
            _label_set(ctx, set_name)  # the value-label callback may not have run yet
        if ctx.scan:
            ctx.stats.append(StatsAccumulator(name, kind))
        elif not ctx.metadata_only:
            # Batching fixes the capacity at one batch; otherwise the whole file
            # has to fit, which is a guess when the header does not say how long it is.
            capacity: cython.Py_ssize_t = ctx.batch_rows
            if capacity == 0:
                capacity = ctx.num_rows if ctx.num_rows >= 0 else _INITIAL_CAPACITY_UNKNOWN_ROWS
            builder: ColumnBuilder = ColumnBuilder(name, kind, capacity)
            if narrowed:
                builder.store = K_NARROWED
            ctx.builders.append(builder)
        return READSTAT_HANDLER_OK
    except BaseException as exc:
        ctx.error = exc
        return READSTAT_HANDLER_ABORT


@cython.cfunc
@cython.inline
def _reads_as_null(
    ctx: ParseContext, value: readstat_value_t, variable: cython.pointer(readstat_variable_t)
) -> cython.bint:
    """Whether a read would store this value as a null rather than keep it.

    Only a scan asks: the types it reports have to fit the values the read that
    follows it actually stores, which is why it counts missingness the same way.
    """
    if not readstat_value_is_missing(value, variable):
        return False
    if readstat_value_is_tagged_missing(value):
        return True
    return not (ctx.preserve_user_missing and readstat_value_is_defined_missing(value, variable))


@cython.cfunc
@cython.exceptval(check=False)
def _handle_scan_value(
    obs_index: cython.int,
    variable: cython.pointer(readstat_variable_t),
    value: readstat_value_t,
    vctx: cython.p_void,
) -> cython.int:
    """The value handler of a scanning pass: summarise the value, store nothing.

    A handler of its own rather than a branch in :func:`_handle_value`, which
    runs once per cell of a real read and is the one function in this file whose
    size is worth minding.
    """
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        row: cython.Py_ssize_t = obs_index
        if row + 1 > ctx.rows_seen:
            ctx.rows_seen = row + 1
        acc: StatsAccumulator = cython.cast(
            StatsAccumulator, ctx.stats[readstat_variable_get_index_after_skipping(variable)]
        )
        # A string column is read as it is stored whatever it holds, so its
        # values are not even looked at.
        if acc.kind != K_STRING and not _reads_as_null(ctx, value, variable):
            acc.observe(value)
        return READSTAT_HANDLER_OK
    except BaseException as exc:
        ctx.error = exc
        return READSTAT_HANDLER_ABORT


@cython.cfunc
@cython.exceptval(-1, check=True)
def _narrowing_failed(
    col: ColumnBuilder, row: cython.Py_ssize_t, v: cython.double, outcome: cython.int
) -> cython.int:
    """Raise for a value the requested type cannot hold; never returns.

    Out of line on purpose: building the message needs Python objects, and inside
    :func:`_handle_value` that machinery would sit in the middle of the path every
    cell of every read takes.
    """
    reason = "is not a whole number" if outcome == NOT_INTEGRAL else "is out of range"
    raise ReadstatError(
        f"{col.name}: row {row} holds {v!r}, which {reason} for the requested {_ARROW_TYPE[col.kind]}"
    )


def _emit_batch(ctx: ParseContext, n_rows: cython.Py_ssize_t) -> None:
    """Hand ``n_rows`` rows to ``on_batch`` and start the builders over.

    The builders are replaced rather than rewound: :meth:`ColumnBuilder.to_arrow`
    hands their buffers to Arrow without copying, so the batch that just left
    owns them now.
    """
    arrays = []
    tags = []
    fresh = []
    for b in ctx.builders:
        col: ColumnBuilder = cython.cast(ColumnBuilder, b)
        if col.length < n_rows:  # trailing rows whose cells never arrived
            col.ensure_row(n_rows - 1)
        arrays.append(col.to_arrow())
        tags.append(col.tags_to_arrow())
        new: ColumnBuilder = ColumnBuilder(col.name, col.kind, ctx.batch_rows)
        new.store = col.store
        fresh.append(new)
    ctx.builders = fresh
    ctx.on_batch(arrays, tags)


@cython.cfunc
@cython.exceptval(check=False)
def _handle_value(
    obs_index: cython.int,
    variable: cython.pointer(readstat_variable_t),
    value: readstat_value_t,
    vctx: cython.p_void,
) -> cython.int:
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        col_index: cython.int = readstat_variable_get_index_after_skipping(variable)
        if obs_index + 1 > ctx.rows_seen:
            ctx.rows_seen = obs_index + 1
        # Rows are indexed from the start of the current batch, which without
        # batching is the start of the file.
        row: cython.Py_ssize_t = obs_index - ctx.batch_start
        if row >= ctx.batch_rows and ctx.batch_rows > 0:
            # The first cell of the row past the batch: every earlier row is complete.
            _emit_batch(ctx, ctx.batch_rows)
            ctx.batch_start = obs_index
            row = 0

        col: ColumnBuilder = cython.cast(ColumnBuilder, ctx.builders[col_index])
        if readstat_value_is_missing(value, variable):
            if readstat_value_is_tagged_missing(value):
                col.set_tagged_null(row, readstat_value_tag(value))
                return READSTAT_HANDLER_OK
            if not (ctx.preserve_user_missing and readstat_value_is_defined_missing(value, variable)):
                col.set_null(row)
                return READSTAT_HANDLER_OK
            # preserve_user_missing=True: fall through and keep the defined-missing value

        k: cython.int = col.store
        if k == K_DOUBLE:
            col.set_double(row, readstat_double_value(value))
        elif k == K_STRING:
            col.set_string(row, readstat_string_value(value))
        elif k == K_INT32:
            col.set_int32(row, readstat_int32_value(value))
        elif k == K_INT16:
            col.set_int16(row, readstat_int16_value(value))
        elif k == K_INT8:
            col.set_int8(row, readstat_int8_value(value))
        elif k == K_FLOAT:
            col.set_float(row, readstat_float_value(value))
        else:  # K_NARROWED
            v: cython.double = readstat_double_value(value)
            outcome: cython.int = col.set_checked(row, v)
            if outcome != FITS:
                _narrowing_failed(col, row, v, outcome)
        return READSTAT_HANDLER_OK
    except BaseException as exc:
        ctx.error = exc
        return READSTAT_HANDLER_ABORT


@cython.cfunc
@cython.exceptval(check=False)
def _handle_value_label(
    val_labels: cython.p_const_char,
    value: readstat_value_t,
    label: cython.p_const_char,
    vctx: cython.p_void,
) -> cython.int:
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        codes: list = _label_set(ctx, _decode(val_labels) or "")
        codes.append({"value": _value_to_python(value), "label": _decode(label) or ""})
        return READSTAT_HANDLER_OK
    except BaseException as exc:
        ctx.error = exc
        return READSTAT_HANDLER_ABORT


@cython.cfunc
@cython.exceptval(check=False)
def _handle_note(note_index: cython.int, note: cython.p_const_char, vctx: cython.p_void) -> cython.int:
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        ctx.notes.append(_decode(note) or "")
        return READSTAT_HANDLER_OK
    except BaseException as exc:
        ctx.error = exc
        return READSTAT_HANDLER_ABORT


@cython.cfunc
@cython.exceptval(check=False)
def _handle_error(message: cython.p_const_char, vctx: cython.p_void) -> cython.void:
    ctx: ParseContext = cython.cast(ParseContext, vctx)
    try:
        ctx.warnings.append(_decode(message) or "")
    except BaseException as exc:
        ctx.error = exc


# ---------------------------------------------------------------------------
# Data source: ReadStat asks for bytes, we take them from a Python file object.
#
# ReadStat reaches the file through five function pointers (open/close/seek/
# read/update) and an opaque io_ctx; replacing its default unistd set with these
# is all it takes to parse something that never was a path on disk.
# ---------------------------------------------------------------------------


@cython.cclass
class _Source:
    file: object
    readinto: object  # bound `file.readinto`, or None: then read() + a copy
    base: cython.Py_ssize_t  # the file object's position when parsing began
    error: object

    def __cinit__(self, file: object):
        self.file = file
        self.readinto = getattr(file, "readinto", None)
        self.base = 0
        self.error = None


@cython.cfunc
@cython.exceptval(check=False)
def _io_open(path: cython.p_const_char, vctx: cython.p_void) -> cython.int:
    """Anchor at wherever the file object currently is; ``path`` is unused."""
    src: _Source = cython.cast(_Source, vctx)
    try:
        src.base = src.file.tell()
        return 0
    except BaseException as exc:  # must not leak into C
        src.error = exc
        return -1


@cython.cfunc
@cython.exceptval(check=False)
def _io_close(vctx: cython.p_void) -> cython.int:
    return 0  # the file object belongs to the caller, who closes it


@cython.cfunc
@cython.exceptval(check=False)
def _io_seek(offset: readstat_off_t, whence: readstat_io_flags_t, vctx: cython.p_void) -> readstat_off_t:
    """Seek, in offsets relative to :attr:`_Source.base`, and report the new one."""
    src: _Source = cython.cast(_Source, vctx)
    try:
        if whence == READSTAT_SEEK_SET:
            position = src.file.seek(src.base + offset, 0)
        elif whence == READSTAT_SEEK_CUR:
            position = src.file.seek(offset, 1)
        elif whence == READSTAT_SEEK_END:
            position = src.file.seek(offset, 2)
        else:
            return -1
        return position - src.base
    except BaseException as exc:
        src.error = exc
        return -1


@cython.cfunc
@cython.exceptval(check=False)
def _io_read(buf: cython.p_void, nbyte: cython.size_t, vctx: cython.p_void) -> cython.ssize_t:
    """Fill ``buf`` with up to ``nbyte`` bytes, short only at end of file.

    ReadStat reads a record at a time and treats a short read as the end of the
    file, so a stream that hands back less than it was asked for without being
    exhausted - a raw unbuffered file, a socket - is read round again here.
    """
    src: _Source = cython.cast(_Source, vctx)
    total: cython.size_t = 0
    n: cython.size_t
    try:
        while total < nbyte:
            if src.readinto is not None:
                view = PyMemoryView_FromMemory(
                    cython.cast(cython.p_char, buf) + total, nbyte - total, PyBUF_WRITE
                )
                n = src.readinto(view)
            else:
                chunk: bytes = src.file.read(nbyte - total)
                n = len(chunk)
                if n > 0:
                    memcpy(cython.cast(cython.p_char, buf) + total, cython.cast(cython.p_char, chunk), n)
            if n == 0:
                break
            total += n
        return total
    except BaseException as exc:
        src.error = exc
        return -1


@cython.cfunc
@cython.exceptval(check=False)
def _io_update(
    file_size: cython.long,
    progress: readstat_progress_handler,
    user_ctx: cython.p_void,
    vctx: cython.p_void,
) -> readstat_error_t:
    """Progress reporting, which we do not offer; ReadStat's default would read our io_ctx as a fd."""
    return READSTAT_OK


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FORMATS = ("sav", "dta")


@cython.cfunc
def _check(rc: readstat_error_t) -> cython.void:
    if rc != READSTAT_OK:
        raise ReadstatError(_decode(readstat_error_message(rc)) or f"readstat error {int(rc)}")


def parse(
    path: bytes | None,
    file_format: FileFormat,
    *,
    file=None,
    metadata_only: bool = False,
    scan: bool = False,
    columns=None,
    types=None,
    row_limit: int = 0,
    row_offset: int = 0,
    encoding: str | None = None,
    preserve_user_missing: bool = False,
    batch_rows: int = 0,
    on_batch=None,
) -> tuple[list, list, object, Metadata, int | None, list, list]:
    """Parse ``path``, or the seekable binary ``file`` object, with ReadStat.

    Exactly one of ``path`` and ``file`` is used: given ``file``, ReadStat reads
    through it from its current position instead of opening ``path``.

    ``metadata_only`` stops before the values; ``scan`` walks them but keeps only
    each column's summary (see :class:`StatsAccumulator`) rather than the data.
    ``types`` maps a variable name to the Arrow type to read it into instead of
    the one the file stores it as, which is how a scan's findings are used.

    ``batch_rows`` with ``on_batch`` reads in batches: every ``batch_rows`` rows
    (and once more for the remainder) ``on_batch(arrays, tags)`` is called with
    that slice of the columns and the builders start over, so no more than one
    batch is ever held. ``arrays`` and ``tags`` in the return value are then
    empty - the batches were the data. Anything ``on_batch`` raises aborts the
    parse and propagates.

    Returns ``(arrays, tags, schema, metadata, num_rows, warnings, stats)``: one
    ``pyarrow.Array`` per column and, per column, its Stata tagged-missing letters
    as a dictionary array or ``None`` (both lists empty when ``metadata_only`` or
    ``scan``); the ``pyarrow.Schema`` of the variables as they are read, filled in
    even when ``metadata_only``; the file's :class:`Metadata`; the row count
    (``None`` when the header does not record it and no rows were read);
    ReadStat's recoverable-problem messages; and one summary dict per column when
    ``scan``, else an empty list.
    """
    if file_format not in FORMATS:
        raise ValueError(f"file_format must be one of {FORMATS}, got {file_format!r}")
    if (path is None) == (file is None):
        raise ValueError("pass exactly one of path and file")
    if metadata_only and scan:
        raise ValueError("pass at most one of metadata_only and scan")
    if (batch_rows > 0) != (on_batch is not None):
        raise ValueError("pass batch_rows and on_batch together, or neither")
    if batch_rows and (metadata_only or scan):
        raise ValueError("batch_rows reads values, so it goes with neither metadata_only nor scan")

    ctx: ParseContext = ParseContext()
    ctx.metadata_only = metadata_only
    ctx.scan = scan
    ctx.preserve_user_missing = preserve_user_missing
    ctx.batch_rows = batch_rows
    ctx.on_batch = on_batch
    if columns is not None:
        ctx.columns = frozenset(columns)
    if types is not None:
        ctx.types = dict(types)

    src: _Source = _Source(file) if file is not None else None

    parser: cython.pointer(readstat_parser_t) = readstat_parser_init()
    if parser is cython.NULL:
        raise MemoryError("readstat_parser_init failed")

    rc: readstat_error_t
    try:
        if src is not None:
            # readstat_set_io_ctx frees the unistd io_ctx these replace.
            _check(readstat_set_open_handler(parser, _io_open))
            _check(readstat_set_close_handler(parser, _io_close))
            _check(readstat_set_seek_handler(parser, _io_seek))
            _check(readstat_set_read_handler(parser, _io_read))
            _check(readstat_set_update_handler(parser, _io_update))
            _check(readstat_set_io_ctx(parser, cython.cast(cython.p_void, src)))

        _check(readstat_set_metadata_handler(parser, _handle_metadata))
        _check(readstat_set_variable_handler(parser, _handle_variable))
        _check(readstat_set_value_label_handler(parser, _handle_value_label))
        _check(readstat_set_note_handler(parser, _handle_note))
        _check(readstat_set_error_handler(parser, _handle_error))
        if scan:
            _check(readstat_set_value_handler(parser, _handle_scan_value))
        elif not metadata_only:
            _check(readstat_set_value_handler(parser, _handle_value))
        if encoding:
            enc: bytes = encoding.encode("ascii")
            _check(readstat_set_file_character_encoding(parser, enc))
        if row_limit:
            _check(readstat_set_row_limit(parser, row_limit))
        if row_offset:
            _check(readstat_set_row_offset(parser, row_offset))

        vctx: cython.p_void = cython.cast(cython.p_void, ctx)
        # With our own open handler the path is never looked at, but ReadStat
        # still hands it along, so it must be a valid pointer.
        path_bytes: bytes = path if path is not None else b""
        c_path: cython.p_const_char = path_bytes
        if file_format == "sav":
            rc = readstat_parse_sav(parser, c_path, vctx)
        elif file_format == "dta":
            rc = readstat_parse_dta(parser, c_path, vctx)
        else:
            t.assert_never(file_format)
    finally:
        readstat_parser_free(parser)

    # A failure in the file object is the cause of whatever ReadStat made of it,
    # so it is reported ahead of both the callbacks' errors and ReadStat's own.
    if src is not None and src.error is not None:
        raise src.error
    if ctx.error is not None:
        raise ctx.error
    _check(rc)

    arrays = []
    tags = []
    if not metadata_only and not scan:
        n_rows: cython.Py_ssize_t = ctx.rows_seen - ctx.batch_start
        if batch_rows:
            if n_rows > 0:  # the remainder; exactly nothing when the rows divided evenly
                _emit_batch(ctx, n_rows)
        else:
            for b in ctx.builders:
                col: ColumnBuilder = cython.cast(ColumnBuilder, b)
                # A column that never received a value (e.g. zero-row file) must still
                # have the right length.
                if col.length < n_rows:
                    col.ensure_row(n_rows - 1)
                arrays.append(col.to_arrow())
                tags.append(col.tags_to_arrow())

    metadata = Metadata(
        variable_labels=ctx.variable_labels,
        # Each variable gets its own copy of the label set it shared during parsing.
        value_labels={
            name: [dict(code) for code in ctx.label_sets[set_name]]
            for name, set_name in ctx.label_set_of.items()
            if ctx.label_sets[set_name]
        },
        formats=ctx.formats,
        storage_widths=ctx.storage_widths,
        display_widths=ctx.display_widths,
        measures=ctx.measures,
        alignments=ctx.alignments,
        missing_values=ctx.missing_values,
        file_label=ctx.file_label,
        notes=ctx.notes,
        multiple_response_sets=ctx.multiple_response_sets,
    )
    num_rows: object = None
    if ctx.num_rows >= 0:
        num_rows = int(ctx.num_rows)
    elif not metadata_only:
        num_rows = int(ctx.rows_seen)
    schema = pa.schema(list(zip(ctx.variables, ctx.column_types, strict=True)))
    stats = [cython.cast(StatsAccumulator, acc).to_dict() for acc in ctx.stats]
    return arrays, tags, schema, metadata, num_rows, ctx.warnings, stats
