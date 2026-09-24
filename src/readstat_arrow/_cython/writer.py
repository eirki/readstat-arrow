# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True, embedsignature=True
# mypy: ignore-errors
"""Cython (pure Python mode) bridge from Arrow arrays to ReadStat's writer API.

ReadStat writes a file row by row: ``readstat_begin_row``, one
``readstat_insert_*_value`` per variable, ``readstat_end_row``. The
:class:`Writer` below drives that loop straight from Arrow buffers (validity
bitmap, typed data, string offsets) so no per-cell Python object is created.

All *planning* - mapping Arrow types to ReadStat types, converting dates back
to raw numbers, choosing storage widths - happens in Python in
:mod:`readstat_arrow.writer`; this module only receives the finished plan as
plain dicts and executes it.

"""

from __future__ import annotations

import typing as t

import cython
import pyarrow as pa
from cython.cimports.libc.string import memcpy
from cython.cimports.readstat_arrow._cython.readstat import (
    READSTAT_OK,
    READSTAT_TYPE_DOUBLE,
    READSTAT_TYPE_FLOAT,
    READSTAT_TYPE_INT8,
    READSTAT_TYPE_INT16,
    READSTAT_TYPE_INT32,
    READSTAT_TYPE_STRING,
    readstat_add_label_set,
    readstat_add_note,
    readstat_add_variable,
    readstat_alignment_t,
    readstat_begin_row,
    readstat_begin_writing_dta,
    readstat_begin_writing_sav,
    readstat_end_row,
    readstat_end_writing,
    readstat_error_message,
    readstat_error_t,
    readstat_insert_double_value,
    readstat_insert_float_value,
    readstat_insert_int8_value,
    readstat_insert_int16_value,
    readstat_insert_int32_value,
    readstat_insert_missing_value,
    readstat_insert_string_value,
    readstat_insert_tagged_missing_value,
    readstat_label_double_value,
    readstat_label_int32_value,
    readstat_label_set_t,
    readstat_label_string_value,
    readstat_label_tagged_value,
    readstat_measure_t,
    readstat_set_data_writer,
    readstat_variable_add_missing_double_range,
    readstat_variable_add_missing_double_value,
    readstat_variable_add_missing_string_range,
    readstat_variable_add_missing_string_value,
    readstat_variable_set_alignment,
    readstat_variable_set_display_width,
    readstat_variable_set_format,
    readstat_variable_set_label,
    readstat_variable_set_label_set,
    readstat_variable_set_measure,
    readstat_variable_t,
    readstat_writer_free,
    readstat_writer_init,
    readstat_writer_set_file_label,
    readstat_writer_t,
)

from readstat_arrow._formats import FileFormat
from readstat_arrow.errors import ReadstatError

# Column kinds: which readstat_insert_* to call. Shared vocabulary with parser.py.
K_STRING = cython.declare(cython.int, 0)
K_INT8 = cython.declare(cython.int, 1)
K_INT16 = cython.declare(cython.int, 2)
K_INT32 = cython.declare(cython.int, 3)
K_FLOAT = cython.declare(cython.int, 4)
K_DOUBLE = cython.declare(cython.int, 5)

_READSTAT_TYPE = {
    K_STRING: READSTAT_TYPE_STRING,
    K_INT8: READSTAT_TYPE_INT8,
    K_INT16: READSTAT_TYPE_INT16,
    K_INT32: READSTAT_TYPE_INT32,
    K_FLOAT: READSTAT_TYPE_FLOAT,
    K_DOUBLE: READSTAT_TYPE_DOUBLE,
}


@cython.cfunc
def _check(rc: readstat_error_t, context: str = "") -> cython.void:
    if rc != READSTAT_OK:
        msg: bytes = readstat_error_message(rc)
        text = msg.decode("utf-8", "replace")
        raise ReadstatError(f"{text} ({context})" if context else text)


@cython.cfunc
@cython.inline
def _c_str(s: object) -> bytes:
    """Encode ``s`` for ReadStat; ``None`` becomes an empty string."""
    return b"" if s is None else s.encode("utf-8")


# ---------------------------------------------------------------------------
# Data sink: ReadStat hands us bytes, we hand them to a Python file object.
# ---------------------------------------------------------------------------


@cython.cclass
class _Sink:
    file: object
    error: object

    def __cinit__(self, file: object):
        self.file = file
        self.error = None


@cython.cfunc
@cython.exceptval(check=False)
def _data_writer(
    data: cython.p_const_void, length: cython.size_t, vctx: cython.p_void
) -> cython.ssize_t:
    sink: _Sink = cython.cast(_Sink, vctx)
    try:
        chunk: bytes = cython.cast(cython.p_char, data)[:length]
        sink.file.write(chunk)
        return length
    except BaseException as exc:  # must not leak into C
        sink.error = exc
        return -1


# ---------------------------------------------------------------------------
# Column: one Arrow array exposed as typed buffers
# ---------------------------------------------------------------------------


@cython.cclass
class _Column:
    """One Arrow array exposed to the row loop through raw buffer pointers.

    ``load`` takes ``Buffer.address`` for each buffer and keeps a reference to the array so the
    memory stays alive.
    """

    name: str
    kind: cython.int
    variable: cython.pointer(readstat_variable_t)

    # Stata tagged missings for the current batch: int8 codes 1..26 (= .a-.z), null = untagged.
    _tag_array: object
    has_tags: cython.bint
    tag_offset: cython.Py_ssize_t
    tag_codes: cython.p_const_schar
    tag_valid: cython.p_const_uchar  # NULL when every tag cell is non-null

    _array: object  # keeps the buffers alive while the pointers below are in use
    all_null: cython.bint  # every cell is null: skip the bitmap and the value entirely
    offset: cython.Py_ssize_t  # Arrow slice offset into the buffers
    valid: cython.p_const_uchar  # validity bitmap, NULL when there are no nulls
    f64: cython.p_const_double
    f32: cython.p_const_float
    i8: cython.p_const_schar
    i16: cython.p_const_short
    i32: cython.p_const_int
    large_offsets: cython.bint
    off32: cython.p_const_int
    off64: cython.p_const_longlong
    sdata: cython.p_const_uchar
    scratch: bytearray  # NUL-terminated copy of the current string value
    scratch_view: cython.uchar[::1]

    def __cinit__(self, name: str, kind: cython.int):
        self.name = name
        self.kind = kind
        self.variable = cython.NULL
        self._tag_array = None
        self.has_tags = False
        self._array = None
        self.all_null = False
        self.offset = 0
        self.valid = cython.NULL
        self.large_offsets = False
        self.scratch = bytearray(64)
        self.scratch_view = memoryview(self.scratch).cast("B")

    def load(self, array, tags) -> None:
        """Point at ``array``'s buffers (a primitive or (large_)string ``pa.Array``).

        ``tags`` is ``None`` or an int8 ``pa.Array`` of tagged-missing codes (1..26 =
        .a-.z, null = untagged) for the same rows.
        """

        self._tag_array = tags
        self.has_tags = tags is not None and tags.null_count < len(tags)
        if self.has_tags:
            self.tag_offset = tags.offset
            tag_buffers = tags.buffers()
            self.tag_valid = (
                _ptr(tag_buffers[0]) if tags.null_count > 0 else cython.NULL
            )
            self.tag_codes = cython.cast(cython.p_const_schar, _ptr(tag_buffers[1]))

        self._array = array
        self.offset = array.offset
        # Arrow keeps null_count, so a completely empty column - common in real
        # survey data - is recognised here and costs nothing per cell later.
        self.all_null = array.null_count == len(array)
        if self.all_null:
            self.valid = cython.NULL
            return
        buffers = array.buffers()
        validity = buffers[0]
        self.valid = (
            _ptr(validity)
            if (validity is not None and array.null_count > 0)
            else cython.NULL
        )
        if self.kind == K_STRING:
            self.large_offsets = pa.types.is_large_string(array.type)
            if self.large_offsets:
                self.off64 = cython.cast(cython.p_const_longlong, _ptr(buffers[1]))
            else:
                self.off32 = cython.cast(cython.p_const_int, _ptr(buffers[1]))
            self.sdata = _ptr(buffers[2]) if buffers[2] is not None else cython.NULL
        else:
            data: cython.p_const_uchar = _ptr(buffers[1])
            if self.kind == K_DOUBLE:
                self.f64 = cython.cast(cython.p_const_double, data)
            elif self.kind == K_FLOAT:
                self.f32 = cython.cast(cython.p_const_float, data)
            elif self.kind == K_INT8:
                self.i8 = cython.cast(cython.p_const_schar, data)
            elif self.kind == K_INT16:
                self.i16 = cython.cast(cython.p_const_short, data)
            else:
                self.i32 = cython.cast(cython.p_const_int, data)

    @cython.cfunc
    @cython.inline
    def is_null(self, i: cython.Py_ssize_t) -> cython.bint:
        if self.valid is cython.NULL:
            return False
        j: cython.Py_ssize_t = self.offset + i
        return not (self.valid[j >> 3] & (1 << (j & 7)))

    @cython.cfunc
    @cython.inline
    def tag_at(self, i: cython.Py_ssize_t) -> cython.char:
        """The tag letter for row i, or 0 when the cell is not a tagged missing."""
        if not self.has_tags:
            return 0
        j: cython.Py_ssize_t = self.tag_offset + i
        if self.tag_valid is not cython.NULL and not (
            self.tag_valid[j >> 3] & (1 << (j & 7))
        ):
            return 0
        return cython.cast(cython.char, 96 + self.tag_codes[j])  # 1 -> 'a'

    @cython.cfunc
    def string_at(self, i: cython.Py_ssize_t) -> cython.p_char:
        """NUL-terminated pointer to the i-th string (valid until the next call)."""
        j: cython.Py_ssize_t = self.offset + i
        start: cython.Py_ssize_t
        end: cython.Py_ssize_t
        if self.large_offsets:
            start = self.off64[j]
            end = self.off64[j + 1]
        else:
            start = self.off32[j]
            end = self.off32[j + 1]
        n: cython.Py_ssize_t = end - start
        if n + 1 > len(self.scratch):
            self.scratch_view = None
            self.scratch = bytearray(max(n + 1, 2 * len(self.scratch)))
            self.scratch_view = memoryview(self.scratch).cast("B")
        if n > 0:
            memcpy(cython.address(self.scratch_view[0]), self.sdata + start, n)
        self.scratch_view[n] = 0
        return cython.cast(cython.p_char, cython.address(self.scratch_view[0]))


@cython.cfunc
@cython.inline
def _ptr(buffer: object) -> cython.p_const_uchar:
    """Raw pointer to a ``pyarrow.Buffer``'s bytes."""
    address: cython.size_t = buffer.address
    return cython.cast(cython.p_const_uchar, address)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@cython.cclass
class Writer:
    """Drives ``readstat_writer_t`` for one output file.

    ``columns`` and ``label_sets`` are the plan produced by
    :mod:`readstat_arrow.writer` (see there for the dict keys). ``write`` may be
    called repeatedly with one array per column; ``close`` finishes the file
    and verifies that exactly ``num_rows`` rows were written.
    """

    _writer: cython.pointer(readstat_writer_t)
    _sink: _Sink
    _columns: list  # list[_Column]
    _rows_written: cython.Py_ssize_t
    _num_rows: cython.Py_ssize_t
    _closed: cython.bint
    # ReadStat stores the `const char *` of a missing string value without copying
    # it (readstat_variable.c: make_string_value), and reads it back when it emits
    # the header. The encoded bytes must outlive that, so they are kept here.
    _missing_strings: list

    def __cinit__(
        self,
        file: object,
        file_format: FileFormat,
        num_rows: int,
        columns: list,
        label_sets: list,
        file_label: str | None,
        notes: list,
    ):
        self._writer = cython.NULL
        self._columns = []
        self._rows_written = 0
        self._num_rows = num_rows
        self._closed = False
        self._missing_strings = []
        self._sink = _Sink(file)

        w: cython.pointer(readstat_writer_t) = readstat_writer_init()
        if w is cython.NULL:
            raise MemoryError("readstat_writer_init failed")
        self._writer = w
        _check(readstat_set_data_writer(w, _data_writer))

        # -- label sets ---------------------------------------------------
        # Python containers cannot hold C pointers, so the set pointers are kept
        # as integers and cast back when attached to a variable.
        sets: dict = {}
        ls: cython.pointer(readstat_label_set_t)
        for spec in label_sets:
            ls = readstat_add_label_set(
                w, _READSTAT_TYPE[spec["kind"]], _c_str(spec["name"])
            )
            for code, label in spec["labels"]:
                c_label: bytes = _c_str(label)
                if spec["kind"] == K_STRING:
                    readstat_label_string_value(ls, _c_str(code), c_label)
                elif spec["kind"] == K_INT32:
                    readstat_label_int32_value(ls, code, c_label)
                else:
                    readstat_label_double_value(ls, code, c_label)
            for tag, label in spec["tags"]:
                readstat_label_tagged_value(ls, ord(tag), _c_str(label))
            sets[spec["name"]] = cython.cast(cython.size_t, ls)

        # -- variables ----------------------------------------------------
        var: cython.pointer(readstat_variable_t)
        for spec in columns:
            col: _Column = _Column(spec["name"], spec["kind"])
            var = readstat_add_variable(
                w,
                _c_str(spec["name"]),
                _READSTAT_TYPE[spec["kind"]],
                spec["storage_width"],
            )
            col.variable = var
            if spec["label"] is not None:
                readstat_variable_set_label(var, _c_str(spec["label"]))
            if spec["format"] is not None:
                readstat_variable_set_format(var, _c_str(spec["format"]))
            if spec["label_set"] is not None:
                ls_addr: cython.size_t = sets[spec["label_set"]]
                readstat_variable_set_label_set(
                    var, cython.cast(cython.pointer(readstat_label_set_t), ls_addr)
                )
            readstat_variable_set_measure(
                var,
                cython.cast(
                    readstat_measure_t, cython.cast(cython.int, spec["measure"])
                ),
            )
            readstat_variable_set_alignment(
                var,
                cython.cast(
                    readstat_alignment_t, cython.cast(cython.int, spec["alignment"])
                ),
            )
            if spec["display_width"]:
                readstat_variable_set_display_width(var, spec["display_width"])
            for value in spec["missing_values"]:
                if spec["kind"] == K_STRING:
                    c_value: bytes = self._keep(value)
                    _check(
                        readstat_variable_add_missing_string_value(var, c_value),
                        spec["name"],
                    )
                else:
                    _check(
                        readstat_variable_add_missing_double_value(var, value),
                        spec["name"],
                    )
            for lo, hi in spec["missing_ranges"]:
                if spec["kind"] == K_STRING:
                    c_lo: bytes = self._keep(lo)
                    c_hi: bytes = self._keep(hi)
                    _check(
                        readstat_variable_add_missing_string_range(var, c_lo, c_hi),
                        spec["name"],
                    )
                else:
                    _check(
                        readstat_variable_add_missing_double_range(var, lo, hi),
                        spec["name"],
                    )
            self._columns.append(col)

        # -- file-level metadata and header --------------------------------
        if file_label:
            _check(readstat_writer_set_file_label(w, _c_str(file_label)))
        for note in notes:
            readstat_add_note(w, _c_str(note))

        vctx: cython.p_void = cython.cast(cython.p_void, self._sink)
        if file_format == "sav":
            rc = readstat_begin_writing_sav(w, vctx, num_rows)
        elif file_format == "dta":
            rc = readstat_begin_writing_dta(w, vctx, num_rows)
        else:
            t.assert_never(file_format)
        self._raise_sink_error()
        _check(rc, "begin writing")

    def __dealloc__(self):
        if self._writer is not cython.NULL:
            readstat_writer_free(self._writer)
            self._writer = cython.NULL

    @cython.cfunc
    def _keep(self, value: object) -> bytes:
        """Encode ``value`` and hold on to it, for the pointers ReadStat borrows."""
        encoded: bytes = _c_str(value)
        self._missing_strings.append(encoded)
        return encoded

    @cython.cfunc
    def _raise_sink_error(self) -> cython.void:
        if self._sink.error is not None:
            exc = self._sink.error
            self._sink.error = None
            raise exc

    def write(self, arrays: list, tags: list) -> None:
        """Write one row per element of ``arrays`` (one per column).

        ``tags`` has one entry per column: ``None`` or an int8 array of Stata
        tagged-missing codes (1..26) with null for untagged cells.
        """
        if self._closed:
            raise ReadstatError("writer is closed")
        n_cols: cython.Py_ssize_t = len(self._columns)
        if len(arrays) != n_cols:
            raise ValueError(f"expected {n_cols} arrays, got {len(arrays)}")
        n_rows: cython.Py_ssize_t = len(arrays[0]) if n_cols else 0
        if self._rows_written + n_rows > self._num_rows:
            raise ReadstatError(
                f"writer was created for {self._num_rows} rows; "
                f"writing {n_rows} more after {self._rows_written} would exceed that"
            )

        cols: list = self._columns
        c: cython.Py_ssize_t
        for c in range(n_cols):
            cython.cast(_Column, cols[c]).load(arrays[c], tags[c])

        w: cython.pointer(readstat_writer_t) = self._writer
        col: _Column
        i: cython.Py_ssize_t
        k: cython.int
        v: cython.double
        rc: readstat_error_t
        tag: cython.char
        for i in range(n_rows):
            rc = readstat_begin_row(w)
            if rc != READSTAT_OK:
                # ReadStat emits the header (and validates variable names etc.) on the
                # very first begin_row, so an error there is about definitions, not data.
                _check(
                    rc,
                    (
                        "writing header"
                        if self._rows_written + i == 0
                        else f"row {self._rows_written + i}"
                    ),
                )
            for c in range(n_cols):
                col = cython.cast(_Column, cols[c])
                if col.all_null or col.is_null(i):
                    tag = col.tag_at(i)
                    if tag == 0:
                        rc = readstat_insert_missing_value(w, col.variable)
                    else:
                        rc = readstat_insert_tagged_missing_value(w, col.variable, tag)
                else:
                    k = col.kind
                    if k == K_DOUBLE:
                        v = col.f64[col.offset + i]
                        if (
                            v != v
                        ):  # NaN: write as (system) missing rather than a raw NaN
                            rc = readstat_insert_missing_value(w, col.variable)
                        else:
                            rc = readstat_insert_double_value(w, col.variable, v)
                    elif k == K_STRING:
                        rc = readstat_insert_string_value(
                            w, col.variable, col.string_at(i)
                        )
                    elif k == K_INT8:
                        rc = readstat_insert_int8_value(
                            w, col.variable, col.i8[col.offset + i]
                        )
                    elif k == K_INT16:
                        rc = readstat_insert_int16_value(
                            w, col.variable, col.i16[col.offset + i]
                        )
                    elif k == K_INT32:
                        rc = readstat_insert_int32_value(
                            w, col.variable, col.i32[col.offset + i]
                        )
                    else:
                        rc = readstat_insert_float_value(
                            w, col.variable, col.f32[col.offset + i]
                        )
                if rc != READSTAT_OK:
                    _check(rc, f"column {col.name!r}, row {self._rows_written + i}")
            _check(readstat_end_row(w), f"row {self._rows_written + i}")
            self._raise_sink_error()
        self._rows_written += n_rows

    def close(self) -> None:
        """Finish the file. Raises if fewer rows than promised were written."""
        if self._closed:
            return
        self._closed = True
        rc: readstat_error_t = readstat_end_writing(self._writer)
        self._raise_sink_error()
        if rc != READSTAT_OK:
            _check(rc, f"{self._rows_written} of {self._num_rows} rows written")

    @property
    def rows_written(self) -> int:
        return self._rows_written
