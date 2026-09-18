"""Writing Arrow tables to SPSS ``.sav`` and Stata ``.dta`` files.

* :class:`SavWriter` / :class:`DtaWriter` are created with the output location,
  the Arrow schema and the total row count, then fed data with
  :meth:`write_table` / :meth:`write_batch`. Use them as context managers.
* :func:`write_sav` / :func:`write_dta` write one table in a single call and use
  the classes under the hood.

Both take an optional :class:`~readstat_arrow.Metadata`; without one the columns
are written with their Arrow names and nothing else declared.

The row count must be known up front because both file formats store it in the
header, which ReadStat writes before the first row.

Things the formats cannot represent, and what happens to them: null strings are
written as empty strings (neither SPSS nor Stata has a missing string); NaN in a
float column is written as system-missing; a timezone on a timestamp is dropped
after converting to UTC; value-label set names are not stored
"""

from __future__ import annotations

import os
import re
import typing as t
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import TracebackType

import pyarrow as pa
import pyarrow.compute as pc

from readstat_arrow import _dates
from readstat_arrow._cython import writer as _writer
from readstat_arrow._formats import FileFormat
from readstat_arrow.errors import ReadstatWarning
from readstat_arrow.metadata import Metadata, Missingness, Value

__all__ = ["DtaWriter", "SavWriter", "write_dta", "write_sav"]

PathLike = str | os.PathLike[str]

# Column kinds understood by the compiled writer (same numbering as _cython/writer.py).
_K_STRING, _K_INT8, _K_INT16, _K_INT32, _K_FLOAT, _K_DOUBLE = range(6)
_KIND_TYPE = {
    _K_STRING: pa.large_string(),
    _K_INT8: pa.int8(),
    _K_INT16: pa.int16(),
    _K_INT32: pa.int32(),
    _K_FLOAT: pa.float32(),
    _K_DOUBLE: pa.float64(),
}
_MEASURE = {"unknown": 0, "nominal": 1, "ordinal": 2, "scale": 3}
_ALIGNMENT = {"unknown": 0, "left": 1, "center": 2, "right": 3}

# Maximum length, in UTF-8 bytes, of the free-text fields each format can store.
# Longer text is truncated at a character boundary (ReadStat itself would cut
# mid-character) and a ReadstatWarning is emitted.
_TEXT_LIMITS: dict[FileFormat, dict[str, int]] = {
    "sav": {"variable label": 256, "value label": 120, "file label": 64},
    # Stata 118 allows 320 bytes for the dataset label too, but ReadStat's writer holds
    # it in a 256-byte buffer, so that is the effective limit.
    "dta": {"variable label": 320, "value label": 32_000, "file label": 256},
}

# Stata keys value labels with an int32, whatever the labelled variable's type,
# and reserves the top of that range for missing values - so the keys it can
# actually hold are the ones a `long` can (see _STATA_INT_RANGES). A missing
# value is labelled by its tag instead, as a one-letter string.
_LABEL_KEY_MIN, _LABEL_KEY_MAX = -2_147_483_647, 2_147_483_620

# Storage width for string columns when neither the metadata nor the caller says.
_DEFAULT_STRING_WIDTH: dict[FileFormat, int] = {"sav": 255, "dta": 244}

# Stata's native numeric types; everything else is widened or stored as double.
_DTA_KIND: dict[t.Any, int] = {
    pa.int8(): _K_INT8,
    pa.bool_(): _K_INT8,
    pa.int16(): _K_INT16,
    pa.uint8(): _K_INT16,
    pa.int32(): _K_INT32,
    pa.uint16(): _K_INT32,
    pa.float32(): _K_FLOAT,
}


class _Writer:
    """Shared implementation of :class:`SavWriter` and :class:`DtaWriter`."""

    _file_format: FileFormat

    def __init__(
        self,
        where: PathLike | t.IO[bytes],
        schema: pa.Schema,
        row_count: int,
        metadata: Metadata | None = None,
        variable_ranges: Mapping[str, tuple[int, int]] | None = None,
        rename_invalid_names: bool = False,
    ) -> None:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        metadata = metadata if metadata is not None else Metadata()
        self.schema = schema  # what write_batch checks against: the caller's own names
        self.row_count = row_count

        written = schema
        self.renamed_variables: dict[str, str] = {}
        if rename_invalid_names:
            renames = _sanitised_names(schema.names, self._file_format)
            self.renamed_variables = {old: new for old, new in renames.items() if old != new}
            for old, new in self.renamed_variables.items():
                metadata = metadata.rename_variable(old, new)
            written = pa.schema([f.with_name(renames[f.name]) for f in schema])
            _warn_renames(self.renamed_variables, self._file_format)
        self.metadata = metadata

        family = _dates.FAMILY_OF_FORMAT[self._file_format]
        ranges = variable_ranges or {}
        self._plans = [
            _ColumnPlan.build(field, metadata, family, self._file_format, ranges.get(original))
            for original, field in zip(schema.names, written, strict=True)
        ]
        label_sets, self._label_set_names = _plan_label_sets(self._plans, self._file_format)
        file_label = _fit(metadata.file_label, self._file_format, "file label", "file")

        if isinstance(where, str | os.PathLike):
            path = os.fspath(where)
            self._file: t.IO[bytes] = open(os.path.expanduser(path), "wb")  # noqa: SIM115 - closed in close()
            self._owns_file = True
        else:
            self._file = where
            self._owns_file = False

        try:
            self._impl = _writer.Writer(
                self._file,
                self._file_format,
                row_count,
                [p.as_spec(self._label_set_names.get(p.name)) for p in self._plans],
                label_sets,
                file_label,
                list(metadata.notes),
            )
        except BaseException:  # pragma: no cover
            if self._owns_file:
                self._file.close()
            raise
        self._closed = False

    # -- writing --------------------------------------------------------------

    def write_table(self, table: pa.Table) -> None:
        """Append every row of ``table``; its schema must match the writer's."""
        for batch in table.to_batches():
            self.write_batch(batch)

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Append every row of ``batch``; its schema must match the writer's."""
        if not batch.schema.equals(self.schema, check_metadata=False):
            raise ValueError(
                f"batch schema does not match the writer's schema:\n{batch.schema}\n"
                f"--- expected ---\n{self.schema}"
            )
        prepared = [plan.prepare(batch.column(i)) for i, plan in enumerate(self._plans)]
        self._impl.write([values for values, _ in prepared], [tags for _, tags in prepared])

    @property
    def rows_written(self) -> int:
        return int(self._impl.rows_written)

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Finish the file. Raises :class:`ReadstatError` if fewer rows than ``row_count`` were written."""
        if self._closed:
            return
        self._closed = True
        try:
            self._impl.close()
        finally:
            if self._owns_file:
                self._file.close()

    def __enter__(self) -> _Writer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.close()
        elif self._owns_file:  # writing failed: don't leave a truncated file that looks finished
            self._closed = True
            self._file.close()

    def __del__(self) -> None:
        if getattr(self, "_owns_file", False) and not getattr(self, "_closed", True):  # pragma: no cover
            self._file.close()


class SavWriter(_Writer):
    """Write an SPSS ``.sav`` file incrementally.

    Parameters
    ----------
    where:
        Output path or a binary file object.
    schema:
        Arrow schema of the tables/batches that will be written.
    row_count:
        Total number of rows that will be written; SPSS stores it in the header.
    metadata:
        Optional variable labels, formats, value labels, missing values, file
        label and notes. Omit it to declare nothing at all; likewise a column of
        ``schema`` that no mapping mentions is written with nothing declared
        about it. Entries naming a column that is not in ``schema`` are ignored.
    rename_invalid_names:
        Rename any column whose name the format would reject - illegal characters
        become ``_``, a name that cannot start as it does gains a ``v``, a
        reserved word gains a trailing ``_``, an over-long one is cut, and the
        results are made unique. Off by default: an illegal name is an error when
        ReadStat writes the header. The renames are warned about and left on
        :attr:`renamed_variables`; metadata entries follow their variable.

    A string variable is written ``storage_width`` bytes wide, or 255 when the
    variable does not say; a longer value raises :class:`ReadstatError`. Use
    :func:`write_sav` instead to size the columns from the data.

    All numeric Arrow types are written as SPSS doubles; dates, timestamps and
    times become SPSS date/datetime/time variables (see :mod:`readstat_arrow._dates`).
    """

    _file_format = "sav"

    def __init__(
        self,
        where: PathLike | t.IO[bytes],
        schema: pa.Schema,
        row_count: int,
        metadata: Metadata | None = None,
        *,
        rename_invalid_names: bool = False,
    ) -> None:
        super().__init__(where, schema, row_count, metadata, rename_invalid_names=rename_invalid_names)


class DtaWriter(_Writer):
    """Write a Stata ``.dta`` file incrementally.

    Parameters
    ----------
    where, schema, row_count, metadata, rename_invalid_names:
        As for :class:`SavWriter`; the default string width is 244, and Stata's
        naming rules are the stricter pair - letters, digits and ``_`` only, 32
        characters, and its own list of reserved words.
    variable_ranges:
        Minimum and maximum values each integer column will hold, for the columns
        you know. Stata's integer types have asymmetric bounds: ``byte`` is
        -127..100, ``int`` is -32_767..32_740, and ``long`` is -2_147_483_647..
        2_147_483_620 (with the top reserved for missing values and a shifted lower
        bound). A column whose values fall outside its type's range is written as the
        next Stata type with room for it, and incoming batches are cast to that type.
        Pass as ``{column_name: (min, max), ...}``.

    Files are written in Stata format 118 (Stata 14 and later, Unicode).

    ``int8``/``int16``/``int32``/``float32``/``float64`` map to Stata's
    ``byte``/``int``/``long``/``float``/``double``. Other integer types are
    widened or stored as double; booleans become ``byte``.
    """

    _file_format = "dta"

    def __init__(
        self,
        where: PathLike | t.IO[bytes],
        schema: pa.Schema,
        row_count: int,
        metadata: Metadata | None = None,
        *,
        variable_ranges: Mapping[str, tuple[int, int]] | None = None,
        rename_invalid_names: bool = False,
    ) -> None:
        super().__init__(where, schema, row_count, metadata, variable_ranges, rename_invalid_names)


def write_sav(
    where: PathLike | t.IO[bytes],
    table: pa.Table,
    metadata: Metadata | None = None,
    *,
    rename_invalid_names: bool = False,
) -> None:
    """Write ``table`` as an SPSS ``.sav`` file. String widths are taken from the data."""
    metadata = _with_measured_widths(table, metadata)
    with SavWriter(
        where, table.schema, table.num_rows, metadata, rename_invalid_names=rename_invalid_names
    ) as writer:
        writer.write_table(table)


def write_dta(
    where: PathLike | t.IO[bytes],
    table: pa.Table,
    metadata: Metadata | None = None,
    *,
    rename_invalid_names: bool = False,
) -> None:
    """Write ``table`` as a Stata ``.dta`` file.

    String widths are taken from the data, and integer columns holding values outside
    Stata's type bounds (e.g. an ``int8`` with value -128, outside byte's -127 minimum)
    are widened to the next Stata type with room for them, since the data is known here;
    see :func:`_widen_for_stata`. The incremental :class:`DtaWriter` cannot do that and
    reports such values as errors instead.
    """
    table = _widen_for_stata(table)
    metadata = _with_measured_widths(table, metadata)
    with DtaWriter(
        where, table.schema, table.num_rows, metadata, rename_invalid_names=rename_invalid_names
    ) as writer:
        writer.write_table(table)


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


class _ColumnPlan:
    """How one Arrow column becomes one ReadStat variable."""

    def __init__(
        self,
        name: str,
        kind: int,
        metadata: Metadata,
        family: _dates.Family,
        temporal: _dates.TemporalKind | None,
        storage_width: int,
        fmt: str | None,
        file_format: FileFormat,
        tagged: bool,
    ):
        self.name = name
        self.kind = kind
        self.metadata = metadata
        self.family = family
        self.temporal = temporal
        self.storage_width = storage_width
        self.format = fmt
        self.file_format = file_format
        self.tagged = tagged  # column is struct<value, tag> (see the reader's preserve_user_missing)

    @classmethod
    def build(
        cls,
        field: pa.Field,
        metadata: Metadata,
        family: _dates.Family,
        file_format: FileFormat,
        value_range: tuple[int, int] | None = None,
    ) -> _ColumnPlan:
        name = field.name
        typ = field.type
        tagged = _is_tag_struct(typ)
        if tagged:
            if file_format == "sav":
                raise ValueError(
                    f"column {field.name!r}: SPSS files cannot store tagged missing values (.a-.z)"
                )
            typ = typ.field("value").type

        temporal = _dates.kind_of_type(typ)
        fmt = metadata.formats.get(name)
        if fmt and (family == "stata") != fmt.startswith("%"):
            fmt = None  # a format from the other family (e.g. SPSS "F8.2" into Stata); let ReadStat default
        if temporal is not None:
            fmt = (
                fmt
                if fmt and _dates.classify(family, fmt) == temporal
                else _dates.DEFAULT_FORMAT[family][temporal]
            )
            kind = _K_INT32 if (family == "stata" and temporal == "date") else _K_DOUBLE
        elif pa.types.is_string(typ) or pa.types.is_large_string(typ):
            kind = _K_STRING
        elif (
            pa.types.is_null(typ)
            or pa.types.is_integer(typ)
            or pa.types.is_floating(typ)
            or pa.types.is_boolean(typ)
        ):
            kind = _K_DOUBLE if file_format == "sav" else _DTA_KIND.get(typ, _K_DOUBLE)
            if value_range is not None and file_format == "dta" and pa.types.is_integer(typ):
                lo, hi = value_range
                widened = _stata_int_type(typ, lo, hi)
                kind = _DTA_KIND.get(widened, _K_DOUBLE) if widened is not None else kind
            if fmt is None and file_format == "sav" and not pa.types.is_floating(typ):
                # Every SPSS numeric is a double in the file, so ReadStat's own default
                # is F8.2 - which shows a count as "1.00". Integers get no decimals.
                fmt = f"F{metadata.display_widths.get(name) or 8}.0"
        else:
            raise TypeError(f"column {field.name!r}: cannot write Arrow type {typ} to a {file_format} file")

        storage_width = 8
        if kind == _K_STRING:
            declared = metadata.storage_widths.get(name) or _DEFAULT_STRING_WIDTH[file_format]
            storage_width = max(declared, 1)

        discrete, span = _missing_parts(metadata.missing_values.get(name), name)
        if discrete or span is not None:
            if file_format == "dta":
                raise ValueError(
                    f"column {field.name!r}: Stata files cannot store SPSS user-defined missing values"
                )
            if kind == _K_STRING and not all(isinstance(v, str) for v in [*discrete, *(span or ())]):
                raise ValueError(
                    f"column {field.name!r}: missing values of a string variable must be strings"
                )

        if kind == _K_STRING and tagged:
            raise ValueError(f"column {field.name!r}: only numeric variables can have tagged missing values")

        return cls(name, kind, metadata, family, temporal, storage_width, fmt, file_format, tagged)

    def prepare(self, array: pa.Array) -> tuple[pa.Array, pa.Array | None]:
        """Convert ``array`` to the exact Arrow type the compiled writer reads for this kind.

        Returns ``(values, tags)``; ``tags`` is an int8 array of tagged-missing
        codes (1..26 = .a-.z, null = untagged) for struct columns, else ``None``.
        """
        tags = None
        if self.tagged:
            tags = _tag_codes(array, self.name)
            array = _struct_values(array)
        if self.temporal is not None:
            array = _dates.to_raw(array, self.family, self.temporal)
        target = _KIND_TYPE[self.kind]
        if not array.type.equals(target):
            array = pc.cast(array, target)
        return array, tags

    def as_spec(self, label_set: str | None) -> dict[str, t.Any]:
        meta, name = self.metadata, self.name
        discrete, span = _missing_parts(meta.missing_values.get(name), name)
        return {
            "name": name,
            "kind": self.kind,
            "storage_width": self.storage_width,
            "label": _fit(meta.variable_labels.get(name), self.file_format, "variable label", name),
            "format": self.format,
            "label_set": label_set,
            "measure": _MEASURE[meta.measures.get(name) or "unknown"],
            "alignment": _ALIGNMENT[meta.alignments.get(name) or "unknown"],
            "display_width": meta.display_widths.get(name) or 0,  # 0 lets ReadStat pick a width
            "missing_values": [_numeric_or_str(x) for x in discrete],
            "missing_ranges": [] if span is None else [tuple(_numeric_or_str(x) for x in span)],
        }


# ReadStat matches this with sscanf("str%d"), which is happy with a prefix, so a
# trailing "_" would not get "str8" past it: such a name has to be pushed along.
_STATA_STR_TYPE = re.compile(r"str\d", re.IGNORECASE)

# What each format accepts in a variable name, for _sanitised_names. ReadStat
# checks its own version of these when it writes the header (and is laxer than
# Stata on length: it allows 129 bytes where Stata 118 allows 32 characters).
_NAME_RULES: dict[FileFormat, dict[str, t.Any]] = {
    "sav": {
        "extra": "._$@#",
        "first": "@",
        "max_chars": 64,
        "reserved": frozenset(
            {"all", "and", "by", "eq", "ge", "gt", "le", "lt", "ne", "not", "or", "to", "with"}
        ),
        "reserved_prefix": None,
    },
    "dta": {
        "extra": "_",
        "first": "_",
        "max_chars": 32,
        "reserved": frozenset(
            {
                "_all",
                "_b",
                "byte",
                "_coef",
                "_cons",
                "double",
                "float",
                "if",
                "in",
                "int",
                "long",
                "_n",
                "_pi",
                "_pred",
                "_rc",
                "_skip",
                "strl",
                "using",
                "with",
            }
        ),
        "reserved_prefix": _STATA_STR_TYPE,
    },
}
_STATA_STR_TYPE = re.compile(r"str\d+$", re.IGNORECASE)


def _sanitised_names(names: Sequence[str], file_format: FileFormat) -> dict[str, str]:
    """Map every name to one the format accepts; a name already legal maps to itself.

    Illegal characters become ``_`` and a name that cannot start as it does gains
    a ``v``; a reserved word gains a trailing ``_``, except Stata's ``str#``,
    which ReadStat matches as a prefix and so has to be pushed off the front;
    anything over the length limit is cut. Non-ASCII characters are kept - both
    formats take them in a Unicode file. Whatever falls out is then made unique
    against every other name, so distinct columns stay distinct and the legal
    names keep what is theirs.
    """
    rules = _NAME_RULES[file_format]
    taken = {name for name in names if _name_is_legal(name, rules)}
    renames: dict[str, str] = {}
    for name in names:
        if name in taken:
            renames[name] = name
            continue
        clean = "".join(c if _name_char_ok(c, rules["extra"]) else "_" for c in name)
        if not clean or not _name_start_ok(clean[0], rules["first"]) or _reserved_prefix(clean, rules):
            clean = "v" + clean
        if clean.lower() in rules["reserved"]:
            clean += "_"
        clean = clean[: rules["max_chars"]]
        candidate, n = clean, 1
        while candidate in taken:
            n += 1
            suffix = f"_{n}"
            candidate = clean[: rules["max_chars"] - len(suffix)] + suffix
        taken.add(candidate)
        renames[name] = candidate
    return renames


def _name_is_legal(name: str, rules: dict[str, t.Any]) -> bool:
    """Whether the format would take ``name`` as it stands."""
    return bool(
        name
        and len(name) <= rules["max_chars"]
        and all(_name_char_ok(c, rules["extra"]) for c in name)
        and _name_start_ok(name[0], rules["first"])
        and name.lower() not in rules["reserved"]
        and not _reserved_prefix(name, rules)
    )


def _reserved_prefix(name: str, rules: dict[str, t.Any]) -> bool:
    pattern = rules["reserved_prefix"]
    return pattern is not None and pattern.match(name) is not None


def _name_start_ok(char: str, extra: str) -> bool:
    return char.isalpha() or char in extra or not char.isascii()


def _name_char_ok(char: str, extra: str) -> bool:
    """Both formats take any printable non-ASCII character in a Unicode file."""
    if not char.isascii():
        return char.isprintable()
    return char.isalnum() or char in extra


def _is_tag_struct(typ: pa.DataType) -> bool:
    """``struct<value: ..., tag: ...>`` as produced by ``read_dta(..., preserve_user_missing=True)``."""
    return pa.types.is_struct(typ) and typ.num_fields == 2 and [f.name for f in typ] == ["value", "tag"]


def _struct_values(array: pa.Array) -> pa.Array:
    """The ``value`` field, with the struct's own nulls (plain ``.``) applied."""
    values = pc.struct_field(array, "value")
    if array.null_count:
        values = pc.if_else(array.is_null(), pa.scalar(None, values.type), values)
    return values


_TAG_LETTERS = pa.array([chr(c) for c in range(ord("a"), ord("z") + 1)], pa.string())


def _tag_codes(array: pa.Array, name: str) -> pa.Array:
    """Tag letters of a ``struct<value, tag>`` column as int8 codes 1..26 (null = untagged)."""
    tags = pc.struct_field(array, "tag")
    tags = pc.cast(tags, pa.string())
    if array.null_count:  # a null struct is a plain '.', whatever the tag field says
        tags = pc.if_else(array.is_null(), pa.scalar(None, pa.string()), tags)
    if pc.any(pc.and_(tags.is_valid(), pc.struct_field(array, "value").is_valid())).as_py():
        raise ValueError(f"column {name!r}: a cell cannot have both a value and a missing-value tag")
    positions = pc.index_in(tags, value_set=_TAG_LETTERS)  # 'a' -> 0 ... 'z' -> 25, unknown -> null
    if pc.any(pc.and_(tags.is_valid(), positions.is_null())).as_py():
        raise ValueError(f"column {name!r}: missing-value tags must be single letters a-z")
    return pc.cast(pc.add(positions, 1), pa.int8())


def _plan_label_sets(
    plans: list[_ColumnPlan], file_format: FileFormat
) -> tuple[list[dict[str, t.Any]], dict[str, str]]:
    """Turn ``Metadata.value_labels`` into ReadStat label sets.

    Both file formats store value labels as named sets that variables refer to, so
    variables with identical labels share one set, named after the first variable
    that uses it. Returns the set specs and a variable-name -> set-name map.
    """
    specs: list[dict[str, t.Any]] = []
    set_name_for_variable: dict[str, str] = {}
    set_name_for_codes: dict[tuple[tuple[Value, str], ...], str] = {}
    for plan in plans:
        codes = tuple((c["value"], c["label"]) for c in plan.metadata.value_labels.get(plan.name) or [])
        if not codes:
            continue
        set_name = set_name_for_codes.get(codes)
        if set_name is None:
            set_name = set_name_for_codes[codes] = plan.name
            specs.append(_label_set_spec(set_name, codes, file_format))
        set_name_for_variable[plan.name] = set_name
    return specs, set_name_for_variable


def _label_set_spec(
    name: str, codes: tuple[tuple[Value, str], ...], file_format: FileFormat
) -> dict[str, t.Any]:
    values = [value for value, _ in codes]
    texts = [_fit(text, file_format, "value label", f"variable {name!r}") or "" for _, text in codes]
    if file_format == "sav":
        if all(isinstance(v, str) for v in values):
            kind = _K_STRING
        elif all(isinstance(v, int | float) for v in values):
            kind = _K_DOUBLE
        else:
            raise ValueError(f"variable {name!r}: code list mixes string and numeric values")
        return {"name": name, "kind": kind, "labels": list(zip(values, texts, strict=True)), "tags": []}
    elif file_format == "dta":
        labels: list[tuple[int, str]] = []
        tags: list[tuple[str, str]] = []
        for value, text in zip(values, texts, strict=True):
            if isinstance(value, str):
                if len(value) == 1 and "a" <= value <= "z":
                    tags.append((value, text))
                else:
                    raise ValueError(f"variable {name!r}: Stata cannot label string value {value!r}")
            elif not float(value).is_integer():
                raise ValueError(f"variable {name!r}: Stata can only label integer values, got {value!r}")
            elif not _LABEL_KEY_MIN <= value <= _LABEL_KEY_MAX:
                raise ValueError(
                    f"variable {name!r}: Stata can only label values a long can hold "
                    f"({_LABEL_KEY_MIN} to {_LABEL_KEY_MAX}), got {value!r}"
                )
            else:
                labels.append((int(value), text))
        return {"name": name, "kind": _K_INT32, "labels": labels, "tags": tags}
    else:
        t.assert_never(file_format)


def _warn_renames(renamed: dict[str, str], file_format: FileFormat) -> None:
    """One warning for the lot: silently writing different names would be worse."""
    if not renamed:
        return
    shown = ", ".join(f"{old!r} -> {new!r}" for old, new in list(renamed.items())[:3])
    if len(renamed) > 3:
        shown += f", ... +{len(renamed) - 3} more"
    warnings.warn(
        f"renamed {len(renamed)} variable(s) to satisfy {file_format} naming rules: {shown}",
        ReadstatWarning,
        stacklevel=4,  # _warn_renames -> _Writer.__init__ -> SavWriter/DtaWriter -> caller
    )


def _fit(text: str | None, file_format: FileFormat, field: str, owner: str) -> str | None:
    """Truncate ``text`` to the format's byte limit for ``field`` without splitting a character."""
    if text is None:
        return None
    limit = _TEXT_LIMITS[file_format][field]
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    # Decoding with "ignore" drops the partial multibyte sequence at the cut, if any.
    fitted = encoded[:limit].decode("utf-8", "ignore")
    warnings.warn(
        f"{field} of {owner} is {len(encoded)} bytes; {file_format} allows {limit}, truncated to {fitted!r}",
        ReadstatWarning,
        stacklevel=2,
    )
    return fitted


def _missing_parts(missing: Missingness | None, name: str) -> tuple[list[Value], tuple[Value, Value] | None]:
    """The discrete values and the range a variable declares missing, in whichever form.

    Also the only place the two shapes are validated, since a ``TypedDict`` is an
    ordinary dictionary at runtime.
    """
    if missing is None:
        return [], None
    spec = t.cast(Mapping[str, t.Any], missing)  # which keys exist is exactly what we are asking
    if "values" in spec:
        values = spec["values"]
        if len(values) > 3:
            raise ValueError(
                f"variable {name!r}: SPSS declares at most three discrete missing values, got {len(values)}"
            )
        return list(values), None  # an empty list declares nothing, as None does
    if "lo" not in spec or "hi" not in spec:
        raise ValueError(
            f"variable {name!r}: missing values must be {{'values': [...]}} or {{'lo': ..., 'hi': ...}}"
        )
    value = spec.get("value")
    return ([] if value is None else [value]), (spec["lo"], spec["hi"])


def _numeric_or_str(x: t.Any) -> float | str:
    return x if isinstance(x, str) else float(x)


# Stata's legal ranges for byte / int / long; the rest of each type's range is missing values.
_STATA_INT_RANGES: list[tuple[pa.DataType, int, int]] = [
    (pa.int8(), -127, 100),
    (pa.int16(), -32_767, 32_740),
    (pa.int32(), -2_147_483_647, 2_147_483_620),
]


def _stata_int_type(arrow_type: pa.DataType, lo: int, hi: int) -> pa.DataType | None:
    """The Stata type a column of ``arrow_type`` needs to hold values from ``lo`` to ``hi``.

    Starts from the type Stata maps ``arrow_type`` to and only ever moves up, so
    a column is never narrowed. Stata's integer types have both an upper and lower
    bound (the upper reserved for missing values, and a shifted lower bound relative
    to Arrow's native types), so both ``lo`` and ``hi`` matter. ``None`` is returned
    when there is nothing to widen - a type Stata does not have at all
    (``int64``, the unsigned ones), which the writer stores as ``double`` anyway.
    """
    kind = _DTA_KIND.get(arrow_type)
    if kind is None:
        return None
    candidates = _STATA_INT_RANGES[kind - _K_INT8 :]  # kinds are ordered byte, int, long
    return next(
        (typ for typ, lo_ok, hi_ok in candidates if lo_ok <= lo and hi <= hi_ok),
        pa.float64(),
    )


def _widen_for_stata(table: pa.Table) -> pa.Table:
    """Widen integer columns whose values fall outside Stata's type bounds.

    Stata's integer types have asymmetric bounds: ``byte`` -127..100, ``int``
    -32_767..32_740, and ``long`` -2_147_483_647..2_147_483_620 (with the top
    reserved for missing values and a shifted lower bound). An Arrow column whose
    values fall outside the bounds of the Stata type it maps to cannot be written as
    that type and is moved up to the next Stata type with room for it - to ``double``
    when not even a ``long`` fits. This only ever widens: a column whose values fit
    keeps the type it came in with, so an ``int32`` of small numbers stays a ``long``
    rather than being packed into a ``byte``.

    Only :func:`write_dta` can do this, because it has the whole column in hand;
    :class:`DtaWriter` is fixed to the schema it was given and reports a value
    outside Stata's bounds as an error instead.
    """
    for i, field in enumerate(table.schema):
        column = table.column(i)
        tagged = _is_tag_struct(field.type)
        values = pc.struct_field(column, "value") if tagged else column
        if not pa.types.is_integer(values.type) or values.null_count == len(values):
            continue
        min_max = pc.min_max(values).as_py()
        target = _stata_int_type(values.type, min_max["min"], min_max["max"])
        if target is None or target.equals(values.type):
            continue
        if tagged:
            struct = column.combine_chunks()
            widened = pa.StructArray.from_arrays(
                [pc.cast(values, target).combine_chunks(), pc.struct_field(struct, "tag")],
                names=["value", "tag"],
                mask=struct.is_null(),
            )
            table = table.set_column(i, field.name, widened)
        else:
            table = table.set_column(i, field.name, pc.cast(column, target))
    return table


def _with_measured_widths(table: pa.Table, metadata: Metadata | None) -> Metadata:
    """``metadata`` with every string column's storage width sized to the data.

    :func:`write_sav` / :func:`write_dta` hold the whole table, so they can size
    string columns exactly rather than leaving them at the format's 255/244
    default - which the file pads every value out to. The measurement wins over a
    declared width: it is what the data actually needs, and taking the
    declaration instead would let a width read from a file drift upwards on every
    round trip (SPSS rounds it up to whole 8-byte cells, Stata adds a byte for a
    possible NUL). The caller's ``metadata`` is left untouched.
    """
    widths = _string_widths(table)
    metadata = metadata if metadata is not None else Metadata()
    if not widths:
        return metadata
    return replace(metadata, storage_widths={**metadata.storage_widths, **widths})


def _string_widths(table: pa.Table) -> dict[str, int]:
    """Maximum UTF-8 byte length of each string column (at least 1)."""
    widths: dict[str, int] = {}
    for field in table.schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            col = table.column(field.name)
            longest = pc.max(pc.binary_length(pc.cast(col, pa.large_binary()))).as_py()
            widths[field.name] = max(int(longest or 0), 1)
    return widths
