"""Writing tests: round trips of tables built here, plus the incremental writer API.

The writers and readers both take a binary file object, so these tests write into
``io.BytesIO`` and read it straight back; only the few tests that are about paths
themselves touch the disk.
"""

from __future__ import annotations

import io
import typing as t
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import readstat_arrow
from conftest import READER_FUNCS, WRITER_CLASSES, WRITER_FUNCS
from readstat_arrow import Code, DtaWriter, Metadata, Missingness
from readstat_arrow._formats import FileFormat

# the first bytes of a finished file
MAGIC: dict[FileFormat, bytes] = {"sav": b"$FL2", "dta": b"<sta"}
# Variable, value and file label limits, as (UTF-8 bytes, characters); None means
# the format caps only the byte length. Stata 118 caps labels at 80 characters too.
LabelLimit = tuple[int, int | None]
LABEL_LIMITS: dict[FileFormat, tuple[LabelLimit, LabelLimit, LabelLimit]] = {
    "sav": ((256, None), (120, None), (64, None)),
    "dta": ((320, 80), (32_000, None), (256, 80)),
}


def _survey() -> tuple[pa.Table, Metadata]:
    """Five rows with one column of every kind SPSS stores, and metadata describing them."""
    table = pa.table(
        {
            "mychar": pa.array(["a", "bb", "", "dd", "ccc"], pa.large_string()),
            "mynum": pa.array([1.5, 2.0, None, -3.25, 0.0]),
            "mylabl": pa.array([1.0, 2.0, 1.0, None, 2.0]),
            "mydate": pa.array(
                [
                    date(2026, 1, 2),
                    date(1999, 12, 31),
                    None,
                    date(1970, 1, 1),
                    date(2000, 2, 29),
                ]
            ),
            "dtime": pa.array(
                [
                    datetime(2026, 1, 2, 3, 4, 5),
                    None,
                    datetime(1999, 12, 31, 23, 59, 59),
                    datetime(1970, 1, 1),
                    datetime(2000, 2, 29, 12, 0),
                ],
                pa.timestamp("us"),
            ),
            "mytime": pa.array(
                [time(0, 0), time(12, 30, 15), None, time(23, 59, 59), time(6, 15)],
                pa.time64("us"),
            ),
        }
    )
    metadata = Metadata(
        variable_labels={
            "mychar": "Character",
            "mynum": "Numeric",
            "mylabl": "Labelled",
            "mydate": "Date",
            "dtime": "Datetime",
            "mytime": "Time",
        },
        value_labels={
            "mylabl": [
                {"value": 1.0, "label": "Male"},
                {"value": 2.0, "label": "Female"},
            ]
        },
        formats={
            "mychar": "A8",
            "mynum": "F8.2",
            "mylabl": "F8.0",
            "mydate": "DATE11",
            "dtime": "DATETIME20",
            "mytime": "TIME8",
        },
        measures={"mychar": "nominal", "mynum": "scale", "mylabl": "nominal"},
        file_label="Tiny survey",
        notes=["written by the test suite"],
    )
    return table, metadata


def _panel() -> tuple[pa.Table, Metadata]:
    """Three rows using Stata's own numeric types, and metadata with Stata formats."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int8()),
            "wave": pa.array([2020, 2021, None], pa.int16()),
            "count": pa.array([100_000, -7, None], pa.int32()),
            "score": pa.array([1.5, None, 2.5], pa.float32()),
            "income": pa.array([1000.0, 2000.5, None]),
            "sex": pa.array(
                [1, 2, None], pa.int32()
            ),  # small values, but the width is kept
            "name": pa.array(["a", "bb", ""], pa.large_string()),
            "day": pa.array([date(2020, 1, 1), None, date(2022, 6, 30)]),
            "when": pa.array(
                [datetime(2020, 1, 1, 12, 0), None, datetime(2022, 6, 30, 23, 59, 59)],
                pa.timestamp("us"),
            ),
        }
    )
    metadata = Metadata(
        variable_labels={
            "id": "Respondent",
            "wave": "Wave",
            "count": "Count",
            "score": "Score",
            "income": "Income",
            "sex": "Sex",
            "name": "Name",
            "day": "Day",
            "when": "Timestamp",
        },
        value_labels={"sex": [{"value": 1, "label": "M"}, {"value": 2, "label": "F"}]},
        formats={"day": "%td", "when": "%tc"},
        file_label="Panel",
    )
    return table, metadata


def test_write_and_read_a_path(fmt: FileFormat, tmp_path: Path) -> None:
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table, metadata = _survey()
    out = tmp_path / f"out.{fmt}"

    write(out, table, metadata)
    back, back_metadata = read(out)

    assert out.exists()
    assert back.equals(table)
    assert back_metadata.file_label == "Tiny survey"


def test_writer_in_batches(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    Writer = WRITER_CLASSES[fmt]
    table, metadata = _survey()
    out = io.BytesIO()

    with Writer(out, table.schema, table.num_rows, metadata) as writer:
        writer.write_table(table.slice(0, 2))
        writer.write_batch(table.slice(2).to_batches()[0])
        assert writer.rows_written == 5

    out.seek(0)
    back, _ = read(out)
    assert back.equals(table)


def test_writer_leaves_the_callers_file_object_open(fmt: FileFormat) -> None:
    Writer = WRITER_CLASSES[fmt]
    table, metadata = _panel()
    out = io.BytesIO()

    with Writer(out, table.schema, table.num_rows, metadata) as writer:
        writer.write_table(table)

    assert not out.closed  # caller's file object is left open
    out.seek(0)
    assert out.read(4) == MAGIC[fmt]


def test_writer_rejects_wrong_schema(fmt: FileFormat) -> None:
    Writer = WRITER_CLASSES[fmt]
    table, metadata = _survey()

    with Writer(io.BytesIO(), table.schema, table.num_rows, metadata) as writer:
        with pytest.raises(ValueError, match="schema"):
            writer.write_table(table.select(["mychar"]))
        writer.write_table(table)


def test_writer_enforces_num_rows(fmt: FileFormat) -> None:
    Writer = WRITER_CLASSES[fmt]
    table, metadata = _survey()

    writer = Writer(io.BytesIO(), table.schema, 3, metadata)
    with pytest.raises(readstat_arrow.ReadstatError, match="3 rows"):
        writer.write_table(table)

    writer = Writer(io.BytesIO(), table.schema, 10, metadata)
    writer.write_table(table)
    with pytest.raises(readstat_arrow.ReadstatError):
        writer.close()


def test_writer_rejects_a_negative_num_rows(fmt: FileFormat) -> None:
    Writer = WRITER_CLASSES[fmt]
    table, _metadata = _survey()
    with pytest.raises(ValueError, match="num_rows must be non-negative"):
        Writer(io.BytesIO(), table.schema, -1)


def test_close_is_idempotent(fmt: FileFormat) -> None:
    """Closing twice is not an error; the second call has nothing left to finish."""
    read = READER_FUNCS[fmt]
    Writer = WRITER_CLASSES[fmt]
    table, metadata = _survey()
    out = io.BytesIO()

    writer = Writer(out, table.schema, table.num_rows, metadata)
    writer.write_table(table)
    writer.close()
    writer.close()

    out.seek(0)
    back, _back_metadata = read(out)
    assert back.equals(table)


def test_a_failed_write_leaves_no_finished_file(
    fmt: FileFormat, tmp_path: Path
) -> None:
    """Leaving the block with an exception closes the file without ending the format properly."""
    read = READER_FUNCS[fmt]
    Writer = WRITER_CLASSES[fmt]
    table, metadata = _survey()
    out = tmp_path / f"partial.{fmt}"

    with (
        pytest.raises(RuntimeError, match="boom"),
        Writer(out, table.schema, table.num_rows, metadata) as writer,
    ):
        writer.write_table(table.slice(0, 2))
        raise RuntimeError("boom")

    assert out.exists()  # what was written stays on disk ...
    # ... but it does not read as a whole file
    with pytest.raises(readstat_arrow.ReadstatError):
        read(out)


def test_columns_without_metadata_are_written_undeclared(fmt: FileFormat) -> None:
    """A column with no metadata is written anyway; metadata for absent columns is ignored."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table({"num": pa.array([1.0, 2.0]), "text": pa.array(["x", "yy"])})
    metadata = Metadata(
        variable_labels={"num": "Numbers", "gone": "Not here"},
        file_label="Partly described",
    )
    out = io.BytesIO()

    write(out, table, metadata)
    out.seek(0)
    back, back_metadata = read(out)

    assert back.to_pydict() == {"num": [1.0, 2.0], "text": ["x", "yy"]}
    assert back.column_names == ["num", "text"]
    # "gone" ignored, "text" declared nothing:
    assert back_metadata.variable_labels == {"num": "Numbers"}
    assert back_metadata.value_labels == {}
    assert back_metadata.file_label == "Partly described"


def test_none_declares_nothing(fmt: FileFormat) -> None:
    """A name mapped to None is exactly as undeclared as a name that is absent."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table({"n": pa.array([1.0]), "s": pa.array(["ab"], pa.large_string())})
    metadata = Metadata(
        variable_labels={"n": None, "s": "Text"},
        value_labels={"n": None},
        formats={"n": None},
        storage_widths={"s": None},
        measures={"n": None},
        display_widths={"n": None},
        missing_values={"n": None},
    )
    nones = io.BytesIO()
    write(nones, table, metadata)
    nones.seek(0)
    bare, bare_metadata = read(nones)

    empty = io.BytesIO()
    write(empty, table, Metadata(variable_labels={"s": "Text"}))
    empty.seek(0)
    same, same_metadata = read(empty)

    assert bare.equals(same)
    assert bare_metadata == same_metadata
    assert bare_metadata.variable_labels == {"s": "Text"}
    assert bare_metadata.value_labels == {}
    # sized from the data, as with no entry at all:
    assert bare_metadata.storage_widths["s"] == 2


UNDECLARED: dict[FileFormat, dict[str, list[t.Any]]] = {
    "sav": {"n": [1.0, 2.0], "s": ["a", "b"]},  # SPSS stores every number as a double
    "dta": {"n": [1, 2], "s": ["a", "b"]},  # Stata keeps the int8 it was handed
}


def test_write_without_metadata(fmt: FileFormat) -> None:
    """No metadata at all: the columns keep their Arrow names and nothing else is declared."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table({"n": pa.array([1, 2], pa.int8()), "s": pa.array(["a", "b"])})
    omitted, spelled_out = io.BytesIO(), io.BytesIO()

    write(omitted, table)
    write(spelled_out, table, Metadata())  # the same thing, spelled out
    assert omitted.getvalue() == spelled_out.getvalue()

    omitted.seek(0)
    back, back_metadata = read(omitted)
    assert back.to_pydict() == UNDECLARED[fmt]
    assert back.column_names == ["n", "s"]
    assert back_metadata.variable_labels == {}
    assert back_metadata.file_label is None


def test_string_widths_survive_repeated_round_trips(fmt: FileFormat) -> None:
    """Metadata from a file must write the same widths back, or they creep up every cycle.

    ReadStat reports a string's storage as the format lays it out - SPSS in whole
    8-byte cells, Stata with a byte for a possible NUL - so the reader normalises
    it back to the declared width before it can be fed to a writer again.
    """
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    Writer = WRITER_CLASSES[fmt]
    table = pa.table(
        {"s3": pa.array(["abc"]), "s20": pa.array(["x" * 20])}
    )  # "long" is Stata-reserved
    out = io.BytesIO()
    write(out, table)  # sizes the columns from the data
    widths = []
    for _ in range(3):  # then keep rewriting with nothing but what was read back
        out.seek(0)
        data, metadata = read(out)
        widths.append([metadata.storage_widths[n] for n in data.column_names])
        out = io.BytesIO()
        with Writer(out, data.schema, data.num_rows, metadata) as writer:
            writer.write_table(data)

    assert widths == [[3, 20]] * 3, f"widths drifted: {widths}"


def test_incremental_writer_without_metadata(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    Writer = WRITER_CLASSES[fmt]
    table = pa.table({"n": pa.array([1.0, 2.0])})
    out = io.BytesIO()

    with Writer(out, table.schema, table.num_rows) as writer:
        writer.write_table(table)

    out.seek(0)
    back, _metadata = read(out)
    assert back.to_pydict() == {"n": [1.0, 2.0]}
    assert back.column_names == ["n"]


def test_unsupported_arrow_type(fmt: FileFormat) -> None:
    write = WRITER_FUNCS[fmt]
    table = pa.table({"nested": pa.array([[1, 2], [3]])})
    with pytest.raises(TypeError, match="nested"):
        write(io.BytesIO(), table)


def test_all_null_columns(fmt: FileFormat) -> None:
    """Entirely empty columns (common in survey data) take a fast path in the writer."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table(
        {
            "num": pa.array([None, None, None], pa.float64()),
            "typed_null": pa.array([None, None, None], pa.null()),
            "filled": pa.array([1.0, 2.0, 3.0]),
        }
    )
    out = io.BytesIO()

    write(out, table)
    out.seek(0)
    back, _ = read(out)

    assert back.column("num").null_count == 3
    assert back.column("typed_null").null_count == 3
    assert back.column("filled").to_pylist() == [1.0, 2.0, 3.0]


def test_an_all_null_string_column(fmt: FileFormat) -> None:
    """Neither format has a null string of its own."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table({"txt": pa.array([None, None, None], pa.large_string())})
    out = io.BytesIO()

    write(out, table)
    out.seek(0)
    back, _ = read(out)

    assert back.column("txt").to_pylist() == ["", "", ""]


TAG_TYPE = pa.dictionary(pa.int8(), pa.string())
TAG_LETTERS = pa.array([chr(c) for c in range(ord("a"), ord("z") + 1)])


def _tagged(values: list[int | None], tags: list[str | None]) -> pa.StructArray:
    """struct<value: int8, tag> exactly as read_dta(preserve_user_missing=True) builds it.

    The tag dictionary is always the full a-z alphabet (Array.equals compares
    dictionaries, not just decoded values); a null struct where both are None.
    """
    value_array = pa.array(values, pa.int8())
    indices = pa.array(
        [None if t is None else ord(t) - ord("a") for t in tags], pa.int8()
    )
    tag_array = pa.DictionaryArray.from_arrays(indices, TAG_LETTERS)
    return pa.StructArray.from_arrays(
        [value_array, tag_array],
        names=["value", "tag"],
        mask=pc.and_(value_array.is_null(), tag_array.is_null()),
    )


def test_an_empty_code_list_writes_no_label_set(fmt: FileFormat) -> None:
    """A label set with no codes in it corrupts a .sav, so an empty list must write nothing."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table({"q": pa.array([1, 2], pa.int32())})
    empty, nothing = io.BytesIO(), io.BytesIO()

    write(empty, table, Metadata(value_labels={"q": []}))
    write(nothing, table, Metadata())  # no entry for "q" at all

    # Byte for byte the same file: no set was created, and none was referenced.
    assert empty.getvalue() == nothing.getvalue()
    empty.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        _back, back_metadata = read(empty)
    assert back_metadata.value_labels == {}


def test_rename_invalid_names_says_nothing_when_nothing_is_renamed(
    fmt: FileFormat,
) -> None:
    read = READER_FUNCS[fmt]
    Writer = WRITER_CLASSES[fmt]
    table = pa.table({"ok": pa.array([1.0]), "also_ok": pa.array([2.0])})
    out = io.BytesIO()

    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        writer = Writer(out, table.schema, 1, rename_invalid_names=True)
        writer.write_table(table)
        writer.close()

    assert writer.renamed_variables == {}
    out.seek(0)
    back, _metadata = read(out)
    assert back.column_names == ["ok", "also_ok"]


@pytest.mark.parametrize(
    ("name", "bad_name"),
    [("sav", "with space"), ("dta", "1leading_digit"), ("sav", "x" * 70)],
)
def test_invalid_variable_names_are_reported_as_header_errors(
    name: FileFormat, bad_name: str
) -> None:
    """ReadStat validates names when it emits the header, which happens at the first row."""
    table = pa.table({bad_name: [1.0]})
    write = readstat_arrow.write_sav if name == "sav" else readstat_arrow.write_dta

    with pytest.raises(readstat_arrow.ReadstatError, match="writing header") as info:
        write(io.BytesIO(), table)
    assert "row 0" not in str(info.value)


def test_temporal_type_variants(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table(
        {
            "ts_tz": pa.array(
                [1_600_000_000_000_000_000, None], pa.timestamp("ns", tz="Europe/Oslo")
            ),
            "d64": pa.array([0, 86_400_000], pa.date64()),
            "dur": pa.array(
                [90_000_000_000, None], pa.duration("us")
            ),  # 25 hours: beyond time64's day
        }
    )
    out = io.BytesIO()

    write(out, table)
    out.seek(0)
    back, _ = read(out)

    # Stata has nothing that denotes elapsed time, so a duration goes out as a plain number of
    # milliseconds -- the unit %tc counts in -- rather than as some instant in 1960.
    # SPSS has DTIME, which round-trips.
    dur_type: pa.DataType
    dur_values: list[float | timedelta | None]
    if fmt == "dta":
        dur_type, dur_values = pa.float64(), [
            90_000_000.0,
            None,
        ]  # 25 hours in milliseconds
    else:
        dur_type, dur_values = pa.duration("us"), [timedelta(days=1, hours=1), None]

    assert back.schema.types == [pa.timestamp("us"), pa.date32(), dur_type]
    assert back.to_pydict() == {
        "ts_tz": [
            datetime(2020, 9, 13, 12, 26, 40),
            None,
        ],  # the UTC instant; neither has a timezone
        "d64": [date(1970, 1, 1), date(1970, 1, 2)],
        "dur": dur_values,
    }


def test_zero_rows(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table(
        {"a": pa.array([], pa.float64()), "s": pa.array([], pa.large_string())}
    )
    out = io.BytesIO()

    write(out, table)
    out.seek(0)
    back, _ = read(out)

    assert back.num_rows == 0
    assert back.column_names == ["a", "s"]


def test_long_labels_are_truncated_on_character_boundaries(fmt: FileFormat) -> None:
    """Limits are in UTF-8 bytes; the cut must never leave half a multibyte character."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    long_variable_label = "ø" * 300  # 600 bytes
    long_value_label = "€" * 100  # 300 bytes
    long_file_label = "日本語" * 40  # 360 bytes
    codes: list[Code] = [{"value": 1, "label": long_value_label}]
    table = pa.table({"v": pa.array([1], pa.int8())})
    metadata = Metadata(
        variable_labels={"v": long_variable_label},
        value_labels={"v": codes},
        file_label=long_file_label,
    )
    out = io.BytesIO()

    with pytest.warns(readstat_arrow.ReadstatWarning, match="truncated"):
        write(out, table, metadata, on_text_limits_exceeded="truncate")
    out.seek(0)
    _, back = read(out)

    var_limit, value_limit, file_limit = LABEL_LIMITS[fmt]
    _assert_fitted(back.variable_labels["v"], long_variable_label, var_limit)
    back_codes = back.value_labels.get("v")
    assert back_codes is not None  # the label survived, however much of it fitted
    _assert_fitted(back_codes[0]["label"], long_value_label, value_limit)
    _assert_fitted(back.file_label, long_file_label, file_limit)


def _assert_fitted(actual: str | None, original: str, limit: LabelLimit) -> None:
    """``actual`` is the longest whole-character prefix of ``original`` that fits within ``limit``."""
    max_bytes, max_chars = limit
    assert actual is not None
    assert original.startswith(actual)
    assert len(actual.encode()) <= max_bytes
    if max_chars is not None:
        assert len(actual) <= max_chars
    if actual != original:  # nothing more would have fit
        next_char = original[len(actual)]
        over_bytes = len(actual.encode()) + len(next_char.encode()) > max_bytes
        over_chars = max_chars is not None and len(actual) + 1 > max_chars
        assert over_bytes or over_chars


UNTOUCHED_CODES: dict[FileFormat, list[Code]] = {
    "sav": [{"value": 1.0, "label": "€" * 40}],
    "dta": [{"value": 1, "label": "€" * 40}],
}


def test_labels_within_limits_are_untouched(fmt: FileFormat) -> None:
    """Labels sized to whichever format is stricter per field, so neither format cuts them."""
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    # exactly 120 bytes, SPSS's limit
    codes: list[Code] = [{"value": 1, "label": "€" * 40}]
    metadata = Metadata(
        variable_labels={"v": "ø" * 80},  # exactly 80 characters, Stata 118's limit
        value_labels={"v": codes},
        file_label="x" * 64,  # exactly 64 bytes, SPSS's limit
    )
    table = pa.table({"v": pa.array([1.0])})
    out = io.BytesIO()

    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        write(out, table, metadata)
    out.seek(0)
    _, back = read(out)

    assert back.variable_labels["v"] == "ø" * 80
    assert back.value_labels["v"] == UNTOUCHED_CODES[fmt]
    assert back.file_label == "x" * 64


def test_overlong_text_is_an_error_by_default(fmt: FileFormat) -> None:
    """Nothing is cut behind the caller's back: the default policy refuses to write."""
    write = WRITER_FUNCS[fmt]
    table = pa.table({"v": pa.array([1.0])})
    metadata = Metadata(variable_labels={"v": "x" * 400})

    with pytest.raises(ValueError, match="variable label of v"):
        write(io.BytesIO(), table, metadata)


def test_the_error_names_the_limit_and_the_way_out(fmt: FileFormat) -> None:
    """The message has to be enough to act on without reading the docs."""
    write = WRITER_FUNCS[fmt]
    table = pa.table({"v": pa.array([1.0])})
    metadata = Metadata(file_label="x" * 400)

    with pytest.raises(
        ValueError, match=r"file label of file is 400 bytes and 400 characters"
    ):
        write(io.BytesIO(), table, metadata)
    with pytest.raises(ValueError, match=r"Pass on_text_limits_exceeded='truncate'"):
        write(io.BytesIO(), table, metadata)


def test_writers_reject_an_unknown_on_text_limits_exceeded(fmt: FileFormat) -> None:
    Writer = WRITER_CLASSES[fmt]
    schema = pa.schema([pa.field("v", pa.float64())])

    with pytest.raises(
        ValueError, match="on_text_limits_exceeded must be 'error' or 'truncate'"
    ):
        Writer(io.BytesIO(), schema, 1, on_text_limits_exceeded="cut")


def test_sav_notes_are_one_80_byte_line() -> None:
    """An SPSS note is a single document-record line, which is 80 bytes wide."""
    table = pa.table({"v": pa.array([1.0])})
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, Metadata(notes=["n" * 80]))
    out.seek(0)
    _, back = readstat_arrow.read_sav(out)
    assert back.notes == ["n" * 80]

    with pytest.raises(ValueError, match="note of note 0 is 81 bytes"):
        readstat_arrow.write_sav(io.BytesIO(), table, Metadata(notes=["n" * 81]))


def test_dta_notes_run_to_67784_bytes() -> None:
    table = pa.table({"v": pa.array([1.0])})
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table, Metadata(notes=["n" * 67_784]))
    out.seek(0)
    _, back = readstat_arrow.read_dta(out)
    assert back.notes == ["n" * 67_784]

    with pytest.raises(ValueError, match="note of note 0 is 67785 bytes"):
        readstat_arrow.write_dta(io.BytesIO(), table, Metadata(notes=["n" * 67_785]))


def test_truncating_a_note_keeps_its_first_bytes(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    table = pa.table({"v": pa.array([1.0])})
    out = io.BytesIO()

    with pytest.warns(readstat_arrow.ReadstatWarning, match="note of note 0"):
        write(
            out,
            table,
            Metadata(notes=["n" * 80_000]),
            on_text_limits_exceeded="truncate",
        )
    out.seek(0)
    _, back = read(out)

    assert back.notes == [NOTE_KEPT[fmt]]


NOTE_KEPT: dict[FileFormat, str] = {"sav": "n" * 80, "dta": "n" * 67_784}


def test_truncating_string_values_cuts_on_a_character_boundary() -> None:
    """2045 is not a multiple of 3, so a naive cut would leave a third of a 日."""
    out = io.BytesIO()

    with pytest.warns(
        readstat_arrow.ReadstatWarning
    ):  # the width and the values are both warned about
        readstat_arrow.write_dta(
            out,
            pa.table({"s": pa.array(["日" * 1000])}),
            on_text_limits_exceeded="truncate",
        )
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out)

    assert (
        back.column("s")[0].as_py() == "日" * 681
    )  # 2043 bytes; a 682nd would need 2046


def test_truncating_only_touches_the_values_that_do_not_fit() -> None:
    out = io.BytesIO()

    with pytest.warns(readstat_arrow.ReadstatWarning):
        readstat_arrow.write_dta(
            out,
            pa.table({"s": pa.array(["short", None, "x" * 3000])}),
            on_text_limits_exceeded="truncate",
        )
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out)

    assert back.column("s").to_pylist() == [
        "short",
        "",
        "x" * 2045,
    ]  # Stata has no missing string


def test_truncating_leaves_values_that_fit_alone() -> None:
    out = io.BytesIO()

    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        readstat_arrow.write_dta(
            out,
            pa.table({"s": pa.array(["short", "日本"])}),
            on_text_limits_exceeded="truncate",
        )
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out)

    assert back.column("s").to_pylist() == ["short", "日本"]


def test_truncating_cuts_values_to_a_declared_width() -> None:
    """The incremental writer is told the width, so the values are brought down to it."""
    table = pa.table({"s": pa.array(["abcdefgh"], pa.large_string())})
    narrow = Metadata(storage_widths={"s": 5})
    out = io.BytesIO()

    with (
        pytest.warns(
            readstat_arrow.ReadstatWarning, match="values longer than 5 bytes truncated"
        ),
        DtaWriter(
            out, table.schema, 1, narrow, on_text_limits_exceeded="truncate"
        ) as writer,
    ):
        writer.write_table(table)
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out)

    assert back.column("s")[0].as_py() == "abcde"


# Tests of one format alone: a file, a record or a rule the other format has no equivalent of.
# Nothing below takes ``fmt``; each says in its name which format it is about.


def test_write_sav_roundtrip() -> None:
    table, metadata = _survey()
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_sav(out)

    assert back.equals(table)
    assert back_metadata.file_label == "Tiny survey"
    assert back_metadata.notes == metadata.notes
    assert back.column_names == table.column_names
    assert back_metadata.variable_labels == metadata.variable_labels
    assert back_metadata.formats == metadata.formats
    assert back_metadata.measures == metadata.measures
    assert back_metadata.value_labels == metadata.value_labels


def test_write_dta_roundtrip() -> None:
    table, metadata = _panel()
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_dta(out)

    assert back.equals(table)  # int8/int16/int32/float32 stay put, dates/times survive
    assert back_metadata.file_label == "Panel"
    assert back_metadata.variable_labels == metadata.variable_labels
    assert back_metadata.value_labels == {
        "sex": [{"value": 1, "label": "M"}, {"value": 2, "label": "F"}]
    }
    assert back_metadata.formats["day"] == "%td"
    assert back_metadata.formats["when"] == "%tc"


def test_sav_metadata_into_dta_drops_spss_formats() -> None:
    table, metadata = (
        _survey()
    )  # metadata full of SPSS formats: F8.2, DATE11, TIME8, ...
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_dta(out)

    assert back.equals(table)
    mynum_format = back_metadata.formats.get("mynum")
    assert mynum_format is not None  # the reader records a format for every variable
    assert mynum_format.startswith(
        "%"
    )  # ... and it is a Stata one, not the SPSS "F8.2"
    assert back_metadata.formats["mydate"] == "%td"
    assert back_metadata.formats["mytime"] == "%tcHH:MM:SS"


def test_sav_preserve_user_missing_values_roundtrip() -> None:
    """SPSS user-missing values stay in the data and keep their declarations."""
    table = pa.table(
        {
            "mynum": pa.array(
                [1.0, -1.0, 2500.0, None]
            ),  # -1 discrete, 2500 inside the range
            "myord": pa.array([1.0, -1.0, -2.0, -3.0]),
        }
    )
    metadata = Metadata(
        missing_values={
            "mynum": {"lo": 2000.0, "hi": 3000.0, "value": -1.0},
            "myord": {"values": [-1.0, -2.0, -3.0]},
        }
    )
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_sav(out, preserve_user_missing=True)
    out.seek(0)
    nulled, _ = readstat_arrow.read_sav(out)

    assert back.equals(table)
    assert nulled.to_pydict() == {
        "mynum": [1.0, None, None, None],
        "myord": [1.0, None, None, None],
    }
    assert back_metadata.missing_values == {
        "mynum": {"lo": 2000.0, "hi": 3000.0, "value": -1.0},
        "myord": {"values": [-1.0, -2.0, -3.0]},
    }


def test_sav_write_from_scratch() -> None:
    """Building metadata by hand, without reading a file first."""
    metadata = Metadata(
        variable_labels={"id": "Respondent id", "agree": "Agrees with statement"},
        value_labels={
            "agree": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]
        },
        measures={"id": "nominal", "agree": "nominal"},
        file_label="Tiny survey",
    )
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "agree": pa.array([1, None, 0], pa.int8()),
        }
    )
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_sav(out)

    assert back.to_pydict() == {
        "id": [1.0, 2.0, 3.0],
        "agree": [1.0, None, 0.0],
    }  # SPSS is all doubles
    assert back_metadata.file_label == "Tiny survey"
    assert back_metadata.variable_labels["agree"] == "Agrees with statement"
    assert back_metadata.value_labels["agree"] == [
        {"value": 0.0, "label": "No"},
        {"value": 1.0, "label": "Yes"},
    ]  # SPSS: doubles


def test_sav_empty_missing_values_list_declares_nothing() -> None:
    """``{"values": []}`` is as empty as ``None``; only more than three is an error."""
    table = pa.table({"q": pa.array([1.0])})

    declares_nothing: list[Missingness | None] = [{"values": []}, None]
    for missing in declares_nothing:
        out = io.BytesIO()
        readstat_arrow.write_sav(out, table, Metadata(missing_values={"q": missing}))
        out.seek(0)
        _back, back_metadata = readstat_arrow.read_sav(out)
        assert back_metadata.missing_values == {}


def test_sav_too_many_discrete_missing_raises() -> None:
    """More than three discrete missing values for a column raises an error."""
    table = pa.table({"q": pa.array([1.0])})
    too_many = Metadata(missing_values={"q": {"values": [1.0, 2.0, 3.0, 4.0]}})
    out = io.BytesIO()

    with pytest.raises(
        ValueError, match=r"'q'.*at most three discrete missing values, got 4"
    ):
        readstat_arrow.write_sav(out, table, too_many)
    assert out.getvalue() == b""  # planning happens before anything is written


def test_sav_missing_values_must_be_one_of_the_two_shapes() -> None:
    """``Missingness`` is a TypedDict, so its shape is only checked when it is used."""
    table = pa.table({"q": pa.array([1.0])})
    malformed = Metadata(
        missing_values={"q": t.cast(Missingness, {"low": 1.0, "high": 2.0})}
    )

    with pytest.raises(ValueError, match=r"'q'.*must be \{'values': \[\.\.\.\]\}"):
        readstat_arrow.write_sav(io.BytesIO(), table, malformed)


def test_dta_cannot_store_user_defined_missing_values() -> None:
    """Stata has tagged missings instead; an SPSS declaration has nowhere to go."""
    table = pa.table({"q": pa.array([1.0])})
    declared: list[Missingness] = [{"values": [9.0]}, {"lo": 8.0, "hi": 9.0}]

    for missing in declared:
        metadata = Metadata(missing_values={"q": missing})
        with pytest.raises(
            ValueError, match=r"'q'.*cannot store SPSS user-defined missing values"
        ):
            readstat_arrow.write_dta(io.BytesIO(), table, metadata)


def test_sav_missing_values_of_a_string_variable_must_be_strings() -> None:
    table = pa.table({"s": pa.array(["a"], pa.large_string())})
    metadata = Metadata(missing_values={"s": {"values": ["Z", 9.0]}})

    with pytest.raises(ValueError, match=r"'s'.*must be strings"):
        readstat_arrow.write_sav(io.BytesIO(), table, metadata)


def test_write_dta_widens_integers_in_reserved_ranges() -> None:
    """Stata keeps the top of each integer range for missing values; write_dta widens past them.

    Widening only: a column whose values fit keeps the width it was handed.
    """
    table = pa.table(
        {
            "fits": pa.array(
                [-127, 100], pa.int8()
            ),  # largest legal byte values: stays int8
            "small": pa.array([1, 2], pa.int32()),  # room to spare, but never narrowed
            "b": pa.array([1, 101], pa.int8()),  # 101 is Stata's '.': becomes int16
            "i": pa.array([1, 32_741], pa.int16()),  # -> int32
            "l": pa.array(
                [1, 2_147_483_621], pa.int32()
            ),  # -> double (Stata has no int64)
            "big": pa.array([1, 2**40], pa.int64()),  # -> double
        }
    )
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table)
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out)

    assert back.schema.types == [
        pa.int8(),
        pa.int32(),
        pa.int16(),
        pa.int32(),
        pa.float64(),
        pa.float64(),
    ]
    assert back.to_pydict() == {
        k: [float(x) if k in ("l", "big") else x for x in v]
        for k, v in table.to_pydict().items()
    }


def test_dta_writer_refuses_values_in_reserved_ranges() -> None:
    """The incremental writer cannot see the data, so an undeclared 101 is an error."""
    table = pa.table({"b": pa.array([1, 101], pa.int8())})
    writer = readstat_arrow.DtaWriter(io.BytesIO(), table.schema, table.num_rows)
    with pytest.raises(readstat_arrow.ReadstatError, match="'b'"):
        writer.write_table(table)


def test_dta_writer_widens_from_declared_ranges() -> None:
    """Told the min and max values a column will hold, the writer picks the type write_dta would."""
    table = pa.table(
        {
            "b": pa.array([1, 101], pa.int8()),  # 101 is Stata's '.': needs int16
            "l": pa.array([1, 2], pa.int32()),  # small now, but declared past a long
            "tagged": _tagged([1, 101], [None, None]),
            "f": pa.array([1.5, 2.5]),  # not an integer: the entry is ignored
        }
    )
    ranges = {"b": (1, 500), "l": (1, 5_000_000_000), "tagged": (1, 200), "f": (9, 9)}
    out = io.BytesIO()

    with readstat_arrow.DtaWriter(
        out, table.schema, table.num_rows, variable_ranges=ranges
    ) as writer:
        for batch in table.to_batches(max_chunksize=1):  # cast per batch, not per table
            writer.write_batch(batch)

    out.seek(0)
    back, _metadata = readstat_arrow.read_dta(out)
    assert back.schema.types == [pa.int16(), pa.float64(), pa.int16(), pa.float64()]
    assert back.column("b").to_pylist() == [1, 101]
    assert back.column("tagged").to_pylist() == [1, 101]


def test_dta_tagged_missing_roundtrip() -> None:
    """{value: null, tag: 'a'} becomes .a, a null struct becomes '.', values pass through."""
    table = pa.table({"v": _tagged([1, None, None, None], [None, "a", None, "z"])})
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table)
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out, preserve_user_missing=True)
    out.seek(0)
    plain, _ = readstat_arrow.read_dta(out)

    assert back.equals(table)
    assert plain.column("v").to_pylist() == [
        1,
        None,
        None,
        None,
    ]  # every kind of missing is null


def test_dta_labelled_tags() -> None:
    """A label set can label the tagged missings .a-.z alongside ordinary values."""
    codes: list[Code] = [
        {"value": 1, "label": "Yes"},
        {"value": 2, "label": "No"},
        {"value": "a", "label": "Refused"},
    ]
    table = pa.table({"q": _tagged([1, 2, None], [None, None, "a"])})
    metadata = Metadata(value_labels={"q": codes})
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_dta(out, preserve_user_missing=True)

    assert back.equals(table)
    assert back_metadata.value_labels["q"] == codes


def test_sav_string_value_labels_roundtrip() -> None:
    """SPSS labels the values of a string variable too, so a code list may be all strings."""
    table = pa.table({"s": pa.array(["a", "b"], pa.large_string())})
    codes: list[Code] = [
        {"value": "a", "label": "Apple"},
        {"value": "b", "label": "Banana"},
    ]
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, Metadata(value_labels={"s": codes}))
    out.seek(0)
    back, back_metadata = readstat_arrow.read_sav(out)

    assert back.equals(table)
    assert back_metadata.value_labels == {"s": codes}


def test_sav_code_list_cannot_mix_strings_and_numbers() -> None:
    """One label set is one type: SPSS labels either string values or numeric ones."""
    table = pa.table({"s": pa.array(["a"], pa.large_string())})
    mixed = Metadata(
        value_labels={
            "s": [{"value": "a", "label": "Apple"}, {"value": 1, "label": "One"}]
        }
    )

    with pytest.raises(ValueError, match=r"'s'.*mixes string and numeric values"):
        readstat_arrow.write_sav(io.BytesIO(), table, mixed)


def test_dta_labels_only_integers_and_tags() -> None:
    """Stata keys a label set with an int32, or with a tag letter for a missing value."""
    table = pa.table({"v": pa.array([1.0])})

    for value, message in (
        ("xy", "cannot label string value"),
        (1.5, "only label integer values"),
    ):
        bad = Metadata(value_labels={"v": [{"value": value, "label": "nope"}]})
        with pytest.raises(ValueError, match=rf"'v'.*{message}"):
            readstat_arrow.write_dta(io.BytesIO(), table, bad)


def test_dta_value_label_keys_stop_where_a_long_does() -> None:
    """Label keys are int32 whatever the variable's type, and the top of it means missing.

    2_147_483_621 upwards is how a ``.dta`` encodes ``.`` and ``.a``-``.z``, so a
    key there would come back as a tag label rather than the number written.
    """
    table = pa.table(
        {"v": pa.array([2_147_483_620, 4_294_967_240], pa.int64())}
    )  # written as double
    out = io.BytesIO()

    fits = Metadata(
        value_labels={"v": [{"value": 2_147_483_620, "label": "the largest long"}]}
    )
    readstat_arrow.write_dta(out, table, fits)
    out.seek(0)
    _back, back_metadata = readstat_arrow.read_dta(out)
    assert back_metadata.value_labels == {
        "v": [{"value": 2_147_483_620, "label": "the largest long"}]
    }

    for key in (
        4_294_967_240,
        2_147_483_622,
        -2_147_483_648,
    ):  # past int32, .a, past long
        bad = Metadata(value_labels={"v": [{"value": key, "label": "nope"}]})
        with pytest.raises(ValueError, match=r"'v'.*only label values a long can hold"):
            readstat_arrow.write_dta(io.BytesIO(), table, bad)


def test_dta_rejects_a_value_labelled_twice() -> None:
    """Stata finds a label by binary search, so one key may carry only one label."""
    table = pa.table({"v": pa.array([1.0])})
    twice = Metadata(
        value_labels={"v": [{"value": 1, "label": "One"}, {"value": 1, "label": "Uno"}]}
    )

    with pytest.raises(ValueError, match=r"'v'.*cannot label value 1 more than once"):
        readstat_arrow.write_dta(io.BytesIO(), table, twice)


def test_dta_rejects_keys_that_collide_once_narrowed() -> None:
    """1 and 1.0 are distinct codes in the metadata but the same int32 key in the file."""
    table = pa.table({"v": pa.array([1.0])})
    colliding = Metadata(
        value_labels={
            "v": [{"value": 1, "label": "One"}, {"value": 1.0, "label": "Uno"}]
        }
    )

    with pytest.raises(ValueError, match=r"'v'.*cannot label value 1 more than once"):
        readstat_arrow.write_dta(io.BytesIO(), table, colliding)


def test_dta_rejects_a_tag_used_twice() -> None:
    table = pa.table({"v": pa.array([1.0])})
    twice = Metadata(
        value_labels={
            "v": [{"value": "a", "label": "Refused"}, {"value": "a", "label": "Again"}]
        }
    )

    with pytest.raises(ValueError, match=r"'v'.*cannot label value 'a' more than once"):
        readstat_arrow.write_dta(io.BytesIO(), table, twice)


def test_dta_string_columns_stop_at_2045_bytes() -> None:
    """Stata's str# types end there; longer text would need strL, which is not written."""
    readstat_arrow.write_dta(io.BytesIO(), pa.table({"s": pa.array(["x" * 2045])}))

    with pytest.raises(
        ValueError, match=r"'s' needs 3000 bytes; Stata strings stop at 2045"
    ):
        readstat_arrow.write_dta(io.BytesIO(), pa.table({"s": pa.array(["x" * 3000])}))


def test_dta_rejects_a_declared_width_past_2045() -> None:
    """The incremental writer honours the declared width, so it is checked there too."""
    table = pa.table({"s": pa.array(["a"], pa.large_string())})
    wide = Metadata(storage_widths={"s": 2046})

    with pytest.raises(
        ValueError, match=r"'s' needs 2046 bytes; Stata strings stop at 2045"
    ):
        DtaWriter(io.BytesIO(), table.schema, 1, wide)


def test_sav_string_columns_have_no_width_ceiling() -> None:
    """SPSS segments a long string across records, so 3000 bytes round-trips whole."""
    out = io.BytesIO()
    readstat_arrow.write_sav(out, pa.table({"s": pa.array(["x" * 3000])}))
    out.seek(0)
    back, _ = readstat_arrow.read_sav(out)

    assert back.column("s")[0].as_py() == "x" * 3000


def test_sav_rejects_tag_structs_and_dta_rejects_string_ones() -> None:
    table = pa.table({"v": _tagged([None], ["a"])})
    with pytest.raises(ValueError, match="tagged missing"):
        readstat_arrow.write_sav(io.BytesIO(), table)

    text = pa.StructArray.from_arrays(
        [pa.array(["x"]), pa.array([None], TAG_TYPE)], names=["value", "tag"]
    )
    table = pa.table({"s": text})
    with pytest.raises(ValueError, match="numeric"):
        readstat_arrow.write_dta(io.BytesIO(), table)


def test_dta_tag_struct_validation() -> None:
    both = pa.StructArray.from_arrays(
        [pa.array([1], pa.int8()), pa.array(["a"], TAG_TYPE)], names=["value", "tag"]
    )
    table = pa.table({"v": both})
    with pytest.raises(ValueError, match="both a value and a missing-value tag"):
        readstat_arrow.write_dta(io.BytesIO(), table)

    bad = pa.StructArray.from_arrays(
        [pa.array([None], pa.int8()), pa.array(["Q"], TAG_TYPE)], names=["value", "tag"]
    )
    table = pa.table({"v": bad})
    with pytest.raises(ValueError, match="single letters a-z"):
        readstat_arrow.write_dta(io.BytesIO(), table)


def test_dta_tag_structs_are_widened_like_plain_columns() -> None:
    table = pa.table(
        {"v": _tagged([1, 101, None], [None, None, "a"])}
    )  # 101 is not a legal Stata byte
    out = io.BytesIO()

    readstat_arrow.write_dta(out, table)
    out.seek(0)
    back, _ = readstat_arrow.read_dta(out, preserve_user_missing=True)

    assert back.schema.field("v").type.field("value").type == pa.int16()
    assert back.column("v").to_pylist() == [
        {"value": 1, "tag": None},
        {"value": 101, "tag": None},
        {"value": None, "tag": "a"},
    ]


def test_sav_integer_columns_default_to_no_decimals() -> None:
    """Every SPSS numeric is a double in the file, so ReadStat's own default shows a count as 1.00."""
    table = pa.table(
        {
            "count": pa.array([1, 2], pa.int32()),
            "flag": pa.array([True, False]),
            "ratio": pa.array([1.5, 2.5]),
            "declared": pa.array([1, 2], pa.int32()),
            "wide": pa.array([1, 2], pa.int32()),
        }
    )
    metadata = Metadata(formats={"declared": "F5.3"}, display_widths={"wide": 12})
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, metadata)
    out.seek(0)
    _schema, _num_rows, back_metadata = readstat_arrow.read_sav_metadata(out)

    assert back_metadata.formats == {
        "count": "F8.0",  # an integer column: no decimals
        "flag": "F8.0",
        "ratio": "F8.2",  # a float column keeps ReadStat's default
        "declared": "F5.3",  # a declared format wins
        "wide": "F12.0",  # the declared display width sets the field width
    }


def test_sav_formats_and_display_width_survive() -> None:
    metadata = Metadata(
        formats={"restricted": "N4", "integer": "F1.0", "text": "A3"},
        display_widths={"restricted": 12, "text": 20},
    )
    table = pa.table(
        {"restricted": [1023.0, 10.0], "integer": [1.0, 2.0], "text": ["ab", "c"]}
    )
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_sav(out)

    assert back.equals(
        table.set_column(2, "text", table.column("text").cast(pa.large_string()))
    )
    assert back_metadata.formats == {
        "restricted": "N4",
        "integer": "F1.0",
        "text": "A3",
    }
    assert back_metadata.display_widths["restricted"] == 12
    assert back_metadata.display_widths["text"] == 20


def test_sav_string_user_missing_roundtrip() -> None:
    """SPSS lets a string variable declare missing values too.

    Three of them, non-ASCII: ReadStat keeps the ``const char *`` it is handed
    rather than copying it, and only reads it back when it writes the header, so
    the encoded bytes have to outlive the call that declares them.
    """
    table = pa.table({"mychar": pa.array(["Z", "a", "æøå", "漢字"], pa.large_string())})
    metadata = Metadata(missing_values={"mychar": {"values": ["Z", "æøå", "漢字"]}})
    out = io.BytesIO()

    readstat_arrow.write_sav(out, table, metadata)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_sav(out, preserve_user_missing=True)
    out.seek(0)
    nulled, _ = readstat_arrow.read_sav(out)

    assert back.equals(table)
    assert back_metadata.missing_values == {"mychar": {"values": ["Z", "æøå", "漢字"]}}
    assert nulled.column("mychar").to_pylist() == [None, "a", None, None]


def test_dta_rename_invalid_names() -> None:
    """Stata's rules are the strict pair: letters, digits and _, 32 characters, reserved words."""
    names = [
        "ok",
        "my var",
        "1st",
        "int",
        "str8",
        "a.b",
        "a b",
        "a_b",
        "kjønn",
        "x" * 40,
    ]
    table = pa.table({name: pa.array([1.0]) for name in names})
    metadata = Metadata(
        variable_labels={"my var": "has a space"},
        value_labels={"1st": [{"value": 1, "label": "one"}]},
    )
    out = io.BytesIO()

    with pytest.warns(readstat_arrow.ReadstatWarning, match="renamed 7 variable"):
        readstat_arrow.write_dta(out, table, metadata, rename_invalid_names=True)
    out.seek(0)
    back, back_metadata = readstat_arrow.read_dta(out)

    assert back.column_names == [
        "ok",  # legal names are untouched, and keep their name even when another sanitises onto it
        "my_var",  # an illegal character becomes _
        "v1st",  # cannot start with a digit
        "int_",  # a reserved word
        "vstr8",  # str# is reserved as a prefix, so a trailing _ would not do
        "a_b_2",  # "a.b" sanitises onto the real "a_b" ...
        "a_b_3",  # ... and so does "a b"
        "a_b",
        "kjønn",  # non-ASCII is fine in a Unicode file
        "x" * 32,  # 32 characters is Stata's limit
    ]
    assert back.num_rows == 1
    # metadata follows its variable
    assert back_metadata.variable_labels == {"my_var": "has a space"}
    assert back_metadata.value_labels == {"v1st": [{"value": 1, "label": "one"}]}


def test_sav_and_dta_rename_invalid_names_is_off_by_default_and_reported() -> None:
    table = pa.table({"my var": pa.array([1.0])})

    with pytest.raises(readstat_arrow.ReadstatError, match="writing header"):
        readstat_arrow.write_dta(io.BytesIO(), table)

    # SPSS takes names Stata will not, so the two formats rename different things.
    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        readstat_arrow.write_sav(io.BytesIO(), pa.table({"a.b": pa.array([1.0])}))

    out = io.BytesIO()
    with pytest.warns(readstat_arrow.ReadstatWarning, match="renamed 1 variable"):
        writer = readstat_arrow.DtaWriter(
            out, table.schema, 1, rename_invalid_names=True
        )
    assert writer.renamed_variables == {"my var": "my_var"}
    writer.write_table(table)  # batches still carry the caller's own names
    writer.close()

    out.seek(0)
    back, _metadata = readstat_arrow.read_dta(out)
    assert back.column_names == ["my_var"]


def test_sav_and_dta_arrow_type_mapping() -> None:
    table = pa.table(
        {
            "b": pa.array([True, False, None]),
            "u8": pa.array([250, 0, None], pa.uint8()),
            "f32": pa.array([1.5, None, 2.5], pa.float32()),
            "i64": pa.array([2**40, 1, None], pa.int64()),
        }
    )
    dta_out, sav_out = io.BytesIO(), io.BytesIO()

    readstat_arrow.write_dta(dta_out, table)
    dta_out.seek(0)
    back, _ = readstat_arrow.read_dta(dta_out)
    # bool -> byte, uint8 -> int (byte only reaches 100), float32 stays float, int64 -> double
    assert back.schema.types == [pa.int8(), pa.int16(), pa.float32(), pa.float64()]
    assert back.to_pydict() == {
        "b": [1, 0, None],
        "u8": [250, 0, None],
        "f32": [1.5, None, 2.5],
        "i64": [2.0**40, 1.0, None],
    }

    readstat_arrow.write_sav(sav_out, table)
    sav_out.seek(0)
    back, _ = readstat_arrow.read_sav(sav_out)
    assert back.schema.types == [pa.float64()] * 4  # SPSS numbers are all doubles
