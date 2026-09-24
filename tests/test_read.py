"""End-to-end tests against the shared sample files (see tests/data/README.md)."""

from __future__ import annotations

import io
import os
import typing as t
import warnings
import zipfile
from datetime import date, datetime, time
from pathlib import Path

import pyarrow as pa
import pytest

import readstat_arrow
from conftest import DATA_DIR, METADATA_READER_FUNCS, READER_FUNCS, SAMPLES
from readstat_arrow import Code
from readstat_arrow._formats import FileFormat

# The columns every sample.* file holds (see tests/data/sample.csv). Formats differ
# in how they store the two labelled integer columns, so each test spells out the
# exact table its format is expected to produce.
MYCHAR = pa.array(["a", "b", "c", "d", "e"], pa.large_string())
MYNUM = pa.array([1.1, 1.2, -1000.3, -1.4, 1000.3], pa.float64())
MYDATE = pa.array(
    [date(2018, 5, 6), date(1880, 5, 6), date(1960, 1, 1), date(1583, 1, 1), None],
    pa.date32(),
)
DTIME = pa.array(
    [
        datetime(2018, 5, 6, 10, 10, 10),
        datetime(1880, 5, 6, 10, 10, 10),
        datetime(1960, 1, 1),
        datetime(1583, 1, 1),
        None,
    ],
    pa.timestamp("us"),
)
MYTIME = pa.array(
    [time(10, 10, 10), time(23, 10, 10), time(0, 0), time(16, 10, 10), None],
    pa.time64("us"),
)
MYLABL = [1, 2, 1, 2, 1]
MYORD = [1, 2, 3, 1, 1]

# SPSS stores labelled values as doubles, Stata as integers (int8 for these, a "byte").
LABELLED_TYPE: dict[FileFormat, pa.DataType] = {"sav": pa.float64(), "dta": pa.int8()}
MYLABL_LABELS: dict[FileFormat, list[Code]] = {
    "sav": [{"value": 1.0, "label": "Male"}, {"value": 2.0, "label": "Female"}],
    "dta": [{"value": 1, "label": "Male"}, {"value": 2, "label": "Female"}],
}
MYORD_LABELS: dict[FileFormat, list[Code]] = {
    "sav": [
        {"value": 1.0, "label": "low"},
        {"value": 2.0, "label": "medium"},
        {"value": 3.0, "label": "high"},
    ],
    "dta": [
        {"value": 1, "label": "low"},
        {"value": 2, "label": "medium"},
        {"value": 3, "label": "high"},
    ],
}


def test_read_sample(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    table, metadata = read(SAMPLES[fmt])

    expected = pa.table(
        {
            "mychar": MYCHAR,
            "mynum": MYNUM,
            "mydate": MYDATE,
            "dtime": DTIME,
            "mylabl": pa.array(MYLABL, LABELLED_TYPE[fmt]),
            "myord": pa.array(MYORD, LABELLED_TYPE[fmt]),
            "mytime": MYTIME,
        }
    )
    assert table.equals(expected)
    assert table.column_names == expected.column_names
    assert metadata.variable_labels["mychar"] == "character"
    assert metadata.value_labels["mylabl"] == MYLABL_LABELS[fmt]
    assert metadata.value_labels["myord"] == MYORD_LABELS[fmt]
    assert "mychar" not in metadata.value_labels

    # What only one of the two files has a way of saying.
    if fmt == "sav":
        assert metadata.measures["mychar"] == "nominal"
        assert metadata.storage_widths["mychar"] == 1  # A1: the declared width, not SPSS's 8-byte cell
        assert metadata.notes  # sample.sav carries a document record
        assert metadata.missing_values == {}
    elif fmt == "dta":
        assert metadata.formats["mytime"] == "%tcHH:MM:SS"
        assert metadata.missing_values == {}  # an SPSS-only concept, so empty here whatever the file
    else:
        t.assert_never(fmt)


def test_column_selection(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    table, metadata = read(SAMPLES[fmt], columns=["mynum", "mychar"])
    # File order wins over the requested order.
    assert table.equals(pa.table({"mychar": MYCHAR, "mynum": MYNUM}))
    assert table.column_names == ["mychar", "mynum"]  # file order wins in the metadata too
    assert list(metadata.formats) == ["mychar", "mynum"]


def test_row_limit_and_offset(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    table, _ = read(SAMPLES[fmt], row_limit=2, row_offset=1)
    assert table.column("mychar").to_pylist() == ["b", "c"]
    assert table.num_rows == 2


def test_read_metadata_only(fmt: FileFormat) -> None:
    read_metadata = METADATA_READER_FUNCS[fmt]
    schema, num_rows, _metadata = read_metadata(SAMPLES[fmt])
    assert num_rows == 5
    assert schema.names == [
        "mychar",
        "mynum",
        "mydate",
        "dtime",
        "mylabl",
        "myord",
        "mytime",
    ]


def test_metadata_schema_matches_a_full_read(fmt: FileFormat) -> None:
    read_metadata = METADATA_READER_FUNCS[fmt]
    read = READER_FUNCS[fmt]
    table, _ = read(SAMPLES[fmt])
    schema, _num_rows, _metadata = read_metadata(SAMPLES[fmt])

    assert schema == table.schema


def test_clean_file_emits_no_warnings(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        read(SAMPLES[fmt])


def test_missing_file(fmt: FileFormat, tmp_path: Path) -> None:
    read = READER_FUNCS[fmt]
    with pytest.raises(readstat_arrow.ReadstatError):
        read(tmp_path / f"nope.{fmt}")


def test_table_survives_ipc_roundtrip(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    table, _ = read(SAMPLES[fmt])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    back = pa.ipc.open_stream(sink.getvalue()).read_all()
    assert back.equals(table)
    assert back.schema.metadata is None


def test_read_metadata_carries_the_value_labels(fmt: FileFormat) -> None:
    read_metadata = METADATA_READER_FUNCS[fmt]
    _schema, num_rows, metadata = read_metadata(SAMPLES[fmt])
    assert num_rows == 5
    assert metadata.value_labels["mylabl"] == MYLABL_LABELS[fmt]


def test_read_from_file_object(fmt: FileFormat) -> None:
    """A file object gives exactly what the same file at a path gives."""
    read = READER_FUNCS[fmt]
    expected, expected_metadata = read(SAMPLES[fmt])
    with SAMPLES[fmt].open("rb") as file:
        table, metadata = read(file)
        assert not file.closed  # the caller's file object is left open

    assert table.equals(expected)
    assert metadata == expected_metadata


def test_read_from_an_os_encoded_path(fmt: FileFormat) -> None:
    """``os.PathLike`` is not the only path: bytes are handed to ReadStat as they are."""
    read = READER_FUNCS[fmt]
    expected, _ = read(SAMPLES[fmt])
    table, _ = read(os.fsencode(SAMPLES[fmt]))
    assert table.equals(expected)


def test_read_from_bytes_io(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    expected, _ = read(SAMPLES[fmt])
    table, _ = read(io.BytesIO(SAMPLES[fmt].read_bytes()))
    assert table.equals(expected)


def test_read_metadata_from_file_object(fmt: FileFormat) -> None:
    read_metadata = METADATA_READER_FUNCS[fmt]
    expected = read_metadata(SAMPLES[fmt])
    with SAMPLES[fmt].open("rb") as file:
        assert read_metadata(file) == expected


def test_read_from_file_object_honours_options(fmt: FileFormat) -> None:
    """The options are the parser's, not the path's: a file object gets all of them."""
    read = READER_FUNCS[fmt]
    table, _ = read(
        io.BytesIO(SAMPLES[fmt].read_bytes()),
        columns=["mynum", "mydate"],
        row_limit=2,
        row_offset=1,
    )
    assert table.column_names == ["mynum", "mydate"]
    assert table.column("mynum").to_pylist() == [1.2, -1000.3]


def test_read_from_unbuffered_file_object(fmt: FileFormat) -> None:
    """A raw file can return a short read; the io handler asks again rather than stopping."""
    read = READER_FUNCS[fmt]
    expected, _ = read(SAMPLES[fmt])
    with SAMPLES[fmt].open("rb", buffering=0) as file:
        table, _ = read(file)
    assert table.equals(expected)


def test_read_from_file_object_starts_where_it_is(fmt: FileFormat) -> None:
    """Reading begins at the file object's position, so a file embedded in a stream works."""
    read = READER_FUNCS[fmt]
    expected, _ = read(SAMPLES[fmt])
    stream = io.BytesIO(b"PREFIX!!" + SAMPLES[fmt].read_bytes())
    stream.seek(8)

    table, _ = read(stream)
    assert table.equals(expected)


def test_read_rejects_unseekable_file_object(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]

    class Unseekable(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

    with pytest.raises(ValueError, match="not seekable"):
        unseekable: t.Any = Unseekable()
        read(unseekable)


def test_read_reports_a_failure_of_the_file_object(fmt: FileFormat) -> None:
    """An exception from the file object reaches the caller, not ReadStat's account of it."""
    read = READER_FUNCS[fmt]

    class Failing(io.BytesIO):
        def readinto(self, buffer: object) -> int:
            raise OSError("no disk today")

    with pytest.raises(OSError, match="no disk today"):
        read(Failing(SAMPLES[fmt].read_bytes()))


def test_read_truncated_file_object(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    truncated = io.BytesIO(SAMPLES[fmt].read_bytes()[:120])
    with pytest.raises(readstat_arrow.ReadstatError):
        read(truncated)


def test_read_from_a_minimal_file_object(fmt: FileFormat) -> None:
    """Only read/seek/tell are required - no readinto, and no seekable to ask."""
    read = READER_FUNCS[fmt]

    class Minimal:
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)

        def read(self, size: int = -1) -> bytes:
            return self._buffer.read(size)

        def seek(self, offset: int, whence: int = 0) -> int:
            return self._buffer.seek(offset, whence)

        def tell(self) -> int:
            return self._buffer.tell()

    expected, _ = read(SAMPLES[fmt])
    minimal: t.Any = Minimal(SAMPLES[fmt].read_bytes())
    table, _ = read(minimal)
    assert table.equals(expected)


def test_read_from_zip_member(fmt: FileFormat, tmp_path: Path) -> None:
    """A file inside an archive, never extracted to disk."""
    read = READER_FUNCS[fmt]
    archive_path = tmp_path / "survey.zip"
    member_name = f"survey.{fmt}"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(SAMPLES[fmt], member_name)

    expected, _ = read(SAMPLES[fmt])
    with zipfile.ZipFile(archive_path) as archive, archive.open(member_name) as member:
        table, _ = read(member)
    assert table.equals(expected)


def test_sav_preserve_user_missing() -> None:
    default, metadata = readstat_arrow.read_sav(DATA_DIR / "sample_missing.sav")
    kept, _ = readstat_arrow.read_sav(DATA_DIR / "sample_missing.sav", preserve_user_missing=True)

    assert metadata.missing_values["mynum"] == {"lo": 2000.0, "hi": 3000.0, "value": -1.0}
    assert metadata.missing_values["myord"] == {"values": [-1.0, -2.0, -3.0]}
    assert "mychar" not in metadata.missing_values
    assert default.column("mynum").null_count > kept.column("mynum").null_count


def test_sav_recoverable_parse_problems_become_warnings() -> None:
    """ReadStat reports some problems and carries on reading; those become ReadstatWarnings.

    None of the sample files is broken in that way, so the file is built here: a
    .sav keeps names over 8 bytes in a separate record that maps the short name
    to the long one, and a short name that is not among the variables is reported
    and skipped.
    """
    out = io.BytesIO()
    readstat_arrow.write_sav(out, pa.table({"averylongvariablename": pa.array([1.0, 2.0])}))
    data = out.getvalue().replace(b"AVERYLON=averylong", b"AVERYLOX=averylong")

    with pytest.warns(readstat_arrow.ReadstatWarning, match="Failed to find AVERYLOX"):
        table, _ = readstat_arrow.read_sav(io.BytesIO(data))
    with pytest.warns(readstat_arrow.ReadstatWarning, match="Failed to find AVERYLOX"):
        readstat_arrow.read_sav_metadata(io.BytesIO(data))

    assert table.column_names == ["AVERYLON"]  # the long name was lost with the record


def test_sav_variable_without_a_display_format() -> None:
    """A .sav can leave a variable's format unset; then nothing is a date and widths stand as read.

    ReadStat's own writer always emits a format, so the file is made by blanking
    the print and write formats of the variable record - the 8 bytes that sit
    just before its 8-byte name.
    """
    out = io.BytesIO()
    readstat_arrow.write_sav(out, pa.table({"num": pa.array([1.0, 2.0])}))
    data = out.getvalue()
    name_at = data.index(b"NUM     ")
    blanked = data[: name_at - 8] + bytes(8) + data[name_at:]

    table, metadata = readstat_arrow.read_sav(io.BytesIO(blanked))
    schema, num_rows, metadata_only = readstat_arrow.read_sav_metadata(io.BytesIO(blanked))

    assert "num" not in metadata.formats  # the file declares none
    assert metadata.storage_widths == {"num": 8}  # and no format to read a declared width out of
    assert table.schema.field("num").type == pa.float64()  # nothing says it is a date
    assert table.column("num").to_pylist() == [1.0, 2.0]
    assert (num_rows, schema, metadata_only) == (2, table.schema, metadata)


def test_sav_utf8_string_values() -> None:
    table, _ = readstat_arrow.read_sav(DATA_DIR / "tegulu.sav")
    assert table.column("Q16br9oe_Q24br9oe").to_pylist() == ["నేను గతంలో వాడిన బ"]


def test_sav_non_ascii_variable_name() -> None:
    table, metadata = readstat_arrow.read_sav(DATA_DIR / "hebrews.sav")
    assert table.column_names == ["ותק_ב"]
    assert metadata.formats["ותק_ב"] == "F8.0"


def test_sav_very_long_strings() -> None:
    """SPSS stores strings over 255 bytes in 252-byte segments; ReadStat reassembles them."""
    table, metadata = readstat_arrow.read_sav(DATA_DIR / "test_width.sav")

    assert metadata.formats["StartDate"] == "A1024"
    assert metadata.storage_widths["StartDate"] == 1024
    # SPSS pads short strings out to 8-byte cells (A18 occupies 24), but the width reported is A18's own.
    assert metadata.formats["ResponseId"] == "A18"
    assert metadata.storage_widths["ResponseId"] == 18
    assert table.column("StartDate").to_pylist()[0] == "2020-07-13 23:19:55"
    assert metadata.formats["Duration__in_seconds_"] == "F40.2"


def test_sav_string_user_missing_values() -> None:
    """`MISSING VALUES mychar ('Z')`: a string value declared missing."""
    table, metadata = readstat_arrow.read_sav(DATA_DIR / "missing_char.sav")
    preserved, _ = readstat_arrow.read_sav(DATA_DIR / "missing_char.sav", preserve_user_missing=True)

    assert table.column("mychar").to_pylist() == [None, "a"]
    assert preserved.column("mychar").to_pylist() == ["Z", "a"]
    assert metadata.missing_values["mychar"] == {"values": ["Z"]}
    assert metadata.value_labels["mychar"] == [{"value": "a", "label": "labeled"}]


def test_sav_missing_ranges_and_labelled_missing_values() -> None:
    table, metadata = readstat_arrow.read_sav(DATA_DIR / "simple_alltypes.sav")
    preserved, _ = readstat_arrow.read_sav(DATA_DIR / "simple_alltypes.sav", preserve_user_missing=True)

    # Three discrete missing values ...
    assert metadata.missing_values["x"] == {"values": [7.0, 8.0, 99.0]}
    assert table.column("x").to_pylist() == [1.0, 2.0, 3.0, 4.0, None, 9.0]
    assert preserved.column("x").to_pylist() == [1.0, 2.0, 3.0, 4.0, 8.0, 9.0]
    # ... and a discrete value plus a range. SPSS's `LO THRU 0` comes back with the
    # concrete lower bound ReadStat reports, not -inf.
    assert metadata.missing_values["z"] == {"lo": -999.0, "hi": 0.0, "value": 999.0}
    assert table.column("z").to_pylist() == [None, None, 1.234, None, 3.14159, None]
    assert preserved.column("z").to_pylist() == [-9.0, None, 1.234, 999.0, 3.14159, None]
    # A missing value can itself carry a value label.
    assert metadata.value_labels["z"] == [{"value": 999.0, "label": "skipped"}]


def test_sav_multiple_response_sets() -> None:
    _schema, _rows, metadata = readstat_arrow.read_sav_metadata(DATA_DIR / "simple_alltypes.sav")
    assert metadata.multiple_response_sets == [
        {
            "name": "$categorical_array",
            "label": None,
            "type": "C",
            "is_dichotomy": False,
            "counted_value": None,
            "variables": ["ca_subvar_1", "ca_subvar_2", "ca_subvar_3"],
        },
        {
            "name": "$mymrset",
            "label": "My multiple response set",
            "type": "D",
            "is_dichotomy": True,
            "counted_value": 1,
            "variables": ["bool1", "bool2", "bool3"],
        },
    ]
    _schema, _rows, without = readstat_arrow.read_sav_metadata(DATA_DIR / "sample.sav")
    assert without.multiple_response_sets == []


def test_dta_tagged_missing_values_are_null_by_default() -> None:
    """Stata's .a-.z are nulls in the table, indistinguishable from '.', unless asked for."""
    table, metadata = readstat_arrow.read_dta(DATA_DIR / "missing_test.dta")

    assert table.schema.types == [pa.float32()] * 9
    assert table.to_pydict() == {f"var{i}": [None] for i in range(1, 9)} | {"var9": [1.0]}
    # A tag can carry a value label.
    assert metadata.value_labels["var1"] == [{"value": "a", "label": "missing"}]


def test_dta_tagged_missing_values_as_structs() -> None:
    """With preserve_user_missing=True every numeric column is struct<value, tag>."""
    table, _ = readstat_arrow.read_dta(DATA_DIR / "missing_test.dta", preserve_user_missing=True)

    tag_type = pa.dictionary(pa.int8(), pa.string())
    assert table.schema.types == [pa.struct([("value", pa.float32()), ("tag", tag_type)])] * 9
    assert table.column("var1").to_pylist() == [{"value": None, "tag": "a"}]
    assert table.column("var6").to_pylist() == [{"value": None, "tag": "z"}]
    assert table.column("var9").to_pylist() == [{"value": 1.0, "tag": None}]
    # A plain '.' is a null struct, so null_count still means "missing"; a tagged cell is not null.
    assert table.column("var7").to_pylist() == [None]
    assert table.column("var7").null_count == 1
    assert table.column("var1").null_count == 0


def test_dta_tag_structs_wrap_every_numeric_column() -> None:
    """The schema depends on the file's dictionary, not on which cells happen to be tagged."""
    table, _ = readstat_arrow.read_dta(DATA_DIR / "sample.dta", preserve_user_missing=True)
    for name in ("mynum", "mylabl", "mydate", "dtime", "mytime"):
        typ = table.schema.field(name).type
        assert pa.types.is_struct(typ) and [f.name for f in typ] == ["value", "tag"]
    assert table.schema.field("mychar").type == pa.large_string()  # strings cannot be missing in Stata
    assert table.schema.field("mydate").type.field("value").type == pa.date32()  # dates are still converted
    assert table.column("mynum").to_pylist()[0] == {"value": 1.1, "tag": None}
