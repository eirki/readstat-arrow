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

DATA_DIR = Path(__file__).parent / "data"

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

# SPSS stores labelled values as doubles, Stata as integers.
SAV_MYLABL_LABELS = [{"value": 1.0, "label": "Male"}, {"value": 2.0, "label": "Female"}]
SAV_MYORD_LABELS = [
    {"value": 1.0, "label": "low"},
    {"value": 2.0, "label": "medium"},
    {"value": 3.0, "label": "high"},
]
DTA_MYLABL_LABELS = [{"value": 1, "label": "Male"}, {"value": 2, "label": "Female"}]
DTA_MYORD_LABELS = [
    {"value": 1, "label": "low"},
    {"value": 2, "label": "medium"},
    {"value": 3, "label": "high"},
]


def test_read_sav() -> None:
    table, meta = readstat_arrow.read_sav(DATA_DIR / "sample.sav")

    expected = pa.table(
        {
            "mychar": MYCHAR,
            "mynum": MYNUM,
            "mydate": MYDATE,
            "dtime": DTIME,
            "mylabl": pa.array(MYLABL, pa.float64()),
            "myord": pa.array(MYORD, pa.float64()),
            "mytime": MYTIME,
        }
    )
    assert table.equals(expected)
    assert table.column_names == expected.column_names
    assert meta.value_labels["mylabl"] == SAV_MYLABL_LABELS
    assert meta.value_labels["myord"] == SAV_MYORD_LABELS
    assert "mychar" not in meta.value_labels


def test_read_dta() -> None:
    table, meta = readstat_arrow.read_dta(DATA_DIR / "sample.dta")

    # Stata stores small integers as int8 (byte).
    expected = pa.table(
        {
            "mychar": MYCHAR,
            "mynum": MYNUM,
            "mydate": MYDATE,
            "dtime": DTIME,
            "mylabl": pa.array(MYLABL, pa.int8()),
            "myord": pa.array(MYORD, pa.int8()),
            "mytime": MYTIME,
        }
    )
    assert table.equals(expected)
    assert meta.formats["mytime"] == "%tcHH:MM:SS"
    assert meta.missing_values == {}  # an SPSS-only concept
    assert meta.value_labels["mylabl"] == DTA_MYLABL_LABELS
    assert meta.value_labels["myord"] == DTA_MYORD_LABELS


def test_variable_metadata() -> None:
    _, meta = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    assert meta.variable_labels["mychar"] == "character"
    assert meta.measures["mychar"] == "nominal"
    assert meta.storage_widths["mychar"] == 1  # A1: the declared width, not SPSS's 8-byte cell
    assert meta.notes  # sample.sav carries a document record


def test_column_selection() -> None:
    table, meta = readstat_arrow.read_sav(DATA_DIR / "sample.sav", columns=["mynum", "mychar"])
    # File order wins over the requested order.
    assert table.equals(pa.table({"mychar": MYCHAR, "mynum": MYNUM}))
    assert table.column_names == ["mychar", "mynum"]  # file order wins in the metadata too
    assert list(meta.formats) == ["mychar", "mynum"]


def test_row_limit_and_offset() -> None:
    table, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav", row_limit=2, row_offset=1)
    assert table.column("mychar").to_pylist() == ["b", "c"]
    assert table.num_rows == 2


def test_preserve_user_missing() -> None:
    default, meta = readstat_arrow.read_sav(DATA_DIR / "sample_missing.sav")
    kept, _ = readstat_arrow.read_sav(DATA_DIR / "sample_missing.sav", preserve_user_missing=True)

    assert meta.missing_values["mynum"] == {"lo": 2000.0, "hi": 3000.0, "value": -1.0}
    assert meta.missing_values["myord"] == {"values": [-1.0, -2.0, -3.0]}
    assert "mychar" not in meta.missing_values
    assert default.column("mynum").null_count > kept.column("mynum").null_count


def test_read_metadata_only() -> None:
    row_count, schema, _meta = readstat_arrow.read_dta_metadata(DATA_DIR / "sample.dta")
    assert row_count == 5
    assert schema.names == [
        "mychar",
        "mynum",
        "mydate",
        "dtime",
        "mylabl",
        "myord",
        "mytime",
    ]


@pytest.mark.parametrize(
    ("name", "read", "read_metadata"),
    [
        ("sample.sav", readstat_arrow.read_sav, readstat_arrow.read_sav_metadata),
        ("sample.dta", readstat_arrow.read_dta, readstat_arrow.read_dta_metadata),
    ],
)
def test_metadata_schema_matches_a_full_read(
    name: str, read: t.Callable[..., t.Any], read_metadata: t.Callable[..., t.Any]
) -> None:
    table, _ = read(DATA_DIR / name)
    _, schema, _meta = read_metadata(DATA_DIR / name)

    assert schema == table.schema


def test_clean_file_emits_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", readstat_arrow.ReadstatWarning)
        readstat_arrow.read_sav(DATA_DIR / "sample.sav")
        readstat_arrow.read_dta(DATA_DIR / "sample.dta")


def test_recoverable_parse_problems_become_warnings() -> None:
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


def test_variable_without_a_display_format() -> None:
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

    table, meta = readstat_arrow.read_sav(io.BytesIO(blanked))
    row_count, schema, metadata_only = readstat_arrow.read_sav_metadata(io.BytesIO(blanked))

    assert "num" not in meta.formats  # the file declares none
    assert meta.storage_widths == {"num": 8}  # and no format to read a declared width out of
    assert table.schema.field("num").type == pa.float64()  # nothing says it is a date
    assert table.column("num").to_pylist() == [1.0, 2.0]
    assert (row_count, schema, metadata_only) == (2, table.schema, meta)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(readstat_arrow.ReadstatError):
        readstat_arrow.read_sav(tmp_path / "nope.sav")


def test_table_survives_ipc_roundtrip() -> None:
    table, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    back = pa.ipc.open_stream(sink.getvalue()).read_all()
    assert back.equals(table)
    assert back.schema.metadata is None


def test_read_sav_metadata() -> None:
    row_count, _schema, meta = readstat_arrow.read_sav_metadata(DATA_DIR / "sample.sav")
    assert row_count == 5
    assert meta.value_labels["mylabl"] == SAV_MYLABL_LABELS


def test_utf8_string_values() -> None:
    table, _ = readstat_arrow.read_sav(DATA_DIR / "tegulu.sav")
    assert table.column("Q16br9oe_Q24br9oe").to_pylist() == ["నేను గతంలో వాడిన బ"]


def test_non_ascii_variable_name() -> None:
    table, meta = readstat_arrow.read_sav(DATA_DIR / "hebrews.sav")
    assert table.column_names == ["ותק_ב"]
    assert meta.formats["ותק_ב"] == "F8.0"


def test_very_long_strings() -> None:
    """SPSS stores strings over 255 bytes in 252-byte segments; ReadStat reassembles them."""
    table, meta = readstat_arrow.read_sav(DATA_DIR / "test_width.sav")

    assert meta.formats["StartDate"] == "A1024"
    assert meta.storage_widths["StartDate"] == 1024
    # SPSS pads short strings out to 8-byte cells (A18 occupies 24), but the width reported is A18's own.
    assert meta.formats["ResponseId"] == "A18"
    assert meta.storage_widths["ResponseId"] == 18
    assert table.column("StartDate").to_pylist()[0] == "2020-07-13 23:19:55"
    assert meta.formats["Duration__in_seconds_"] == "F40.2"


def test_string_user_missing_values() -> None:
    """`MISSING VALUES mychar ('Z')`: a string value declared missing."""
    table, meta = readstat_arrow.read_sav(DATA_DIR / "missing_char.sav")
    preserved, _ = readstat_arrow.read_sav(DATA_DIR / "missing_char.sav", preserve_user_missing=True)

    assert table.column("mychar").to_pylist() == [None, "a"]
    assert preserved.column("mychar").to_pylist() == ["Z", "a"]
    assert meta.missing_values["mychar"] == {"values": ["Z"]}
    assert meta.value_labels["mychar"] == [{"value": "a", "label": "labeled"}]


def test_missing_ranges_and_labelled_missing_values() -> None:
    table, meta = readstat_arrow.read_sav(DATA_DIR / "simple_alltypes.sav")
    preserved, _ = readstat_arrow.read_sav(DATA_DIR / "simple_alltypes.sav", preserve_user_missing=True)

    # Three discrete missing values ...
    assert meta.missing_values["x"] == {"values": [7.0, 8.0, 99.0]}
    assert table.column("x").to_pylist() == [1.0, 2.0, 3.0, 4.0, None, 9.0]
    assert preserved.column("x").to_pylist() == [1.0, 2.0, 3.0, 4.0, 8.0, 9.0]
    # ... and a discrete value plus a range. SPSS's `LO THRU 0` comes back with the
    # concrete lower bound ReadStat reports, not -inf.
    assert meta.missing_values["z"] == {"lo": -999.0, "hi": 0.0, "value": 999.0}
    assert table.column("z").to_pylist() == [None, None, 1.234, None, 3.14159, None]
    assert preserved.column("z").to_pylist() == [-9.0, None, 1.234, 999.0, 3.14159, None]
    # A missing value can itself carry a value label.
    assert meta.value_labels["z"] == [{"value": 999.0, "label": "skipped"}]


def test_multiple_response_sets() -> None:
    _, _schema, meta = readstat_arrow.read_sav_metadata(DATA_DIR / "simple_alltypes.sav")
    assert meta.multiple_response_sets == [
        {
            "name": "categorical_array",
            "label": None,
            "type": "C",
            "is_dichotomy": False,
            "counted_value": None,
            "variables": ["ca_subvar_1", "ca_subvar_2", "ca_subvar_3"],
        },
        {
            "name": "mymrset",
            "label": "My multiple response set",
            "type": "D",
            "is_dichotomy": True,
            "counted_value": 1,
            "variables": ["bool1", "bool2", "bool3"],
        },
    ]
    _, _schema, without = readstat_arrow.read_sav_metadata(DATA_DIR / "sample.sav")
    assert without.multiple_response_sets == []


def test_stata_tagged_missing_values_are_null_by_default() -> None:
    """Stata's .a-.z are nulls in the table, indistinguishable from '.', unless asked for."""
    table, meta = readstat_arrow.read_dta(DATA_DIR / "missing_test.dta")

    assert table.schema.types == [pa.float32()] * 9
    assert table.to_pydict() == {f"var{i}": [None] for i in range(1, 9)} | {"var9": [1.0]}
    # A tag can carry a value label.
    assert meta.value_labels["var1"] == [{"value": "a", "label": "missing"}]


def test_stata_tagged_missing_values_as_structs() -> None:
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


def test_tag_structs_wrap_every_numeric_column() -> None:
    """The schema depends on the file's dictionary, not on which cells happen to be tagged."""
    table, _ = readstat_arrow.read_dta(DATA_DIR / "sample.dta", preserve_user_missing=True)
    for name in ("mynum", "mylabl", "mydate", "dtime", "mytime"):
        typ = table.schema.field(name).type
        assert pa.types.is_struct(typ) and [f.name for f in typ] == ["value", "tag"]
    assert table.schema.field("mychar").type == pa.large_string()  # strings cannot be missing in Stata
    assert table.schema.field("mydate").type.field("value").type == pa.date32()  # dates are still converted
    assert table.column("mynum").to_pylist()[0] == {"value": 1.1, "tag": None}


@pytest.mark.parametrize("name", ["sample.sav", "sample.dta"])
def test_read_from_file_object(name: str) -> None:
    """A file object gives exactly what the same file at a path gives."""
    path = DATA_DIR / name
    read = readstat_arrow.read_sav if name.endswith(".sav") else readstat_arrow.read_dta

    expected, expected_meta = read(path)
    with path.open("rb") as file:
        table, meta = read(file)
        assert not file.closed  # the caller's file object is left open

    assert table.equals(expected)
    assert meta == expected_meta


def test_read_from_an_os_encoded_path() -> None:
    """``os.PathLike`` is not the only path: bytes are handed to ReadStat as they are."""
    expected, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    table, _ = readstat_arrow.read_sav(os.fsencode(DATA_DIR / "sample.sav"))
    assert table.equals(expected)


def test_read_from_bytes_io() -> None:
    expected, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    table, _ = readstat_arrow.read_sav(io.BytesIO((DATA_DIR / "sample.sav").read_bytes()))
    assert table.equals(expected)


def test_read_metadata_from_file_object() -> None:
    expected = readstat_arrow.read_dta_metadata(DATA_DIR / "sample.dta")
    with (DATA_DIR / "sample.dta").open("rb") as file:
        assert readstat_arrow.read_dta_metadata(file) == expected


def test_read_from_file_object_honours_options() -> None:
    """The options are the parser's, not the path's: a file object gets all of them."""
    table, _ = readstat_arrow.read_sav(
        io.BytesIO((DATA_DIR / "sample.sav").read_bytes()),
        columns=["mynum", "mydate"],
        row_limit=2,
        row_offset=1,
    )
    assert table.column_names == ["mynum", "mydate"]
    assert table.column("mynum").to_pylist() == [1.2, -1000.3]


def test_read_from_unbuffered_file_object() -> None:
    """A raw file can return a short read; the io handler asks again rather than stopping."""
    expected, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    with (DATA_DIR / "sample.sav").open("rb", buffering=0) as file:
        table, _ = readstat_arrow.read_sav(file)
    assert table.equals(expected)


def test_read_from_file_object_starts_where_it_is() -> None:
    """Reading begins at the file object's position, so a file embedded in a stream works."""
    expected, _ = readstat_arrow.read_dta(DATA_DIR / "sample.dta")
    stream = io.BytesIO(b"PREFIX!!" + (DATA_DIR / "sample.dta").read_bytes())
    stream.seek(8)

    table, _ = readstat_arrow.read_dta(stream)
    assert table.equals(expected)


def test_read_rejects_unseekable_file_object() -> None:
    class Unseekable(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

    with pytest.raises(ValueError, match="not seekable"):
        unseekable: t.Any = Unseekable()
        readstat_arrow.read_sav(unseekable)


def test_read_reports_a_failure_of_the_file_object() -> None:
    """An exception from the file object reaches the caller, not ReadStat's account of it."""

    class Failing(io.BytesIO):
        def readinto(self, buffer: object) -> int:
            raise OSError("no disk today")

    with pytest.raises(OSError, match="no disk today"):
        readstat_arrow.read_sav(Failing((DATA_DIR / "sample.sav").read_bytes()))


def test_read_truncated_file_object() -> None:
    truncated = io.BytesIO((DATA_DIR / "sample.sav").read_bytes()[:120])
    with pytest.raises(readstat_arrow.ReadstatError):
        readstat_arrow.read_sav(truncated)


def test_read_from_a_minimal_file_object() -> None:
    """Only read/seek/tell are required - no readinto, and no seekable to ask."""

    class Minimal:
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)

        def read(self, size: int = -1) -> bytes:
            return self._buffer.read(size)

        def seek(self, offset: int, whence: int = 0) -> int:
            return self._buffer.seek(offset, whence)

        def tell(self) -> int:
            return self._buffer.tell()

    expected, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    minimal: t.Any = Minimal((DATA_DIR / "sample.sav").read_bytes())
    table, _ = readstat_arrow.read_sav(minimal)
    assert table.equals(expected)


def test_read_from_zip_member(tmp_path: Path) -> None:
    """A file inside an archive, never extracted to disk."""
    archive_path = tmp_path / "survey.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(DATA_DIR / "sample.sav", "survey.sav")

    expected, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")
    with zipfile.ZipFile(archive_path) as archive, archive.open("survey.sav") as member:
        table, _ = readstat_arrow.read_sav(member)
    assert table.equals(expected)
