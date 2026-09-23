"""Reading columns at the narrowest type that holds them: ``scan_and_narrow_types``."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import pyarrow as pa
import pytest

import readstat_arrow
from conftest import DATA_DIR, READER_FUNCS, SAMPLES, WRITER_FUNCS
from readstat_arrow._cython import parser as _parser
from readstat_arrow._formats import FileFormat

SAMPLE_FILES = [path.name for path in sorted(DATA_DIR.glob("*.sav")) if path.name != "sample.zsav"] + [
    path.name for path in sorted(DATA_DIR.glob("*.dta"))
]


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_narrowing_never_changes_a_value(name: str) -> None:
    """Whatever width a column is read at, the values are the ones the file holds."""
    read = readstat_arrow.read_sav if name.endswith(".sav") else readstat_arrow.read_dta
    with warnings.catch_warnings():  # a few samples hold problems ReadStat recovers from
        warnings.simplefilter("ignore", readstat_arrow.ReadstatWarning)
        wide, wide_meta = read(DATA_DIR / name)
        narrow, narrow_meta = read(DATA_DIR / name, scan_and_narrow_types=True)

    assert narrow.cast(wide.schema).equals(wide)
    assert narrow_meta == wide_meta


def test_narrowing_measures_only_what_it_reads(fmt: FileFormat, tmp_path: Path) -> None:
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    path = tmp_path / f"window.{fmt}"
    write(
        path,
        pa.table({"x": pa.array([1.0, 2.0, 5000.0], pa.float64()), "y": pa.array([1.0, 2.0, 3.0])}),
    )

    whole, _ = read(path, scan_and_narrow_types=True)
    window, _ = read(path, scan_and_narrow_types=True, row_limit=2, columns=["x"])

    # Both passes get the same arguments, so the width always fits the rows read.
    assert whole.schema.field("x").type == pa.int16()  # 5000 needs the second byte
    assert window.schema.field("x").type == pa.int8()  # the two rows it read do not
    assert window.column_names == ["x"]
    assert window.column("x").to_pylist() == [1, 2]


def test_float32_is_used_when_every_value_survives_it(fmt: FileFormat, tmp_path: Path) -> None:
    read = READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    path = tmp_path / f"halves.{fmt}"
    write(path, pa.table({"x": pa.array([0.5, -1.25, None], pa.float64())}))

    table, _ = read(path, scan_and_narrow_types=True)

    assert table.schema.field("x").type == pa.float32()  # not whole numbers, but exact in four bytes
    assert table.column("x").to_pylist() == [0.5, -1.25, None]


def test_narrowing_reads_a_file_object_from_where_it_started(fmt: FileFormat) -> None:
    read = READER_FUNCS[fmt]
    padding = b"\x00" * 7
    data = padding + SAMPLES[fmt].read_bytes()
    file = io.BytesIO(data)
    file.seek(len(padding))

    # Two passes over one file object: the first has to leave it where it found
    # it, which is not the start of the stream.
    table, _ = read(file, scan_and_narrow_types=True)

    assert table.schema.field("mylabl").type == pa.int8()
    assert table.num_rows == 5


def test_narrowing_reads_a_sav_at_the_width_its_values_need() -> None:
    wide, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav")

    narrow, meta = readstat_arrow.read_sav(DATA_DIR / "sample.sav", scan_and_narrow_types=True)

    assert narrow.schema == pa.schema(
        {
            "mychar": pa.large_string(),  # a string is as wide as its bytes
            "mynum": pa.float64(),  # 1.1 survives neither an integer nor a float32
            "mydate": pa.date32(),
            "dtime": pa.timestamp("us"),
            "mylabl": pa.int8(),
            "myord": pa.int8(),
            "mytime": pa.time64("us"),
        }
    )
    assert narrow.nbytes < wide.nbytes
    assert narrow.cast(wide.schema).equals(wide)
    # The metadata is the file's, untouched by the width the columns were read at.
    assert meta == readstat_arrow.read_sav(DATA_DIR / "sample.sav")[1]


def test_sav_narrowing_leaves_a_column_alone_when_nothing_narrower_holds_it() -> None:
    stored, _, _ = readstat_arrow.read_sav_metadata(DATA_DIR / "test_width.sav")
    narrow, _ = readstat_arrow.read_sav(DATA_DIR / "test_width.sav", scan_and_narrow_types=True)

    assert stored.field("ResponseId").type == pa.large_string()
    assert narrow.schema.field("ResponseId").type == pa.large_string()
    assert stored.field("Duration__in_seconds_").type == pa.float64()
    assert narrow.schema.field("Duration__in_seconds_").type == pa.int16()  # 884 to 2611 seconds


def test_sav_narrowed_columns_keep_their_nulls() -> None:
    table, _ = readstat_arrow.read_sav(DATA_DIR / "sample_missing.sav", scan_and_narrow_types=True)

    assert table.schema.field("mylabl").type == pa.int8()
    assert table.column("mylabl").to_pylist() == [1, 2, 1, 2, 1, None, None]


def test_sav_nulls_are_left_out_of_the_range_a_type_has_to_cover() -> None:
    # mydate's last row is missing; the four that are there are whole seconds far
    # past int32, and the null costs nothing.
    table, _ = readstat_arrow.read_sav(DATA_DIR / "sample.sav", scan_and_narrow_types=True)

    assert table.schema.field("mydate").type == pa.date32()
    assert table.column("mydate")[4].as_py() is None


def test_sav_narrowing_sees_user_missing_values_when_the_read_keeps_them(tmp_path: Path) -> None:
    path = tmp_path / "declared.sav"
    readstat_arrow.write_sav(
        path,
        pa.table({"x": pa.array([1.0, 2.0, 999.0], pa.float64())}),
        readstat_arrow.Metadata(missing_values={"x": {"values": [999]}}),
    )

    default, _ = readstat_arrow.read_sav(path, scan_and_narrow_types=True)
    preserved, _ = readstat_arrow.read_sav(path, scan_and_narrow_types=True, preserve_user_missing=True)

    assert default.schema.field("x").type == pa.int8()  # 999 reads as a null and bounds nothing
    assert preserved.schema.field("x").type == pa.int16()  # kept, it needs the second byte
    assert preserved.column("x").to_pylist() == [1, 2, 999]


def test_narrowing_on_a_dta_whose_types_are_already_narrow() -> None:
    stored, _, _ = readstat_arrow.read_dta_metadata(DATA_DIR / "sample.dta")
    narrow, _ = readstat_arrow.read_dta(DATA_DIR / "sample.dta", scan_and_narrow_types=True)

    assert stored.field("mylabl").type == pa.int8()  # Stata stores small integers as int8
    assert narrow.schema.field("mylabl").type == pa.int8()  # so there is nothing to narrow
    assert narrow.schema.field("mynum").type == pa.float64()


def test_dta_narrowing_wraps_tagged_missings_around_the_narrowed_value() -> None:
    table, _ = readstat_arrow.read_dta(
        DATA_DIR / "sample.dta", scan_and_narrow_types=True, preserve_user_missing=True
    )

    assert table.schema.field("mylabl").type == pa.struct(
        {"value": pa.int8(), "tag": pa.dictionary(pa.int8(), pa.string())}
    )


def test_dta_column_of_nothing_but_nulls_narrows_to_one_byte_a_row() -> None:
    table, _ = readstat_arrow.read_dta(DATA_DIR / "missing_test.dta", scan_and_narrow_types=True)

    assert table.schema.field("var1").type == pa.int8()
    assert table.column("var1").to_pylist() == [None]


def test_sav_width_that_cannot_hold_a_value_is_refused_by_the_parser(tmp_path: Path) -> None:
    """The guard behind narrowing, which no measured width can trip.

    ReadStat converts a double to a narrower type by casting, which wraps around
    silently, so the parser range-checks every value of a narrowed column instead.
    Reached here through the compiled parser directly, since the widths a read
    uses are measured from the very rows it then reads.
    """
    path = tmp_path / "thousands.sav"
    readstat_arrow.write_sav(path, pa.table({"x": pa.array([1.0, 1000.0], pa.float64())}))
    encoded = str(path).encode()

    with pytest.raises(readstat_arrow.ReadstatError, match=r"x: row 1 holds 1000.0.*out of range.*int8"):
        _parser.parse(encoded, "sav", types={"x": pa.int8()})

    with pytest.raises(readstat_arrow.ReadstatError, match=r"not a whole number.*int8"):
        _parser.parse(str(DATA_DIR / "sample.sav").encode(), "sav", types={"mynum": pa.int8()})

    with pytest.raises(readstat_arrow.ReadstatError, match=r"out of range.*float"):
        huge = tmp_path / "huge.sav"
        readstat_arrow.write_sav(huge, pa.table({"x": pa.array([1e39], pa.float64())}))
        _parser.parse(str(huge).encode(), "sav", types={"x": pa.float32()})


def test_sav_type_no_column_can_be_read_into_is_refused_by_the_parser() -> None:
    sample = str(DATA_DIR / "sample.sav").encode()

    with pytest.raises(ValueError, match=r"uint8.*not a type"):
        _parser.parse(sample, "sav", types={"mylabl": pa.uint8()})

    with pytest.raises(ValueError, match=r"mychar.*cannot hold"):
        _parser.parse(sample, "sav", types={"mychar": pa.int8()})

    with pytest.raises(ValueError, match=r"mylabl.*cannot hold"):
        _parser.parse(sample, "sav", types={"mylabl": pa.large_string()})
