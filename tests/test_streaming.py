"""Tests for the streaming readers, ``open_sav`` / ``open_dta``.

What a batch holds is settled by :mod:`test_read` - these check that reading a
file a batch at a time gives back exactly what reading it whole gives back,
whatever the batch size, and that a read that goes wrong says so.
"""

from __future__ import annotations

import io
import threading
import typing as t
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import readstat_arrow
from conftest import DATA_DIR, METADATA_READER_FUNCS, OPEN_FUNCS, READER_FUNCS, SAMPLES
from readstat_arrow._formats import FileFormat

SAMPLE_ROWS = 5


def read_whole(fmt: FileFormat, **kwargs: t.Any) -> tuple[pa.Table, readstat_arrow.Metadata]:
    return READER_FUNCS[fmt](SAMPLES[fmt], **kwargs)


def batches_of(fmt: FileFormat, **sizing: int) -> list[pa.RecordBatch]:
    """Every batch of a streamed read, and the reader's own schema checked against each."""
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    collected: list[pa.RecordBatch] = []
    assert reader.read_batches(collected.append, **sizing) == sum(b.num_rows for b in collected)
    assert all(batch.schema.equals(reader.schema) for batch in collected)
    return collected


# ---------------------------------------------------------------------------
# the batches add up to the whole
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_rows", [1, 2, 4, SAMPLE_ROWS, 1000])
def test_batches_reassemble_into_the_whole_table(fmt: FileFormat, batch_rows: int) -> None:
    expected, _ = read_whole(fmt)
    batches = batches_of(fmt, batch_rows=batch_rows)
    assert pa.Table.from_batches(batches, expected.schema).equals(expected)


@pytest.mark.parametrize(
    ("batch_rows", "expected"),
    [(1, [1] * SAMPLE_ROWS), (2, [2, 2, 1]), (5, [5]), (6, [5])],
)
def test_batch_sizes(fmt: FileFormat, batch_rows: int, expected: list[int]) -> None:
    """Full batches, then the remainder - and nothing empty when the rows divide evenly."""
    assert [b.num_rows for b in batches_of(fmt, batch_rows=batch_rows)] == expected


def test_schema_and_num_rows_are_what_the_metadata_reader_reports(fmt: FileFormat) -> None:
    schema, expected_rows, _ = METADATA_READER_FUNCS[fmt](SAMPLES[fmt])
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    assert reader.schema.equals(schema)
    assert reader.num_rows == expected_rows == SAMPLE_ROWS


def test_reading_uses_no_thread(fmt: FileFormat) -> None:
    """The parse drives, in the calling thread: there is nobody to hand a batch to."""
    before = threading.active_count()
    during: list[int] = []
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    reader.read_batches(lambda batch: during.append(threading.active_count()), batch_rows=1)
    assert during and all(count == before for count in during)
    assert threading.active_count() == before


def test_a_reader_can_be_read_again(fmt: FileFormat) -> None:
    """Nothing is consumed by reading, so a second pass gives the same rows."""
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    assert reader.read_all().equals(reader.read_all())


def test_opening_reads_no_rows(fmt: FileFormat) -> None:
    """Opening is the metadata pass and nothing else."""
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    assert reader.schema.names and reader.metadata.variable_labels
    assert reader.read_all().num_rows == SAMPLE_ROWS  # the rows were still all there


# ---------------------------------------------------------------------------
# the read options carry over
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"columns": ["mynum", "mychar"]},
        {"row_limit": 3},
        {"row_offset": 2},
        {"row_offset": 1, "row_limit": 2},
        {"scan_and_narrow_types": True},
        {"preserve_user_missing": True},
        {"columns": ["mynum"], "scan_and_narrow_types": True},
    ],
)
def test_options_match_the_whole_file_read(fmt: FileFormat, options: dict[str, t.Any]) -> None:
    expected, expected_metadata = read_whole(fmt, **options)
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt], **options)
    assert reader.read_all().equals(expected)
    assert reader.metadata == expected_metadata


def test_row_offset_past_the_end_gives_a_schema_and_no_batches(fmt: FileFormat) -> None:
    schema, _, _ = METADATA_READER_FUNCS[fmt](SAMPLES[fmt])
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt], row_offset=SAMPLE_ROWS + 10)
    assert reader.schema.equals(schema)
    assert reader.read_batches(lambda batch: None) == 0


def test_reads_from_a_file_object(fmt: FileFormat) -> None:
    expected, _ = read_whole(fmt)
    with SAMPLES[fmt].open("rb") as file:
        assert OPEN_FUNCS[fmt](file).read_all().equals(expected)


def test_a_stream_is_left_where_the_next_read_needs_it(fmt: FileFormat) -> None:
    """Both the scan and each read are passes of their own, so each has to rewind."""
    expected, _ = read_whole(fmt, scan_and_narrow_types=True)
    stream = io.BytesIO(SAMPLES[fmt].read_bytes())
    reader = OPEN_FUNCS[fmt](stream, scan_and_narrow_types=True)
    assert reader.read_all().equals(expected)
    assert reader.read_all().equals(expected)  # and again, from the same stream


def test_tagged_missings_keep_one_type_across_batches() -> None:
    """A batch with no tagged missing in it is still a struct, or the batches would not match."""
    expected, _ = readstat_arrow.read_dta(DATA_DIR / "missing_test.dta", preserve_user_missing=True)
    reader = readstat_arrow.open_dta(DATA_DIR / "missing_test.dta", preserve_user_missing=True)
    batches: list[pa.RecordBatch] = []
    reader.read_batches(batches.append, batch_rows=1)
    assert all(batch.schema.equals(reader.schema) for batch in batches)
    assert pa.Table.from_batches(batches, reader.schema).equals(expected)
    assert any(pa.types.is_struct(field.type) for field in expected.schema)


def test_the_metadata_is_complete_before_the_first_batch(fmt: FileFormat) -> None:
    """Including a .dta's value labels, which the format keeps after the data.

    They are there because the metadata is read in a pass of its own before the
    batches start, rather than accumulated as they arrive.
    """
    _, expected = read_whole(fmt)
    assert expected.value_labels
    assert OPEN_FUNCS[fmt](SAMPLES[fmt]).metadata == expected


def test_the_metadata_covers_only_the_columns_being_read(fmt: FileFormat) -> None:
    """The metadata pass sees every variable; a read of two declares nothing of the rest."""
    metadata = OPEN_FUNCS[fmt](SAMPLES[fmt], columns=["mychar", "mylabl"]).metadata
    assert set(metadata.variable_labels) <= {"mychar", "mylabl"}
    assert set(metadata.value_labels) == {"mylabl"}


# ---------------------------------------------------------------------------
# stopping, and going wrong
# ---------------------------------------------------------------------------


def test_raising_from_the_callback_abandons_the_parse(fmt: FileFormat) -> None:
    """Stopping early is raising; the exception comes back out with the parse dropped."""
    seen = 0

    def stop_after_two(batch: pa.RecordBatch) -> None:
        nonlocal seen
        seen += 1
        if seen == 2:
            raise ZeroDivisionError

    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    with pytest.raises(ZeroDivisionError):
        reader.read_batches(stop_after_two, batch_rows=1)
    assert seen == 2  # and not the remaining three


def test_recoverable_parse_problems_become_warnings() -> None:
    """As :func:`test_read.test_sav_recoverable_parse_problems_become_warnings`, streamed.

    This one is reported while ReadStat is still on the header, so it comes out
    of opening the file rather than out of reading it.
    """
    out = io.BytesIO()
    readstat_arrow.write_sav(out, pa.table({"averylongvariablename": pa.array([1.0, 2.0])}))
    data = out.getvalue().replace(b"AVERYLON=averylong", b"AVERYLOX=averylong")

    with pytest.warns(readstat_arrow.ReadstatWarning, match="Failed to find AVERYLOX"):
        reader = readstat_arrow.open_sav(io.BytesIO(data))
    assert reader.read_all().column_names == ["AVERYLON"]


def test_batch_rows_must_be_positive(fmt: FileFormat) -> None:
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        reader.read_batches(lambda batch: None, batch_rows=0)


def truncated_sample(tmp_path: Path, fmt: FileFormat) -> Path:
    """The sample file with its last tenth cut off, which is into the data in both formats."""
    out = tmp_path / f"truncated.{fmt}"
    whole = SAMPLES[fmt].read_bytes()
    out.write_bytes(whole[: len(whole) * 9 // 10])
    return out


def test_a_truncated_dta_raises_on_opening(tmp_path: Path) -> None:
    """The metadata pass has to reach the value labels at the end, so it fails first."""
    with pytest.raises(readstat_arrow.ReadstatError):
        readstat_arrow.open_dta(truncated_sample(tmp_path, "dta"))


def test_a_truncated_sav_raises_while_reading(tmp_path: Path) -> None:
    """The metadata pass stops at the dictionary, so nothing notices until the batches."""
    reader = readstat_arrow.open_sav(truncated_sample(tmp_path, "sav"))
    assert reader.schema.names
    with pytest.raises(readstat_arrow.ReadstatError):
        reader.read_batches(lambda batch: None)


def test_a_file_that_fails_partway_delivers_what_came_before(tmp_path: Path) -> None:
    """The batches read before the damage are handed over; then the error is raised.

    Written and truncated here rather than taken from ``tests/data``: the samples
    are small enough that a .sav is through its data before the first batch is
    handed over, leaving nothing to go wrong by then.
    """
    rows = 20_000
    schema = pa.schema([("n", pa.float64()), ("s", pa.large_string())])
    table = pa.table(
        {
            "n": pa.array(range(rows), pa.float64()),
            "s": pa.array([f"row {i}" for i in range(rows)], pa.large_string()),
        },
        schema=schema,
    )
    whole = tmp_path / "big.sav"
    with readstat_arrow.SavWriter(whole, schema, rows) as writer:
        writer.write_table(table)
    written = whole.read_bytes()
    truncated = tmp_path / "truncated.sav"
    truncated.write_bytes(written[: len(written) * 6 // 10])

    read = 0

    def count(batch: pa.RecordBatch) -> None:
        nonlocal read
        read += batch.num_rows

    reader = readstat_arrow.open_sav(truncated)
    with pytest.raises(readstat_arrow.ReadstatError):
        reader.read_batches(count, batch_rows=1000)
    assert 0 < read < rows


# ---------------------------------------------------------------------------
# what it is for
# ---------------------------------------------------------------------------


def test_converts_to_parquet_a_batch_at_a_time(tmp_path: Path, fmt: FileFormat) -> None:
    expected, _ = read_whole(fmt)
    out = tmp_path / "out.parquet"
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    with pq.ParquetWriter(out, reader.schema) as writer:
        assert reader.read_batches(writer.write_batch, batch_rows=2) == expected.num_rows
    written = pq.read_table(out)
    assert written.schema.names == expected.schema.names
    # Parquet has no large_string of its own; everything else survives as it was.
    assert written.cast(expected.schema).equals(expected)
    assert pq.ParquetFile(out).num_row_groups == 3


def test_converts_between_formats_a_batch_at_a_time(tmp_path: Path, fmt: FileFormat) -> None:
    """The batches go straight into the writers, so neither side holds the file."""
    other: FileFormat = "dta" if fmt == "sav" else "sav"
    out = tmp_path / f"converted.{other}"
    writer_class = readstat_arrow.SavWriter if other == "sav" else readstat_arrow.DtaWriter
    reader = OPEN_FUNCS[fmt](SAMPLES[fmt])
    assert reader.num_rows is not None
    with writer_class(out, reader.schema, reader.num_rows, reader.metadata) as writer:
        reader.read_batches(writer.write_batch)
    round_tripped, round_tripped_meta = READER_FUNCS[other](out)
    expected, _ = read_whole(fmt)
    assert round_tripped.num_rows == expected.num_rows
    assert round_tripped.column("mychar").equals(expected.column("mychar"))
    assert round_tripped_meta.variable_labels == reader.metadata.variable_labels
