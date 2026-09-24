"""The two file formats, so a test can be written once and run twice.

A test that takes ``fmt`` runs against both ``.sav`` and ``.dta``: it looks its
entry points up by format name. A test that names ``read_sav`` or ``write_dta``
directly is one that is about that format alone.

Only the API lives here. What a format is expected to *produce* stays in the test
module, as a literal per format keyed by ``fmt``.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import pytest

import readstat_arrow
from readstat_arrow._formats import FileFormat, file_format_values

DATA_DIR = Path(__file__).parent / "data"

# ``tests/data/sample.*``: the same five rows in both formats.
SAMPLES = {"sav": DATA_DIR / "sample.sav", "dta": DATA_DIR / "sample.dta"}
READER_FUNCS = {"sav": readstat_arrow.read_sav, "dta": readstat_arrow.read_dta}
OPEN_FUNCS = {"sav": readstat_arrow.open_sav, "dta": readstat_arrow.open_dta}
METADATA_READER_FUNCS = {
    "sav": readstat_arrow.read_sav_metadata,
    "dta": readstat_arrow.read_dta_metadata,
}
WRITER_FUNCS = {"sav": readstat_arrow.write_sav, "dta": readstat_arrow.write_dta}
WRITER_CLASSES = {"sav": readstat_arrow.SavWriter, "dta": readstat_arrow.DtaWriter}


@pytest.fixture(name="fmt", params=file_format_values, ids=file_format_values)
def make_format(request: pytest.FixtureRequest) -> FileFormat:
    return t.cast(FileFormat, request.param)
