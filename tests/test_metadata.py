"""The Metadata container: its mappings, editing them, and its repr."""

from __future__ import annotations

import io
from dataclasses import replace
from textwrap import dedent

import pyarrow as pa
import pytest

import readstat_arrow
from conftest import DATA_DIR, METADATA_READER_FUNCS, SAMPLES, WRITER_FUNCS
from readstat_arrow import Code, Metadata
from readstat_arrow._formats import FileFormat

# SPSS stores labelled values as doubles, Stata as integers.
SAMPLE_VALUE_LABELS: dict[FileFormat, dict[str, list[Code]]] = {
    "sav": {
        "mylabl": [{"value": 1.0, "label": "Male"}, {"value": 2.0, "label": "Female"}],
        "myord": [
            {"value": 1.0, "label": "low"},
            {"value": 2.0, "label": "medium"},
            {"value": 3.0, "label": "high"},
        ],
    },
    "dta": {
        "mylabl": [{"value": 1, "label": "Male"}, {"value": 2, "label": "Female"}],
        "myord": [
            {"value": 1, "label": "low"},
            {"value": 2, "label": "medium"},
            {"value": 3, "label": "high"},
        ],
    },
}
YES_NO: dict[FileFormat, list[Code]] = {
    "sav": [{"value": 1.0, "label": "Yes"}, {"value": 2.0, "label": "No"}],
    "dta": [{"value": 1, "label": "Yes"}, {"value": 2, "label": "No"}],
}


def test_mappings_hold_only_what_the_file_declares(fmt: FileFormat) -> None:
    read_metadata = METADATA_READER_FUNCS[fmt]
    schema, _num_rows, metadata = read_metadata(SAMPLES[fmt])

    assert schema.names == ["mychar", "mynum", "mydate", "dtime", "mylabl", "myord", "mytime"]
    assert metadata.variable_labels["mychar"] == "character"
    assert metadata.value_labels == SAMPLE_VALUE_LABELS[fmt]
    assert "mychar" not in metadata.value_labels  # ... and an undeclared name is simply absent


def test_describing_a_variable_is_ordinary_dictionary_work() -> None:
    metadata = Metadata()
    metadata.variable_labels["agree"] = "Agrees with statement"
    metadata.value_labels["agree"] = [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]

    assert metadata.variable_labels == {"agree": "Agrees with statement"}
    assert metadata.value_labels == {"agree": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]}


def test_variables_do_not_share_label_lists(fmt: FileFormat) -> None:
    """Variables that share one label set in the file come back with independent lists."""
    read_metadata = METADATA_READER_FUNCS[fmt]
    write = WRITER_FUNCS[fmt]
    codes: list[Code] = [{"value": 1, "label": "Yes"}, {"value": 2, "label": "No"}]
    metadata = Metadata(value_labels={"q1": codes, "q2": codes})
    table = pa.table({"q1": pa.array([1], pa.int8()), "q2": pa.array([2], pa.int8())})
    shared = io.BytesIO()
    write(shared, table, metadata)  # identical lists -> one label set

    shared.seek(0)
    _schema, _num_rows, back = read_metadata(shared)
    q1, q2 = back.value_labels["q1"], back.value_labels["q2"]
    assert q1 == q2 == YES_NO[fmt]
    assert q1 is not q2
    assert q1[0] is not q2[0]  # the Codes are copies too, not shared dictionaries


def test_rename_variable(fmt: FileFormat) -> None:
    read_metadata = METADATA_READER_FUNCS[fmt]
    _schema, _num_rows, metadata = read_metadata(SAMPLES[fmt])

    renamed = metadata.rename_variable("mylabl", "sex")

    assert renamed.value_labels["sex"] == metadata.value_labels["mylabl"]
    assert renamed.variable_labels["sex"] == metadata.variable_labels["mylabl"]
    assert "mylabl" not in renamed.value_labels  # gone from every mapping ...
    assert "mylabl" not in renamed.formats
    assert "mylabl" in metadata.value_labels  # ... and the original is untouched
    # The entry keeps its place, so the repr still reads in file order.
    assert list(renamed.formats) == ["mychar", "mynum", "mydate", "dtime", "sex", "myord", "mytime"]

    with pytest.raises(ValueError, match="'myord' already declares something"):
        metadata.rename_variable("mylabl", "myord")


def test_repr_cuts_long_mappings_short() -> None:
    """A file's worth of variables has to stay readable at a prompt."""
    metadata = Metadata(
        variable_labels={f"v{i}": f"Variable {i}" for i in range(5)},
        missing_values={"v0": {"values": [9.0]}},
        file_label="Big",
    )

    assert repr(metadata) == dedent("""\
        Metadata(
            variable_labels={'v0': 'Variable 0', 'v1': 'Variable 1', 'v2': 'Variable 2', ... +2 more},
            value_labels={},
            formats={},
            storage_widths={},
            display_widths={},
            measures={},
            alignments={},
            missing_values={'v0': {'values': [9.0]}},
            file_label='Big',
            notes=[],
            multiple_response_sets=[]
        )""")


def _rename_all(metadata: Metadata, names: list[str], *, suffix: str) -> Metadata:
    for name in names:
        metadata = metadata.rename_variable(name, name + suffix)
    return metadata


# Tests of one format alone: a file, a record or a rule the other format has no equivalent of.
# Nothing below takes ``fmt``; each says in its name which format it is about.


def test_sav_merge() -> None:
    _sav_schema, _num_rows, sav = readstat_arrow.read_sav_metadata(DATA_DIR / "sample.sav")
    other_schema, _other_num_rows, other = readstat_arrow.read_sav_metadata(DATA_DIR / "sample_missing.sav")
    # Give the second file distinct variable names, as if it were another block of columns.
    other = _rename_all(other, other_schema.names, suffix="_b")

    merged = sav.merge(other)

    assert merged.notes == [*sav.notes, *other.notes]
    assert merged.file_label == sav.file_label
    # Every variable keeps its own labels; nothing to reconcile between the files.
    assert merged.value_labels["mylabl"] == sav.value_labels["mylabl"]
    assert merged.value_labels["mylabl_b"] == other.value_labels["mylabl_b"]
    assert merged.missing_values["myord_b"] == {"values": [-1.0, -2.0, -3.0]}


def test_sav_merge_prefers_self_on_overlap() -> None:
    schema, _num_rows, metadata = readstat_arrow.read_sav_metadata(DATA_DIR / "sample.sav")
    labels: dict[str, str | None] = {name: "other" for name in schema.names}
    labels["extra"] = "Extra"
    other = replace(metadata, variable_labels=labels)

    merged = metadata.merge(other)

    assert merged.variable_labels["mychar"] == "character"  # self's version, not "other"
    assert merged.variable_labels["extra"] == "Extra"  # ... but other's own entries come along
    assert metadata.merge(metadata) == replace(metadata, notes=[*metadata.notes, *metadata.notes])
