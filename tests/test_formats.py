"""The tables keyed by file format: that they have a column for every format.

``FormatMap`` is what makes mypy report every per-format table a new format leaves
unfilled, the way :func:`typing.assert_never` reports every dispatch that has not
grown a branch. Nothing in the type system ties its fields to ``FileFormat`` itself,
though, so that last link is checked here.
"""

from __future__ import annotations

import typing as t

from readstat_arrow._dates import TemporalKind, TemporalMap
from readstat_arrow._formats import FormatMap, file_format_values


def test_format_map_has_a_field_per_file_format() -> None:
    assert FormatMap.__required_keys__ == frozenset(file_format_values)


def test_temporal_map_has_a_field_per_temporal_kind() -> None:
    assert TemporalMap.__required_keys__ == frozenset(t.get_args(TemporalKind))
