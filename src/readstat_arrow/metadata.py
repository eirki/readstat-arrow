"""Typed containers for file- and variable-level metadata."""

from __future__ import annotations

import typing as t
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

Measure = t.Literal["unknown", "nominal", "ordinal", "scale"]
Alignment = t.Literal["unknown", "left", "center", "right"]

# A value of a variable: numbers for numeric variables, str for string variables
# and for Stata tagged-missing values (``"a"`` for ``.a``).
Value = int | float | str


class Code(t.TypedDict):
    """One labelled value of a variable: ``{"value": 1, "label": "Male"}``."""

    value: Value
    label: str


class MissingValues(t.TypedDict):
    """One to three discrete values declared missing: ``{"values": [7, 8, 9]}``.

    SPSS's ``MISSING VALUES q (7, 8, 9)``.
    """

    values: Sequence[Value]


class MissingRange(t.TypedDict):
    """An inclusive range declared missing: ``{"lo": -999, "hi": 0}``.

    SPSS's ``MISSING VALUES q (LO THRU 0)``. ``lo`` may be ``-inf`` and ``hi``
    ``inf``, SPSS's ``LO`` and ``HI``. One discrete value fits beside the range -
    ``{"lo": -999, "hi": 0, "value": 999}``s
    """

    lo: Value
    hi: Value
    value: t.NotRequired[Value]


# What one variable declares missing. By default these values are read as null;
# with preserve_user_missing=True they stay in the data, and the declaration is
# how to tell them from real ones. Stata has no equivalent (its tagged missings
# .a-.z are preserved as a `tag` field on the column instead), so nothing is
# declared for a .dta variable.
Missingness = MissingValues | MissingRange

PER_VARIABLE = (
    "variable_labels",
    "value_labels",
    "formats",
    "storage_widths",
    "display_widths",
    "measures",
    "alignments",
    "missing_values",
)


_REPR_ITEMS = 3


def _elided(items: Sequence[object]) -> str:
    """``[a, b, c, ... +7 more]``: the first few entries, then how many were left out."""
    shown = ", ".join(repr(item) for item in items[:_REPR_ITEMS])
    return (
        f"[{shown}]"
        if len(items) <= _REPR_ITEMS
        else f"[{shown}, ... +{len(items) - _REPR_ITEMS} more]"
    )


def _elided_map(mapping: Mapping[object, object]) -> str:
    """``{a: 1, b: 2, ... +7 more}``, in the same spirit as :func:`_elided`."""
    items = list(mapping.items())
    shown = ", ".join(f"{k!r}: {v!r}" for k, v in items[:_REPR_ITEMS])
    return (
        f"{{{shown}}}"
        if len(items) <= _REPR_ITEMS
        else f"{{{shown}, ... +{len(items) - _REPR_ITEMS} more}}"
    )


@dataclass(slots=True, repr=False)
class Metadata:
    """Everything a file records besides the data itself.

    Every per-variable field is a mapping from a variable name to one attribute,
    so reading a label is ``metadata.variable_labels["q1"]`` and setting one is an
    ordinary dictionary assignment. A name that is absent, or mapped to ``None``,
    declares nothing, and the writers then fall back to the format's own default -
    so ``Metadata()`` is a valid "nothing declared" starting point, and a mapping
    built by dict comprehension need not filter its empty entries out.
    """

    variable_labels: Mapping[str, str | None] = field(default_factory=dict)
    value_labels: Mapping[str, list[Code] | None] = field(default_factory=dict)
    formats: Mapping[str, str | None] = field(default_factory=dict)
    storage_widths: Mapping[str, int | None] = field(default_factory=dict)
    display_widths: Mapping[str, int | None] = field(default_factory=dict)
    measures: Mapping[str, Measure | None] = field(default_factory=dict)
    alignments: Mapping[str, Alignment | None] = field(default_factory=dict)
    missing_values: Mapping[str, Missingness | None] = field(default_factory=dict)

    file_label: str | None = None
    notes: list[str] = field(default_factory=list)
    multiple_response_sets: list[dict[str, t.Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        """One field per line, with long lists and mappings cut short."""
        parts = [f"{name}={_elided_map(getattr(self, name))}" for name in PER_VARIABLE]
        parts += [
            f"file_label={self.file_label!r}",
            f"notes={_elided(self.notes)}",
            f"multiple_response_sets={_elided(self.multiple_response_sets)}",
        ]
        body = ",\n".join(f"    {p}" for p in parts)
        return f"Metadata(\n{body}\n)"

    def rename_variable(self, old: str, new: str) -> Metadata:
        """Move everything declared about ``old`` to ``new``, in every mapping.

        Returns a new Metadata; the original is untouched. Each entry keeps its
        place in its mapping. Raises ``ValueError`` if ``new`` already declares
        something, which renaming onto it would discard.
        """
        if any(new in getattr(self, field_name) for field_name in PER_VARIABLE):
            raise ValueError(f"variable {new!r} already declares something")
        renamed: dict[str, t.Any] = {
            field_name: {
                (new if k == old else k): v
                for k, v in getattr(self, field_name).items()
            }
            for field_name in PER_VARIABLE
        }
        return replace(self, **renamed)

    def merge(self, other: Metadata) -> Metadata:
        """Combine with ``other`` (e.g. before writing two tables' columns side by side).

        Returns a new Metadata; both originals are untouched.

        Every mapping is merged; on a name clash ``self`` wins. File-level fields come
        from ``self``; notes and multiple-response sets are concatenated.
        """
        combined: dict[str, t.Any] = {
            field_name: {**getattr(other, field_name), **getattr(self, field_name)}
            for field_name in PER_VARIABLE
        }
        return replace(
            self,
            notes=[*self.notes, *other.notes],
            multiple_response_sets=[
                *self.multiple_response_sets,
                *other.multiple_response_sets,
            ],
            **combined,
        )
