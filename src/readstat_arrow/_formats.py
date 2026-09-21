"""The file formats this package reads and writes."""

import typing as t

FileFormat = t.Literal["sav", "dta"]
file_format_values = t.get_args(FileFormat)

T = t.TypeVar("T")


class FormatMap(t.TypedDict, t.Generic[T]):
    """One ``T`` per file format.

    Adding a format means adding a field here, and mypy then reports every table that
    has not been filled in for it
    """

    sav: T
    dta: T


# The name of the software each format belongs to, for messages aimed at the caller.
DISPLAY_NAME: FormatMap[str] = {"sav": "SPSS", "dta": "Stata"}

# Whether the format stores Stata-style tagged missing values (.a to .z) ...
SUPPORTS_TAGGED_MISSING: FormatMap[bool] = {"sav": False, "dta": True}

# ... and whether it stores SPSS-style user-defined missing values (values and ranges).
SUPPORTS_USER_MISSING: FormatMap[bool] = {"sav": True, "dta": False}


def is_native_format(file_format: FileFormat, fmt: str) -> bool:
    """Whether ``fmt`` is a display format of ``file_format``'s own family.

    Each family is recognisable from the start of the string: Stata's formats are
    "%"-prefixed (``%td``, ``%9.0g``), SPSS's begin with the format name (``F8.2``,
    ``DATE11``). A format from another family says nothing about how this file should
    display the column, so callers drop it rather than pass it on.
    """
    if file_format == "dta":
        return fmt.startswith("%")
    elif file_format == "sav":
        return fmt[:1].isalpha()
    else:
        t.assert_never(file_format)
