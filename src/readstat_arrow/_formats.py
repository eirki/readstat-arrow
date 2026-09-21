"""The file formats this package reads and writes."""

import typing as t

FileFormat = t.Literal["sav", "dta"]
file_format_values = t.get_args(FileFormat)
