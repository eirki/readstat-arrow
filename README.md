# readstat-arrow

Read and write SPSS (`.sav`) and Stata (`.dta`) files as [Apache Arrow](https://arrow.apache.org/)
tables.

A thin, typed Python wrapper around the excellent [ReadStat](https://github.com/WizardMac/ReadStat)
C library. Where [pyreadstat](https://github.com/Roche/pyreadstat) returns pandas data frames,
`readstat-arrow` returns a `pyarrow.Table`, which converts for free to pandas, polars, DuckDB,
Parquet and anything else that speaks Arrow.

```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("survey.sav")  # -> (pyarrow.Table, Metadata)
readstat_arrow.write_dta("survey.dta", table, meta)  # the same pair back out, as Stata
```

## Examples

### Write an SPSS file

```python
import pyarrow as pa
import readstat_arrow
from readstat_arrow import Metadata

table = pa.table({"id": pa.array([1, 2, 3], pa.int32()), "sex": pa.array([1, 2, None])})

meta = Metadata(
    file_label="Tiny survey",
    variable_labels={"id": "Respondent id", "sex": "Sex"},
    value_labels={
        "sex": [
            {"value": 1, "label": "Male"},
            {"value": 2, "label": "Female"},
            {"value": 9, "label": "Unknown"},
        ]
    },
    formats={"sex": "F1.0"},
    measures={"sex": "nominal"},
    storage_widths={"sex": 1},
    missing_values={"sex": {"values": [9]}},  # or a range: {"lo": 90, "hi": 99}
)

readstat_arrow.write_sav("tiny.sav", table, meta)

readstat_arrow.write_sav("bare.sav", table)  # no metadata at all: just the columns
```

`Metadata` is a set of mappings from variable name to one attribute — `variable_labels`,
`value_labels`, `formats`, `storage_widths`, `display_widths`, `measures`, `alignments`,
`missing_values` — plus the file-level `file_label`, `notes` and `multiple_response_sets`. Which
columns exist, and in what order, is the Arrow schema's business. Nothing is required: a name absent
from a mapping declares nothing and the writer falls back to the format's own default, and the
metadata may be left out altogether. SPSS stores every number as a double, so `id` reads back as
`float64`, not `int32`.

### Write a Stata file, whole or in batches

```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("survey.sav")
readstat_arrow.write_dta("survey.dta", table, meta)  # a whole table in one call

# or stream it, without holding every row in memory
with readstat_arrow.DtaWriter("survey.dta", table.schema, table.num_rows, meta) as writer:
    for batch in table.to_batches(max_chunksize=100_000):
        writer.write_batch(batch)
```

`DtaWriter` (and `SavWriter`) needs the final row count up front.

### Read an SPSS file

```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("survey.sav")

table.num_rows  # -> 12_345
table.schema  # dates and times are already temporal types
table.column("q1")

table.column_names  # -> every variable, in file order
meta.file_label  # -> "2026 satisfaction survey"
meta.variable_labels["q1"]  # -> "How satisfied are you ...?"
meta.value_labels["q1"]  # -> [{"value": 1, "label": "Very unsatisfied"}, ...]
meta.missing_values["q1"]  # -> {"values": [9]}, or {"lo": 90, "hi": 99}

table.to_pandas()  # or
import polars as pl

pl.from_arrow(table)
```

### Read part of a Stata file

```python
import readstat_arrow

# header only, no rows decoded: the schema a full read would give comes back
# beside the metadata
row_count, schema, meta = readstat_arrow.read_dta_metadata("panel.dta")
schema.names  # -> ["id", "income", ...], the variables in file order
schema.field("income").type  # -> the type read_dta would give that column

table, meta = readstat_arrow.read_dta(
    "panel.dta",
    columns=["id", "income"],  # only these, in this order
    row_offset=1_000,  # skip the first 1_000 rows
    row_limit=1_000,  # then read at most 1_000
    preserve_user_missing=True,  # keep Stata's tagged missings .a-.z
)

# with preserve_user_missing every numeric column is struct<value, tag>
table.column("income")[0].as_py()  # -> {"value": None, "tag": "a"}, Stata's .a
table.column("income")[1].as_py()  # -> {"value": 42.0, "tag": None}, a real number
table.column("income")[2].as_py()  # -> None, a plain .
```

### Read a big file in less memory

Where memory is the constraint, holding the whole table in it is the expensive part of reading a
file — and the type a column is stored as is often wider than its values need. A `.sav` is the worst
of it: every numeric column is a 64-bit double whatever it holds, so a survey of one-digit codes
costs 8 bytes a cell. A `.dta` has narrow types of its own — `byte`, `int`, `long`, `float` — but a
variable is only as narrow as whoever wrote the file declared it.

`scan_and_narrow_types=True` reads each column at the width its values actually need instead:

```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("big.sav", scan_and_narrow_types=True)

table.schema.field("q1").type  # -> DataType(int8), where the file says double
```

The file is parsed twice — once to measure the values, keeping none of them, then once to read them
at the widths that fit — so the trade is time for memory: about twice the wall clock of a plain read,
and no column is ever built at its stored width.

Only a type that holds the column exactly is ever chosen: integer types when every value was a whole
number, `float32` when every value round-trips through it, else `float64`. The ladder is `int8`,
`int16`, `int32`, `float32`, `float64` — there is no `int64`, which would save nothing over the
double it replaces, and no unsigned type. Strings are untouched, and a column of nothing but nulls
comes back as `int8`. Dates and times come back as the same temporal types either way: the width
they are read at only decides how much there is to convert.

### Read from something other than a path

Every `read_*` function also takes a binary file object, so a file that arrives over the network or
out of an archive never has to be written to disk first.

```python
import io, zipfile
import readstat_arrow

# straight out of a zip archive, without extracting it
with zipfile.ZipFile("survey.zip") as archive, archive.open("survey.sav") as member:
    table, meta = readstat_arrow.read_sav(member)

# or from bytes you already have in hand
table, meta = readstat_arrow.read_sav(io.BytesIO(downloaded))

# an open file works too, and is left open where reading stopped
with open("survey.sav", "rb") as file:
    row_count, schema, meta = readstat_arrow.read_sav_metadata(file)
```

The file object must be seekable, and is read from wherever it currently is - so a `.sav` embedded
in a larger stream can be read by seeking to its first byte. The writers have taken a file object
all along.


## Development

Requires [uv](https://docs.astral.sh/uv/) and a C compiler.

```sh
git clone <repo-url>
cd readstat-arrow
uv sync            # builds the Cython extension into .venv
uv run coverage run -m pytest && uv run coverage report || uv run  coverage html
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pre-commit install    # optional: run those same checks on every commit
```

`uv run mypy` checks `src/` and `tests/` in strict mode. The Cython sources in
`src/readstat_arrow/_cython/` are excluded — they are typed for Cython's C type system, which mypy
cannot follow, and Cython checks them at compile time.

ReadStat is vendored as a git submodule at `vendor/ReadStat`; bump it with `git submodule update
--remote vendor/ReadStat`. `uv sync` rebuilds the extension whenever `_cython/`, `setup.py` or the
ReadStat sources change (see `[tool.uv] cache-keys` in `pyproject.toml`).

## Layout

```
pyproject.toml            project metadata, deps, tool config (uv/ruff/mypy/pytest)
setup.py                  Cython extension definition (compiles ReadStat in)
vendor/ReadStat/          git submodule
src/readstat_arrow/
  __init__.py             public API re-exports
  reader.py               read_* and read_*_metadata functions, table assembly, type narrowing
  writer.py               SavWriter / DtaWriter, write_* functions, type planning
  metadata.py             the Metadata dataclass and its per-variable mappings
  errors.py               ReadstatError, ReadstatWarning
  _formats.py             FileFormat literal type
  _dates.py               display-format -> temporal type conversion
  _cython/                everything Cython compiles (private)
    parser.py             pure-Python-mode Cython: ReadStat callbacks -> Arrow buffers
    writer.py             pure-Python-mode Cython: Arrow buffers -> readstat_insert_*
    readstat.pxd          C declarations for readstat.h
tests/                    pytest suite; sample files under tests/data/
```

## Versioning

Releases use [CalVer](https://calver.org/) in the form `YYYY.MM.DD.INC0` (e.g. `2026.09.01.0`, then
`2026.09.01.1` for a fix in the same dat). There are no compatibility promises encoded in the
number. The version is set once in `pyproject.toml` and exposed as `readstat_arrow.__version__`.

## Licence

MIT. ReadStat is MIT-licensed; the sample files under `tests/data/` come from pyreadstat (Apache
2.0) — see `tests/data/README.md`.
