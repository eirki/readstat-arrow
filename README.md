# readstat-arrow

Read and write SPSS (`.sav`) and Stata (`.dta`) files as [Apache
Arrow](https://arrow.apache.org/) tables, using the excellent
[ReadStat](https://github.com/WizardMac/ReadStat) C library:

```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("survey.sav")  # -> (pyarrow.Table, Metadata)
readstat_arrow.write_dta("survey.dta", table, meta)  # the same pair back out, as Stata
```

## Motivation
There are several options in this space. `readstat-arrow` relies on the ReadStat
library for reading and writing since it is mature and battle-tested, and on
`Apache Arrow` for in-memory representation in Python since it provides
zero-copy interoperability with libraries like Pandas, Polars, and DuckDB.

Much of the inspiration for `readstat-arrow` comes from the great
[pyreadstat](https://github.com/Roche/pyreadstat) library.


## Status

Early development. The library works and is tested, but the API is not stable:
names, signatures and return shapes may change in any release, without a
deprecation period. Pin an exact version if you depend on it. There is currently
support for reading and writing SPSS and Stata files. More file formats can be
added on request.

## Examples

### Reading files

```python
import readstat_arrow

# SPSS
table, meta = readstat_arrow.read_sav("survey.sav")
# Stata:
table, meta = readstat_arrow.read_dta("survey.dta")

```

Every `read_*` function returns the same pair: a `pyarrow.Table`, and a
`Metadata` object.

`Metadata` is a set of mappings from variable name to one attribute —
`variable_labels`, `value_labels`, `formats`, `storage_widths`,
`display_widths`, `measures`, `alignments`, `missing_values` — plus the
file-level `file_label`, `notes` and `multiple_response_sets`. Which columns
exist, and in what order, is the Arrow schema's business, not the metadata's.
Nothing is required: a name absent from a mapping simply declares nothing.

```python
Metadata(
    file_label="2026 satisfaction survey",
    variable_labels={
        "id": "Respondent id",
        "q1": "How satisfied are you ...?",
        "q2": "How many hours ...?",
    },
    value_labels={
        "q1": [
            {"value": 1, "label": "Very unsatisfied"},
            {"value": 5, "label": "Very satisfied"},
            {"value": 9, "label": "No answer"},
        ]
    },
    formats={"q1": "F1.0", "q2": "F2.0"},
    measures={"q1": "ordinal", "q2": "scale"},
)
```


### Use with Pandas or Polars
```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("survey.sav")

# pandas:
df = table.to_pandas()
# polars:
import polars as pl
df = pl.from_arrow(table)
```

### Writing files:

```python
import pyarrow as pa

import readstat_arrow
from readstat_arrow import Metadata

table = pa.table({"id": [1, 2, 3], "q1": [1, 5, 4], "q2": [10, 11, 2]})

meta = Metadata(
    file_label="2026 satisfaction survey",
    variable_labels={
        "id": "Respondent id",
        "q1": "How satisfied are you ...?",
        "q2": "How many hours ...?",
    },
    value_labels={
        "q1": [
            {"value": 1, "label": "Very unsatisfied"},
            {"value": 5, "label": "Very satisfied"},
        ]
    },
    formats={"q1": "F1.0", "q2": "F2.0"},
    measures={"q1": "ordinal", "q2": "scale"},
)

# SPSS:
readstat_arrow.write_sav("survey.sav", table, meta)
# Stata:
readstat_arrow.write_dta("survey.dta", table, meta)
```

### Writing in batches
To avoid holding an entire table in memory at once, it is possible to write
files in batches


```python

import pyarrow as pa
import readstat_arrow
from readstat_arrow import Metadata

def my_data_source():
    # Let's pretend this comes from some external source
    for _ in range(10):
        yield pa.table(
            {
                "q1": pa.array([1, 2, 3, 4, 5], pa.int8()),
                "q2": pa.array([10, 20, 30, 40, 50], pa.int32()),
                "q3": pa.array([100, 200, 300, 400, 500], pa.int32()),
            }
        )

# we need to know these up front:
schema = pa.schema({"q1": pa.int8(), "q2": pa.int32(), "q3": pa.int32()})
num_rows = 50
meta = Metadata()

# SPSS:
with readstat_arrow.SavWriter("survey.sav", schema, num_rows, meta) as writer:
    for table in my_data_source():
        writer.write_table(table)
# Stata:
with readstat_arrow.DtaWriter("survey.dta", schema, num_rows, meta) as writer:
    for table in my_data_source():
        writer.write_table(table)

```




### Handling missing values

By default every kind of missing — system-missing, a Stata tagged missing, an
SPSS value the file declares missing — reads as an Arrow null. With
`preserve_user_missing=True` the user-level ones survive, in the way each format
has of saying them.

Stata tags its missings `.a` to `.z`, so every numeric column becomes a
`struct<value, tag>` to carry both:

```python
import readstat_arrow

table, meta = readstat_arrow.read_dta("panel.dta", preserve_user_missing=True)

table.schema.field("income").type  # -> struct<value: double, tag: dictionary<int8, string>>
table.column("income")[0].as_py()  # -> {"value": None, "tag": "a"}, Stata's .a
table.column("income")[1].as_py()  # -> {"value": 42.0, "tag": None}, a real number
table.column("income")[2].as_py()  # -> None, a plain .
```

SPSS instead declares ordinary values missing, so those values simply stay in
the column — the type is unchanged, and `Metadata.missing_values` says which
values were the missing ones:

```python
import readstat_arrow

default, meta = readstat_arrow.read_sav("survey.sav")
kept, _ = readstat_arrow.read_sav("survey.sav", preserve_user_missing=True)

meta.missing_values["q1"]  # -> {"values": [9.0]}, declared by MISSING VALUES q1 (9)

default.column("q1")[0].as_py()  # -> None, the 9 collapsed to null
kept.column("q1")[0].as_py()  # -> 9.0, the declared missing value itself
```

System-missing is null either way, and the writers accept both shapes back.

An SPSS declaration takes one of two shapes in `Metadata.missing_values`:

```python
Metadata(
    missing_values={
        # up to three discrete values: MISSING VALUES q1 (7, 8, 9)
        "q1": {"values": [7, 8, 9]},
        # an inclusive range: MISSING VALUES q2 (90 THRU 99)
        "q2": {"lo": 90, "hi": 99},
        # a range and one value beside it: MISSING VALUES q3 (LO THRU 0, 999)
        "q3": {"lo": float("-inf"), "hi": 0, "value": 999},
    }
)
```

`-inf` and `inf` are SPSS's `LO` and `HI`, and more than three discrete values
is an error — SPSS itself allows no more.


### Reading the metadata without the data

`read_sav_metadata` and `read_dta_metadata` stop before reading the actual data.
What comes back is the schema a full read would have given, the row count, and
the `Metadata` object.

```python
import readstat_arrow

schema, num_rows, meta = readstat_arrow.read_dta_metadata("panel.dta")

schema.names  # -> ["id", "year", "income", ...], the variables in file order
schema.field("income").type  # -> the type a full read would give that column
num_rows  # -> 4_000_000, from the header
meta.variable_labels["income"]  # -> "Annual income, NOK"
```

The row count is `None` where the file does not record one — Stata files always
do, some non-SPSS writers of `.sav` do not. The schema describes a
`preserve_user_missing=False` read, so it does not show the `struct<value, tag>`
columns that option gives a `.dta`.

### Reading only part of a file

Use `columns`, `row_offset` and `row_limit` to read parts of a file:

```python
import readstat_arrow

table, meta = readstat_arrow.read_dta(
    "panel.dta",
    columns=["id", "income"],  # only these two; they come back in file order
    row_offset=1_000,  # skip the first 1_000 rows
    row_limit=1_000,  # then read at most 1_000
)
```

`row_limit=0` means no limit, and the returned `Metadata` covers the columns
that were read, not the whole file.

### Narrowing types to reduce memory usage

In memory-constrained environments, it can be difficult to hold the whole table in
memory at once — especially since the type a column is stored as is often wider
than its values need. A `.sav` is the worst of it: every numeric column is a
64-bit double whatever it holds, so a even survey of one-digit codes costs 8 bytes a
cell. A `.dta` has narrow types of its own — `byte`, `int`, `long`, `float` —
but a variable is only as narrow as whoever wrote the file declared it.

`scan_and_narrow_types=True` reads each column at the width its values actually
need instead:

```python
import readstat_arrow

table, meta = readstat_arrow.read_sav("big.sav", scan_and_narrow_types=True)

table.schema.field("q1").type  # -> DataType(int8), where the file says double
```

The file is parsed twice — once to measure the values, keeping none of them,
then once to read them at the widths that fit — so the trade is time for memory.

Only a type that holds the column exactly is ever chosen: integer types when
every value was a whole number, `float32` when every value round-trips through
it, else `float64`. The ladder is `int8`, `int16`, `int32`, `float32`,
`float64`. Strings are untouched.

### Reading in batches

Read a fixed number of rows at a time and hand each one over as a
`pyarrow.RecordBatch`,

```python
import pyarrow.parquet as pq
import readstat_arrow

reader = readstat_arrow.open_sav("big.sav")
with pq.ParquetWriter("big.parquet", reader.schema) as writer:
    reader.read_batches(writer.write_batch)
```

`open_sav` and `open_dta` return a `SavStreamingReader` / `DtaStreamingReader`.
Opening reads the metadata and nothing else, so `schema`, `num_rows` and
`metadata` are all there before the data itself is read:

```python
reader = readstat_arrow.open_sav("panel.sav")
reader.schema  # the schema every batch has
reader.num_rows  # rows the header declares, or None
reader.metadata.variable_labels["income"]
```

`read_batches(callback)` then reads the file, calling `callback` with each batch
and returning the rows read.

The writers in `readstat-arrow` take a batch at a time too, so converting between the two
formats can be done on the fly:

```python
reader = readstat_arrow.open_sav("panel.sav")
with readstat_arrow.DtaWriter(
    "survey.dta", reader.schema, reader.num_rows, reader.metadata
) as writer:
    reader.read_batches(writer.write_batch)
```

A reader takes the same arguments the matching `read_*` takes: `columns`,
`row_offset`, `row_limit`, `encoding`, `preserve_user_missing`, and
`scan_and_narrow_types`.

### Making a `pyarrow.RecordBatchReader`

`read_batches` pushes: it drives the parse and calls you. Some consumers want to
pull instead — DuckDB, `pyarrow.dataset.write_dataset`, anything that takes a
`pyarrow.RecordBatchReader`. Turning one around into the other needs a thread,
and can be done like this:

```python
import queue
import threading

import pyarrow as pa


def record_batch_reader(reader, *, batch_rows=65_536, ahead=2):
    """A pyarrow.RecordBatchReader over a readstat-arrow streaming reader."""
    queued: queue.Queue = queue.Queue(maxsize=ahead)
    done = object()

    def run():
        try:
            reader.read_batches(queued.put, batch_rows=batch_rows)
        except BaseException as exc:  # comes back out of the consumer
            queued.put(exc)
        else:
            queued.put(done)

    threading.Thread(target=run, daemon=True).start()

    def batches():
        while True:
            item = queued.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    return pa.RecordBatchReader.from_batches(reader.schema, batches())
```

`maxsize` is the backpressure: the parse runs at most `ahead` batches in front
of whoever is reading and then waits, so the file is never held. What comes back
is an ordinary `pyarrow.RecordBatchReader`, which DuckDB will query in place:

```python
import duckdb
import readstat_arrow

survey = record_batch_reader(readstat_arrow.open_sav("big.sav"))
duckdb.sql("select region, avg(income) from survey group by region").show()
```

or `pyarrow.dataset` will write out partitioned:

```python
import pyarrow.dataset as ds

ds.write_dataset(
    record_batch_reader(readstat_arrow.open_dta("panel.dta")),
    "panel/",
    format="parquet",
)
```

One thing to know: a consumer that stops reading part way leaves the worker
thread parked on a full queue until the process ends. Read it to the end, or add
a flag the callback checks if that matters.

### Read from something other than a path

Every `read_*` function also takes a binary file object, so a file that arrives
over the network or out of an archive never has to be written to disk first.

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
    schema, num_rows, meta = readstat_arrow.read_sav_metadata(file)
```

The file object must be seekable, and is read from wherever it currently is - so
a `.sav` embedded in a larger stream can be read by seeking to its first byte.
The writers have taken a file object all along.

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
`src/readstat_arrow/_cython/` are excluded — they are typed for Cython's C type
system, which mypy cannot follow, and Cython checks them at compile time.

ReadStat is vendored as a git submodule at `vendor/ReadStat`; bump it with `git
submodule update --remote vendor/ReadStat`.

`uv sync` rebuilds the extension whenever `_cython/`, `setup.py` or the ReadStat
sources change (see `[tool.uv] cache-keys` in `pyproject.toml`).

## Layout

```
pyproject.toml            project metadata, deps, tool config (uv/ruff/mypy/pytest)
setup.py                  Cython extension definition (compiles ReadStat in)
vendor/ReadStat/          git submodule
src/readstat_arrow/
  __init__.py             public API re-exports
  reader.py               read_*, read_*_metadata, open_* (streaming), table assembly, type narrowing
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

Releases use [CalVer](https://calver.org/) in the form `YYYY.MM.DD.INC0` (e.g.
`2026.9.1.0`, then `2026.9.1.1` for a fix on the same day). There are no
compatibility promises encoded in the number. The version is set once in
`pyproject.toml` and exposed as `readstat_arrow.__version__`.

## Licence

MIT. ReadStat is MIT-licensed; the sample files under `tests/data/` come from
pyreadstat (Apache 2.0) — see `tests/data/README.md`.
