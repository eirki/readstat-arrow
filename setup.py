from __future__ import annotations

import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent
READSTAT_SRC = ROOT / "vendor" / "ReadStat" / "src"
CYTHON_DIR = ROOT / "src" / "readstat_arrow" / "_cython"
WIN_ICONV_DIR = ROOT / "vendor" / "win-iconv"


# ReadStat sources: the core plus the SPSS (.sav) and Stata (.dta) modules.
# SAS and SPSS-portable support is left out until we need it.
#
# The zsav translation units go too: they call deflate()/uncompress() with no
# #if HAVE_ZLIB guard of their own, so compiling them into a build that does not
# link zlib (see below) leaves those symbols undefined in the .so.  Lazy binding
# hides that on some interpreters and fails the import outright on others.
# Their callers in readstat_sav_{read,write}.c are all inside #if HAVE_ZLIB, so
# dropping the files costs nothing here.
_EXCLUDED_DIRS = {"bin", "test", "fuzz", "sas"}
_EXCLUDED_FILES = {
    "readstat_por.c",
    "readstat_por_parse.c",
    "readstat_por_read.c",
    "readstat_por_write.c",
    "readstat_zsav_compress.c",
    "readstat_zsav_read.c",
    "readstat_zsav_write.c",
}
readstat_sources = sorted(
    str(p.relative_to(ROOT))
    for p in READSTAT_SRC.rglob("*.c")
    if not (set(p.relative_to(READSTAT_SRC).parts[:-1]) & _EXCLUDED_DIRS) and p.name not in _EXCLUDED_FILES
)

# No zlib: .zsav (zlib-compressed .sav) is out of scope, so ReadStat is built
# without HAVE_ZLIB and reports such files as "unsupported compression".
#
# iconv, on the other hand, is not optional: ReadStat's readstat_iconv.h includes
# <iconv.h> unconditionally.  glibc bundles an implementation, macOS ships one in
# the SDK, and MSVC has none at all -- hence the vendored win_iconv on Windows.
if sys.platform == "win32":
    iconv_sources = [str((WIN_ICONV_DIR / "win_iconv.c").relative_to(ROOT))]
    iconv_include_dirs = [str(WIN_ICONV_DIR)]
    libraries: list[str] = []
    define_macros: list[tuple[str, str]] = [
        # win_iconv's iconv.h honours ICONV_CONST when it is already defined, but
        # readstat_iconv.h defines it (to empty) *after* including <iconv.h>.  Left
        # alone the two disagree: iconv() gets declared taking `const char **inbuf`
        # while ReadStat passes `char **`.  Define it up front so both pick `char **`.
        ("ICONV_CONST", ""),
        # locale_charset() is unused here; dropping it keeps the symbol out of the
        # extension.  Note we deliberately do *not* define USE_LIBICONV_DLL, which
        # would let $WINICONV_LIBICONV_DLL redirect conversion to an arbitrary DLL.
        ("DISABLE_LOCALE_CHARSET", "1"),
        ("_CRT_SECURE_NO_WARNINGS", "1"),
    ]
else:
    iconv_sources = []
    iconv_include_dirs = []
    libraries = ["iconv"] if sys.platform == "darwin" else []  # glibc bundles iconv
    define_macros = []


def extension(module: str) -> Extension:
    """One extension module per compiled file; each links its own copy of ReadStat."""
    return Extension(
        f"readstat_arrow._cython.{module}",
        sources=[
            str((CYTHON_DIR / f"{module}.py").relative_to(ROOT)),
            *readstat_sources,
            *iconv_sources,
        ],
        include_dirs=[str(READSTAT_SRC), *iconv_include_dirs],
        libraries=libraries,
        define_macros=define_macros,
    )


extensions = [extension("parser"), extension("writer")]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        include_path=[str(ROOT / "src")],
        compiler_directives={"embedsignature": True},
    ),
)
