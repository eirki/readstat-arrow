from __future__ import annotations

import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent
READSTAT_SRC = ROOT / "vendor" / "ReadStat" / "src"
CYTHON_DIR = ROOT / "src" / "readstat_arrow" / "_cython"
WIN_ICONV_DIR = ROOT / "vendor" / "win-iconv"
ZLIB_DIR = ROOT / "vendor" / "zlib"


# ReadStat sources: the core plus the SPSS (.sav, .zsav) and Stata (.dta) modules.
# SAS and SPSS-portable support is left out until we need it.
_EXCLUDED_DIRS = {"bin", "test", "fuzz", "sas"}
_EXCLUDED_FILES = {
    "readstat_por.c",
    "readstat_por_parse.c",
    "readstat_por_read.c",
    "readstat_por_write.c",
}
readstat_sources = sorted(
    str(p.relative_to(ROOT))
    for p in READSTAT_SRC.rglob("*.c")
    if not (set(p.relative_to(READSTAT_SRC).parts[:-1]) & _EXCLUDED_DIRS)
    and p.name not in _EXCLUDED_FILES
)

# zlib, for .zsav (zlib-compressed .sav), is compiled in from vendor/zlib on
# every platform.  Only the sources behind deflate/inflate/uncompress are built;
# the gz* file API is unused.
#
# Z_PREFIX renames every zlib symbol to z_*, through macros in zconf.h that
# ReadStat picks up by including <zlib.h>.  Without it our deflate() would be an
# exported symbol in the .so and could be resolved against - or shadow - the
# system libz that the interpreter's own zlib module has already loaded.
zlib_sources = sorted(str(p.relative_to(ROOT)) for p in ZLIB_DIR.glob("*.c"))
zlib_macros = [("HAVE_ZLIB", "1"), ("Z_PREFIX", "1")]

# iconv is needed too: ReadStat's readstat_iconv.h includes
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
            *zlib_sources,
            *iconv_sources,
        ],
        include_dirs=[str(READSTAT_SRC), str(ZLIB_DIR), *iconv_include_dirs],
        libraries=libraries,
        define_macros=[*zlib_macros, *define_macros],
    )


extensions = [extension("parser"), extension("writer")]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        include_path=[str(ROOT / "src")],
        compiler_directives={"embedsignature": True},
    )
)
