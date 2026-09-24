# Vendored dependencies

## ReadStat

Upstream: https://github.com/WizardMac/ReadStat
Commit:   835b88c8c79d00fcd028c9cfa1226de2fd18be20

Copied verbatim from upstream; no local patches.

To re-sync:
    git clone https://github.com/WizardMac/ReadStat /tmp/ReadStat

    git -C /tmp/ReadStat log --format="%H" -1

    rm -rf vendor/ReadStat && \
    mkdir vendor/ReadStat && \
    cp /tmp/ReadStat/LICENSE vendor/ReadStat && \
    cp -r /tmp/ReadStat/src vendor/ReadStat/src && \
    rm -rf vendor/ReadStat/src/bin && \
    rm -rf vendor/ReadStat/src/fuzz && \
    rm -rf vendor/ReadStat/src/test && \
    rm -rf vendor/ReadStat/src/txt && \
    rm -rf /tmp/ReadStat

    # update the Commit lines above

## win-iconv

Upstream: https://github.com/win-iconv/win-iconv
Commit:   70a279dcfe6318bbe63e019e33b3230b55c19762

Copied verbatim from upstream; no local patches.

Only needed on Windows: ReadStat includes <iconv.h> unconditionally and MSVC
ships no such header.  win_iconv implements iconv on top of the Win32 codepage
APIs and is in the public domain, so -- unlike GNU libiconv (LGPL) -- it can be
compiled straight into the wheels without adding copyleft obligations.  See
setup.py for the compile-time defines it needs.

Three of upstream's eight files are used.  CMakeLists.txt and win_iconv_test.c
are build/test scaffolding, iconv.def exports symbols for a shared DLL (we link
statically), and localcharset.h is a consumer header for a function ReadStat
never calls.

To re-sync:
    git clone https://github.com/win-iconv/win-iconv /tmp/win-iconv

    git -C /tmp/win-iconv log --format="%H" -1

    rm -rf vendor/win-iconv && \
    mkdir vendor/win-iconv && \
    cp /tmp/win-iconv/iconv.h /tmp/win-iconv/win_iconv.c /tmp/win-iconv/readme.txt \
       vendor/win-iconv/ && \
    rm -rf /tmp/win-iconv

    # update the Commit lines above
