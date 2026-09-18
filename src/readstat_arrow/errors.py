"""Exception types raised by readstat-arrow."""


class ReadstatError(Exception):
    """Raised when the underlying ReadStat C library reports an error."""


class ReadstatWarning(UserWarning):
    """Emitted for problems ReadStat recovered from while reading a file.

    Examples: a value-label record pointing at an unknown variable, text that
    could not be transliterated to UTF-8, an unrecognised record that was skipped.
    The read still succeeds; filter or escalate with the :mod:`warnings` module.
    """
