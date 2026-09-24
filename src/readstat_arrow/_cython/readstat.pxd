# Cython declarations for the ReadStat C API (vendor/ReadStat/src/readstat.h).
# Compiled together with parser.py; see readstat_arrow._cython.
#
# Reader and writer subsets of the API, in that order. SAS and SPSS-portable
# functions are left out until they are needed.

from posix.types cimport off_t

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t
from libc.time cimport time_t


cdef extern from "readstat.h":

    enum:
        READSTAT_HANDLER_OK
        READSTAT_HANDLER_ABORT
        READSTAT_HANDLER_SKIP_VARIABLE

    ctypedef enum readstat_type_t:
        READSTAT_TYPE_STRING
        READSTAT_TYPE_INT8
        READSTAT_TYPE_INT16
        READSTAT_TYPE_INT32
        READSTAT_TYPE_FLOAT
        READSTAT_TYPE_DOUBLE
        READSTAT_TYPE_STRING_REF

    ctypedef enum readstat_type_class_t:
        READSTAT_TYPE_CLASS_STRING
        READSTAT_TYPE_CLASS_NUMERIC

    ctypedef enum readstat_measure_t:
        READSTAT_MEASURE_UNKNOWN
        READSTAT_MEASURE_NOMINAL
        READSTAT_MEASURE_ORDINAL
        READSTAT_MEASURE_SCALE

    ctypedef enum readstat_alignment_t:
        READSTAT_ALIGNMENT_UNKNOWN
        READSTAT_ALIGNMENT_LEFT
        READSTAT_ALIGNMENT_CENTER
        READSTAT_ALIGNMENT_RIGHT

    ctypedef enum readstat_compress_t:
        READSTAT_COMPRESS_NONE
        READSTAT_COMPRESS_ROWS
        READSTAT_COMPRESS_BINARY

    ctypedef enum readstat_endian_t:
        READSTAT_ENDIAN_NONE
        READSTAT_ENDIAN_LITTLE
        READSTAT_ENDIAN_BIG

    ctypedef enum readstat_error_t:
        READSTAT_OK
        READSTAT_ERROR_OPEN
        READSTAT_ERROR_READ
        READSTAT_ERROR_MALLOC
        READSTAT_ERROR_USER_ABORT
        READSTAT_ERROR_PARSE
        # ... remaining codes are only ever passed to readstat_error_message()

    const char *readstat_error_message(readstat_error_t error_code)

    ctypedef struct mr_set_t:
        char   type
        char  *name
        char  *label
        int    is_dichotomy
        int    counted_value
        char **subvariables
        int    num_subvars

    ctypedef struct readstat_metadata_t:
        pass

    int readstat_get_row_count(readstat_metadata_t *metadata)
    int readstat_get_var_count(readstat_metadata_t *metadata)
    time_t readstat_get_creation_time(readstat_metadata_t *metadata)
    time_t readstat_get_modified_time(readstat_metadata_t *metadata)
    int readstat_get_file_format_version(readstat_metadata_t *metadata)
    int readstat_get_file_format_is_64bit(readstat_metadata_t *metadata)
    readstat_compress_t readstat_get_compression(readstat_metadata_t *metadata)
    readstat_endian_t readstat_get_endianness(readstat_metadata_t *metadata)
    const char *readstat_get_table_name(readstat_metadata_t *metadata)
    const char *readstat_get_file_label(readstat_metadata_t *metadata)
    const char *readstat_get_file_encoding(readstat_metadata_t *metadata)
    const mr_set_t *readstat_get_multiple_response_sets(readstat_metadata_t *metadata)
    size_t readstat_get_multiple_response_sets_length(readstat_metadata_t *metadata)

    ctypedef struct readstat_value_t:
        pass

    ctypedef struct readstat_variable_t:
        pass

    # Value accessors
    readstat_type_t readstat_value_type(readstat_value_t value)
    readstat_type_class_t readstat_value_type_class(readstat_value_t value)
    int readstat_value_is_missing(readstat_value_t value, readstat_variable_t *variable)
    int readstat_value_is_system_missing(readstat_value_t value)
    int readstat_value_is_tagged_missing(readstat_value_t value)
    int readstat_value_is_defined_missing(readstat_value_t value, readstat_variable_t *variable)
    char readstat_value_tag(readstat_value_t value)
    char readstat_int8_value(readstat_value_t value)
    int16_t readstat_int16_value(readstat_value_t value)
    int32_t readstat_int32_value(readstat_value_t value)
    float readstat_float_value(readstat_value_t value)
    double readstat_double_value(readstat_value_t value)
    const char *readstat_string_value(readstat_value_t value)

    # Variable accessors
    int readstat_variable_get_index(const readstat_variable_t *variable)
    int readstat_variable_get_index_after_skipping(const readstat_variable_t *variable)
    const char *readstat_variable_get_name(const readstat_variable_t *variable)
    const char *readstat_variable_get_label(const readstat_variable_t *variable)
    const char *readstat_variable_get_format(const readstat_variable_t *variable)
    const char *readstat_variable_get_informat(const readstat_variable_t *variable)
    readstat_type_t readstat_variable_get_type(const readstat_variable_t *variable)
    readstat_type_class_t readstat_variable_get_type_class(const readstat_variable_t *variable)
    size_t readstat_variable_get_storage_width(const readstat_variable_t *variable)
    int readstat_variable_get_display_width(const readstat_variable_t *variable)
    readstat_measure_t readstat_variable_get_measure(const readstat_variable_t *variable)
    readstat_alignment_t readstat_variable_get_alignment(const readstat_variable_t *variable)
    int readstat_variable_get_missing_ranges_count(const readstat_variable_t *variable)
    readstat_value_t readstat_variable_get_missing_range_lo(const readstat_variable_t *variable, int i)
    readstat_value_t readstat_variable_get_missing_range_hi(const readstat_variable_t *variable, int i)

    # Callback types
    ctypedef int (*readstat_metadata_handler)(readstat_metadata_t *metadata, void *ctx)
    ctypedef int (*readstat_note_handler)(int note_index, const char *note, void *ctx)
    ctypedef int (*readstat_variable_handler)(int index, readstat_variable_t *variable,
                                              const char *val_labels, void *ctx)
    ctypedef int (*readstat_fweight_handler)(readstat_variable_t *variable, void *ctx)
    ctypedef int (*readstat_value_handler)(int obs_index, readstat_variable_t *variable,
                                           readstat_value_t value, void *ctx)
    ctypedef int (*readstat_value_label_handler)(const char *val_labels, readstat_value_t value,
                                                 const char *label, void *ctx)
    ctypedef void (*readstat_error_handler)(const char *error_message, void *ctx)
    ctypedef int (*readstat_progress_handler)(double progress, void *ctx)

    # I/O handlers: override them to read from something other than a path on disk.
    ctypedef off_t readstat_off_t

    ctypedef enum readstat_io_flags_t:
        READSTAT_SEEK_SET
        READSTAT_SEEK_CUR
        READSTAT_SEEK_END

    ctypedef int (*readstat_open_handler)(const char *path, void *io_ctx)
    ctypedef int (*readstat_close_handler)(void *io_ctx)
    ctypedef readstat_off_t (*readstat_seek_handler)(readstat_off_t offset, readstat_io_flags_t whence,
                                                     void *io_ctx)
    ctypedef ssize_t (*readstat_read_handler)(void *buf, size_t nbyte, void *io_ctx)
    ctypedef readstat_error_t (*readstat_update_handler)(long file_size,
                                                          readstat_progress_handler progress_handler,
                                                          void *user_ctx, void *io_ctx)

    ctypedef struct readstat_parser_t:
        pass

    readstat_parser_t *readstat_parser_init()
    void readstat_parser_free(readstat_parser_t *parser)

    readstat_error_t readstat_set_metadata_handler(readstat_parser_t *parser, readstat_metadata_handler handler)
    readstat_error_t readstat_set_note_handler(readstat_parser_t *parser, readstat_note_handler handler)
    readstat_error_t readstat_set_variable_handler(readstat_parser_t *parser, readstat_variable_handler handler)
    readstat_error_t readstat_set_fweight_handler(readstat_parser_t *parser, readstat_fweight_handler handler)
    readstat_error_t readstat_set_value_handler(readstat_parser_t *parser, readstat_value_handler handler)
    readstat_error_t readstat_set_value_label_handler(readstat_parser_t *parser, readstat_value_label_handler handler)
    readstat_error_t readstat_set_error_handler(readstat_parser_t *parser, readstat_error_handler handler)
    readstat_error_t readstat_set_progress_handler(readstat_parser_t *parser, readstat_progress_handler handler)

    readstat_error_t readstat_set_open_handler(readstat_parser_t *parser, readstat_open_handler handler)
    readstat_error_t readstat_set_close_handler(readstat_parser_t *parser, readstat_close_handler handler)
    readstat_error_t readstat_set_seek_handler(readstat_parser_t *parser, readstat_seek_handler handler)
    readstat_error_t readstat_set_read_handler(readstat_parser_t *parser, readstat_read_handler handler)
    readstat_error_t readstat_set_update_handler(readstat_parser_t *parser, readstat_update_handler handler)
    readstat_error_t readstat_set_io_ctx(readstat_parser_t *parser, void *io_ctx)

    readstat_error_t readstat_set_file_character_encoding(readstat_parser_t *parser, const char *encoding)
    readstat_error_t readstat_set_handler_character_encoding(readstat_parser_t *parser, const char *encoding)
    readstat_error_t readstat_set_row_limit(readstat_parser_t *parser, long row_limit)
    readstat_error_t readstat_set_row_offset(readstat_parser_t *parser, long row_offset)

    readstat_error_t readstat_parse_dta(readstat_parser_t *parser, const char *path, void *user_ctx)
    readstat_error_t readstat_parse_sav(readstat_parser_t *parser, const char *path, void *user_ctx)

    # ------------------------------------------------------------------
    # Writer API
    # ------------------------------------------------------------------

    ctypedef struct readstat_label_set_t:
        pass

    ctypedef struct readstat_writer_t:
        pass

    # Receives the bytes ReadStat produces; return the number written or -1.
    ctypedef ssize_t (*readstat_data_writer)(const void *data, size_t len, void *ctx)

    readstat_writer_t *readstat_writer_init()
    void readstat_writer_free(readstat_writer_t *writer)
    readstat_error_t readstat_set_data_writer(readstat_writer_t *writer, readstat_data_writer data_writer)

    readstat_label_set_t *readstat_add_label_set(readstat_writer_t *writer, readstat_type_t type, const char *name)
    void readstat_label_double_value(readstat_label_set_t *label_set, double value, const char *label)
    void readstat_label_int32_value(readstat_label_set_t *label_set, int32_t value, const char *label)
    void readstat_label_string_value(readstat_label_set_t *label_set, const char *value, const char *label)
    void readstat_label_tagged_value(readstat_label_set_t *label_set, char tag, const char *label)

    readstat_variable_t *readstat_add_variable(readstat_writer_t *writer, const char *name, readstat_type_t type,
                                               size_t storage_width)
    void readstat_variable_set_label(readstat_variable_t *variable, const char *label)
    void readstat_variable_set_format(readstat_variable_t *variable, const char *format)
    void readstat_variable_set_label_set(readstat_variable_t *variable, readstat_label_set_t *label_set)
    void readstat_variable_set_measure(readstat_variable_t *variable, readstat_measure_t measure)
    void readstat_variable_set_alignment(readstat_variable_t *variable, readstat_alignment_t alignment)
    void readstat_variable_set_display_width(readstat_variable_t *variable, int display_width)
    readstat_error_t readstat_variable_add_missing_double_value(readstat_variable_t *variable, double value)
    readstat_error_t readstat_variable_add_missing_double_range(readstat_variable_t *variable, double lo, double hi)
    readstat_error_t readstat_variable_add_missing_string_value(readstat_variable_t *variable, const char *value)
    readstat_error_t readstat_variable_add_missing_string_range(readstat_variable_t *variable, const char *lo,
                                                                const char *hi)

    void readstat_add_note(readstat_writer_t *writer, const char *note)
    readstat_error_t readstat_writer_set_file_label(readstat_writer_t *writer, const char *file_label)
    readstat_error_t readstat_writer_set_file_timestamp(readstat_writer_t *writer, time_t timestamp)
    readstat_error_t readstat_writer_set_file_format_version(readstat_writer_t *writer, uint8_t file_format_version)
    readstat_error_t readstat_writer_set_compression(readstat_writer_t *writer, readstat_compress_t compression)

    readstat_error_t readstat_begin_writing_dta(readstat_writer_t *writer, void *user_ctx, long row_count)
    readstat_error_t readstat_begin_writing_sav(readstat_writer_t *writer, void *user_ctx, long row_count)

    readstat_error_t readstat_begin_row(readstat_writer_t *writer)
    readstat_error_t readstat_insert_int8_value(readstat_writer_t *writer, const readstat_variable_t *variable, int8_t value)
    readstat_error_t readstat_insert_int16_value(readstat_writer_t *writer, const readstat_variable_t *variable, int16_t value)
    readstat_error_t readstat_insert_int32_value(readstat_writer_t *writer, const readstat_variable_t *variable, int32_t value)
    readstat_error_t readstat_insert_float_value(readstat_writer_t *writer, const readstat_variable_t *variable, float value)
    readstat_error_t readstat_insert_double_value(readstat_writer_t *writer, const readstat_variable_t *variable, double value)
    readstat_error_t readstat_insert_string_value(readstat_writer_t *writer, const readstat_variable_t *variable, const char *value)
    readstat_error_t readstat_insert_missing_value(readstat_writer_t *writer, const readstat_variable_t *variable)
    readstat_error_t readstat_insert_tagged_missing_value(readstat_writer_t *writer, const readstat_variable_t *variable, char tag)
    readstat_error_t readstat_end_row(readstat_writer_t *writer)
    readstat_error_t readstat_end_writing(readstat_writer_t *writer)
