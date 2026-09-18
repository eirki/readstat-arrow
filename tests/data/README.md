# Test data

All files are copied from the [pyreadstat](https://github.com/Roche/pyreadstat)
test suite (`test_data/`), © 2018 Hoffmann-La Roche, licensed under the
Apache License 2.0.

## sample.* (basic round trips)

`sample.csv` holds the expected contents. Both files hold the same 5 rows × 7 columns:

| column | type   | notes                                   |
|--------|--------|-----------------------------------------|
| mychar | string |                                         |
| mynum  | double |                                         |
| mydate | date   | last row missing                        |
| dtime  | datetime | last row missing                      |
| mylabl | numeric | value labels 1 = Male, 2 = Female; int8 in .dta |
| myord  | numeric | value labels 1 = low, 2 = medium, 3 = high; int8 in .dta |
| mytime | time   | last row missing                        |

## Edge cases

| file | exercises |
|------|-----------|
| `sample_missing.sav` | SPSS user-defined missing values: discrete values and a range |
| `missing_char.sav` | a *string* value declared missing (`'Z'`), plus a labelled string value |
| `missing_test.dta` | Stata tagged missings `.a`–`.z` as the only content; a label on tag `a` |
| `simple_alltypes.sav` | missing ranges with negative bounds, labelled missing values, multiple-response sets, `QYR` format |
| `test_width.sav` | very long strings (`A1024`, stored in segments) and 8-byte-padded widths |
| `tegulu.sav` | UTF-8 string values (Telugu) |
| `hebrews.sav` | a non-ASCII variable name |
