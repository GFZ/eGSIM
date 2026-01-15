"""flatfile root module"""
from __future__ import annotations

from io import IOBase, StringIO
from datetime import datetime
import re
from typing import Union, Any
import tokenize

from pandas.errors import ParserError
from tables import HDF5ExtError
import pandas as pd

from ..validation import InputError, ConflictError
from . import columns


_csv_default_args = (
    ('na_values', ("", "null", "NULL", "None",
                   "nan", "-nan", "NaN", "-NaN",
                   "NA", "N/A", "n/a",  "<NA>", "#N/A", "#NA")),
    ('keep_default_na', False),
    ('comment', '#'),
    ('encoding', 'utf-8-sig')
)


EVENT_ID_COLUMN_NAME = 'evt_id'


def read_flatfile(
        filepath_or_buffer: Union[str, IOBase],
        rename: dict[str, str] = None,
        dtypes: dict[str, Union[str, list]] = None,
        defaults: dict[str, Any] = None,
        csv_sep: str = None,
        **kwargs) -> pd.DataFrame:
    """
    Read a flatfile from either a comma-separated values (CSV) or HDF file,
    returning the corresponding pandas DataFrame.

    :param filepath_or_buffer: str, path object or file-like object of the data.
        HDF files are the recommended formats **but support only files on-disk as
        parameter**. CSV files on the other hand can be supplied as in-memory stream, or
        compressed files that will be inferred from the extension (e.g. 'gzip', 'zip')
    :param rename: a dict mapping a file column to a new column name. Mostly useful
        for renaming columns to standard flatfile names, delegating all data types
        check to the function without (see also dtypes and defaults for info)
    :param dtypes: dict of file column names mapped to user-defined data types, to
        check and cast column data. Standard flatfile columns should not be present,
        otherwise the value provided in this dict will overwrite the registered dtype,
        if set. Columns in `dtypes` not present in the file will be ignored.
        Dict values can be either 'int', 'bool', 'float', 'str', 'datetime', 'category'`,
        list: 'category' and lists denote data that can take only a limited amount of
        possible values and should be mostly used with string data for saving space
        (with "category", pandas will infer the possible values from the data. In this
        case, note that with CSV files each category will be of type `str`).
    :param defaults: dict of file column names mapped to user-defined default to
        replace missing values. Because 'int' and 'bool' columns do not support missing
        values, with CSV files a default should be provided (e.g. 0 or False) to avoid
        raising Exceptions.
        Standard flatfile columns do not need to be present. If they are, the value here
        will overwrite the default dtype, if set. Columns in `defaults` not present in
        the file will be ignored
    :param csv_sep: the separator (or delimiter), only used for CSV files.
        None means 'infer' (look in `kwargs` and if not found, infer from data header)

    :return: pandas DataFrame representing a Flat file
    """
    is_hdf = False
    cur_pos = None
    if isinstance(filepath_or_buffer, IOBase):
        cur_pos = filepath_or_buffer.tell()

    try:
        dfr = pd.read_hdf(filepath_or_buffer, **kwargs)
        is_hdf = True
    except (HDF5ExtError, NotImplementedError):
        import traceback
        # HdfError -> some error in the data
        # NotImplementedError -> generic buffer not implemented in HDF
        # try with CSV:
        if cur_pos is not None:
            filepath_or_buffer.seek(cur_pos)

        kwargs = dict(_csv_default_args) | kwargs
        if csv_sep is None:
            kwargs['sep'] = _infer_csv_sep(filepath_or_buffer, **kwargs)
        else:
            kwargs['sep'] = csv_sep

        # harmonize dtypes with only Column.Dtype enums or pd.,CategoricalDtype objects:
        # also put in kwargs['dtype'] the associated dtypes compatible with `read_csv`:
        kwargs['dtype'] = kwargs.get('dtype') or {}
        dtypes_raw = dtypes or {}
        dtypes = {}
        for c, v in dtypes_raw.items():
            if not isinstance(v, str):
                try:
                    v = pd.CategoricalDtype(v)
                    assert columns.get_dtype_of(v.categories) is not None
                except (AssertionError, TypeError, ValueError):
                    raise ValueError(f'{c}: categories must be of the same type')
            else:
                try:
                    v = columns.Dtype[v]
                except KeyError:
                    raise ValueError(f'{c}: invalid dtype {v}')
            dtypes[c] = v
            # ignore bool int and date-times, we will parse them later
            if v in (columns.Dtype.BOOL, columns.Dtype.INT, columns.Dtype.DATETIME):
                continue
            kwargs['dtype'][c] = v.name if isinstance(v, Column.Dtype) else v  # noqa

        try:
            dfr = pd.read_csv(filepath_or_buffer, **kwargs)
        except ValueError as exc:
            # invalid_columns = _read_csv_inspect_failure(filepath_or_buffer, **kwargs)
            raise ColumnDataError(str(exc)) from None

    if rename:
        dfr.rename(columns=rename, inplace=True)
        # rename defaults and dtypes:
        for old, new in rename.items():
            if dtypes and old in dtypes:
                dtypes[new] = dtypes.pop(old)
            if defaults and old in defaults:
                defaults[new] = dtypes.pop(old)

    validate_flatfile_dataframe(dfr, dtypes, defaults, 'raise' if is_hdf else 'coerce')
    optimize_flatfile_dataframe(dfr)
    if not isinstance(dfr.index, pd.RangeIndex):
        dfr.reset_index(drop=True, inplace=True)
    return dfr


def _infer_csv_sep(filepath_or_buffer: IOBase, **kwargs) -> str:
    """Infer `sep` from kwargs, and or return it"""
    sep = kwargs.get('sep')
    if sep is not None:
        return sep
    nrows = kwargs.pop('nrows', None)
    header = []
    for _sep in [';', ',', r'\s+']:
        _header = _read_csv_get_header(filepath_or_buffer, sep=_sep, **kwargs)
        if len(_header) > len(header):
            sep = _sep
            header = _header
    if nrows is not None:
        kwargs['nrows'] = nrows
    return sep


def _read_csv_get_header(filepath_or_buffer: IOBase, sep=None, **kwargs) -> list[str]:
    _pos = None
    if isinstance(filepath_or_buffer, IOBase):
        _pos = filepath_or_buffer.tell()
    # use only args necessary to parse columns, we might raise unnecessary errors
    # otherwise (these errors might be fixed afterward before reading the whole csv):
    args = ['header', 'names', 'skip_blank_lines', 'skipinitialspace', 'engine',
            'lineterminator', 'quotechar', 'quoting', 'doublequote', 'escapechar',
            'comment', 'dialect', 'delim_whitespace']
    _kwargs = {k: kwargs[k] for k in args if k in kwargs}
    _kwargs['nrows'] = 0  # read just header
    _kwargs['sep'] = sep
    ret = pd.read_csv(filepath_or_buffer, **_kwargs).columns  # noqa
    if _pos is not None:
        filepath_or_buffer.seek(_pos)
    return ret


def validate_flatfile_dataframe(
        dfr: pd.DataFrame,
        extra_dtypes: dict[str, Union[Column.Dtype, pd.CategoricalDtype]] = None,  # noqa
        extra_defaults: dict[str, Any] = None,
        mixed_dtype_categorical='raise'):
    """
    Validate the flatfile dataframe checking data types, conflicting column names,
    or missing mandatory columns (e.g. IMT related columns). This method raises
    or returns None on success

    :param dfr: the flatfile, as pandas DataFrame
    :param extra_dtypes: dict of column names mapped to the desired data type.
        Standard flatfile columns should not to be present (unless for some reason
        their dtype must be overwritten). pd.CategoricalDtype categories must be
        all the same type (this is supposed to have been checked beforehand)
    :param extra_defaults: dict of column names mapped to the desired default value
        to replace missing data. Standard flatfile columns do not need to be present
        (unless for some reason their dtype must be overwritten)
    :param mixed_dtype_categorical: what to do when `dtype=Column.Dtype.CATEGORY`, i.e.,
        with the categories to be inferred and not explicitly given, and `value`
        (if array-like) contains mixed dtypes (e.g. float and strings).
        Then pass None to ignore and return `value` as it is, 'raise'
        (the default) to raise ValueError, and 'coerce' to cast all items to string
    """
    # post-process:
    invalid_columns = []
    if not extra_defaults:
        extra_defaults = {}
    if not extra_dtypes:
        extra_dtypes = {}
    # check dtypes correctness (actual vs expected) and try to fix mismatching ones:
    for col in dfr.columns:
        if col in extra_dtypes:
            xp_dtype = extra_dtypes[col]
        else:
            xp_dtype = columns.get_dtype(col)
            if xp_dtype == columns.Dtype.CATEGORY:
                xp_dtype = columns.get_categorical_dtype(col)

        if xp_dtype is None:
            continue

        if col in extra_dtypes:
            default = extra_defaults.get(col)
            if default is not None:
                default = columns.cast_to_dtype(
                    default, xp_dtype, mixed_dtype_categorical
                )
        else:
            default = columns.get_default(col)
        if default is not None:
            is_na = pd.isna(dfr[col])
            dfr.loc[is_na, col] = default

        # cast to expected dtype (no op if dtype is already ok):
        try:
            dfr[col] = columns.cast_to_dtype(
                dfr[col], xp_dtype, mixed_dtype_categorical
            )
        except (ParserError, ValueError):
            invalid_columns.append(col)
            continue

    if invalid_columns:
        raise ColumnDataError(*invalid_columns)

    # check no dupes:
    ff_cols = set(dfr.columns)
    has_imt = False
    for c in dfr.columns:
        aliases = set(columns.get_aliases(c))
        if len(aliases & ff_cols) > 1:
            raise IncompatibleColumnError(list(aliases & ff_cols))
        if not has_imt and columns.get_type(c) == columns.Type.INTENSITY:
            has_imt = True

    if not has_imt:
        raise MissingColumnError('No IMT column found')

    return dfr


def optimize_flatfile_dataframe(dfr: pd.DataFrame):
    """Optimize the given dataframe by replacing str column with
    categorical (if the conversion saves memory)
    """
    for c in dfr.columns:
        if columns.get_dtype_of(dfr[c]) == columns.Dtype.STR:
            cat_dtype = dfr[c].astype('category')
            if cat_dtype.memory_usage(deep=True) < dfr[c].memory_usage(deep=True):
                dfr[c] = cat_dtype


class FlatfileError(InputError):
    """Subclass of :class:`smtk.validators.InputError` for describing flatfile
    errors (specifically, column errors). See subclasses for details. Remember
    that `str(FlatfileError(arg1, arg2, ...)) = str(arg1) + ", " + str(arg2) + ...
    """
    pass


class MissingColumnError(FlatfileError, AttributeError, KeyError):
    """MissingColumnError. It inherits also from AttributeError and
    KeyError to be compliant with pandas and OpenQuake"""
    pass


class IncompatibleColumnError(ConflictError, FlatfileError):
    pass


class ColumnDataError(FlatfileError, ValueError, TypeError):
    pass


class FlatfileQueryError(FlatfileError):
    """Error while filtering flatfile rows via query expressions"""
    pass


def query(flatfile: pd.DataFrame, query_expression: str, raise_no_rows=True) \
        -> pd.DataFrame:
    """
    Call `flatfile.query` with some utilities:
     - ISO-861 strings (e.g. "2006-01-31") will be converted to datetime objects
     - booleans can be also lower case (true or false)
     - Some series methods can be called with the dot notation [col].[method]:
       notna(), median(), mean(), min(), max()
    """
    # Setup custom keyword arguments to dataframe query
    __kwargs = {
        'local_dict': {},
        'global_dict': {},  # 'pd': pd, 'np': np
        # add support for bools lower case (why doesn't work if set in local_dict?):
        'resolvers': [prepare_expr(query_expression, flatfile.columns)]
    }
    # evaluate expression:
    try:
        ret = flatfile.query(query_expression, **__kwargs)
    except Exception as exc:
        raise FlatfileQueryError(str(exc)) from None
    if raise_no_rows and ret.empty:
        raise FlatfileQueryError('no rows matching query')
    return ret


def prepare_expr(expr: str, columns: list[str]) -> dict:
    """
    Prepare the given selection expression to add a layer of protection
    to potentially dangerous untrusted input. Returns a dict to be used as
    `resolvers` argument in `pd.Dataframe.query`
    """
    # valid names:
    cols = set(columns)
    # backtick-quoted column names is pandas syntax. Check them now and replace them
    # with a valid name (replacement_col):
    replacement_col = '_'
    while replacement_col in columns:
        replacement_col += '_'
    new_expr = re.sub(f'`.*?`', replacement_col, expr)
    if new_expr != expr:
        # check that columns are correct
        cols.add(replacement_col)
        for match in re.finditer(f'`.*?`', expr):
            if match.group()[1:-1] not in columns:
                raise FlatfileQueryError(f'undefined column "{match.group()[1:-1]}"')

    meth_placeholder = replacement_col + '_'
    while meth_placeholder in cols:
        meth_placeholder += '_'
    new_expr = re.sub(r'\.\s*(notna|mean|std|median|min|max)\s*\(\s*\)',
                      f' {meth_placeholder} ',
                      new_expr)

    # analyze string and return the replacements to be done:
    replacements = {}
    toknums = []
    tokvals = []  # either str, or the int tokenize.NEWLINE
    # allowed sequences:
    allowed_names = cols | {'true', 'True', 'false', 'False'}
    token_generator = tokenize.generate_tokens(StringIO(new_expr).readline)
    for toknum, tokval, start, _, _ in token_generator:
        last_toknum = toknums[-1] if len(toknums) else None
        last_tokval = tokvals[-1] if len(tokvals) else None

        if toknum == tokenize.NAME:
            if tokval not in cols:
                if tokval == 'true':
                    replacements[tokval] = True
                elif tokval == 'false':
                    replacements[tokval] = False
        elif toknum == tokenize.STRING:
            try:
                dtime = datetime.fromisoformat(tokval[1:-1])
                replacements[tokval] = dtime
            except (TypeError, ValueError):
                pass

        skip_check_sequence = tokval == meth_placeholder and last_tokval in cols
        if not skip_check_sequence:
            if toknum == tokenize.NAME and tokval not in allowed_names:
                raise FlatfileQueryError(f'undefined column "{tokval}"')
            if not valid_expr_sequence(last_toknum, last_tokval, toknum, tokval):
                if last_toknum is None:
                    raise FlatfileQueryError(f'invalid first chunk {tokval}')
                elif last_toknum == tokenize.NEWLINE:
                    raise FlatfileQueryError(f'multi-line expression not allowed')
                elif toknum == tokenize.NEWLINE:
                    raise FlatfileQueryError(f'invalid last chunk "{last_tokval}"')
                else:
                    raise FlatfileQueryError(f'invalid sequence '
                                             f'"{last_tokval}" + "{tokval}"')

        tokvals.append(tokval)
        toknums.append(toknum)

    return replacements


def valid_expr_sequence(tok_num1: int, tok_val1: str, tok_num2: int, tok_val2: str):
    """Return true if the given sequence of two tokens is valid"""
    OP, STRING, NAME, NUMBER, NEWLINE, EOM = (tokenize.OP, tokenize.STRING,  # noqa
                                              tokenize.NAME, tokenize.NUMBER,
                                              tokenize.NEWLINE, tokenize.ENDMARKER)
    if tok_num1 is None:
        if tok_num2 == OP:
            return tok_val2 in '(~'
        return tok_num2 in {NAME, NUMBER, STRING}
    elif tok_num1 == NEWLINE:
        return tok_num2 == EOM
    elif tok_num2 == NEWLINE:
        return tok_num1 in {NUMBER, NAME, STRING} or tok_val1 == ')'
    elif (tok_num1, tok_num2) == (OP, OP):
        return (tok_val1 + tok_val2 in
                {'(~', '~(', '((', '~~', '))', '&(', '|(', ')|', ')&'})
    elif (tok_num1, tok_num2) == (NAME, OP):
        return tok_val2 in {'==', '!=', '<', '<=', '>', '>=', ')', '+', '-', '*', '/'}
    elif (tok_num1, tok_num2) == (OP, NAME):
        return tok_val1 in {'==', '!=', '<', '<=', '>', '>=', '(', '+', '-', '*', '/'}
    elif (tok_num1, tok_num2) == (NUMBER, OP):
        return tok_val2 in {'==', '!=', '<', '<=', '>', '>=', ')', '+', '-', '*', '/'}
    elif (tok_num1, tok_num2) == (OP, NUMBER):
        return tok_val1 in {'==', '!=', '<', '<=', '>', '>=', '(', '+', '-', '*', '/'}
    elif (tok_num1, tok_num2) == (STRING, OP):
        return tok_val2 in {'==', '!=', '<', '<=', '>', '>=', ')'}
    elif (tok_num1, tok_num2) == (OP, STRING):
        return tok_val1 in {'==', '!=', '<', '<=', '>', '>=', '('}
    else:
        return False
