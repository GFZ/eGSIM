"""
flatfile columns module. For clarity, it is recommended to import this
module as `from egsim.flatfile import columns` and call its functions via
`columns.<function>`
"""
from __future__ import annotations

from os.path import join, dirname
from datetime import datetime

from pandas.core.base import IndexOpsMixin
from pandas.errors import ParserError
from typing import Union, Any, Optional
from enum import Enum

import numpy as np
import pandas as pd

from yaml import load as yaml_load
try:
    from yaml import CSafeLoader as SafeLoader  # faster, if available
except ImportError:
    from yaml import SafeLoader  # same as using yaml.safe_load

from egsim.smtk.registry import sa_period


class Type(Enum):
    """Flatfile column types"""
    RUPTURE = 'rupture_parameter'
    SITE = 'site_parameter'
    DISTANCE = 'distance_measure'
    INTENSITY = 'intensity_measure'


class Dtype(Enum):
    """Enum where members are registered column dtype"""
    # for ref `numpy.sctypeDict.keys()` lists the possible numpy values
    FLOAT = "numeric float"
    INT = "numeric integer"
    BOOL = "boolean"
    DATETIME = "ISO formatted date-time"
    STR = "string of text"
    CATEGORY = "categorical"


def is_valid(column: str) -> bool:
    """
    Return whether the given argument is a valid registered flatfile column name
    (including aliases. 'SA(<period>)' is valid and will default to 'SA')
    """
    return bool(_props_of(column))


def get_names(type: Optional[Type] = None) -> set[str]:  # noqa
    """
    Return a set of strings with all column names matching the given type. If the
    argument is missing or None, all registered column names are returned. Aliases
    are not included, only primary names
    """
    _data = _load_flatfile_columns_registry()
    return {
        c_name for c_name, c_props in _data.items()
        if type is None or c_props.get('type', None) == type
    }


def get_type(column: str) -> Union[Type, None]:
    """
    Return the `Type` enum item of the given column, or None.
    if `column` is 'SA(<period>)', it will default to 'SA'
    """
    return _props_of(column).get('type', None)


def get_default(column: str) -> Union[None, Any]:
    """
    Return the default of the given column name (used to fill missing data), or
    None if no default is set. if `column` is 'SA(<period>)', it will default to 'SA'
    """
    return _props_of(column).get('default', None)


def get_aliases(column: str) -> tuple[str]:
    """
    Return all possible names of the given column, as tuple set of strings
    where the first element is assured to be the flatfile default column name
    (primary name) and all remaining the secondary names. The tuple will be composed
    of `column` alone if `column` does not have registered aliases.
    if `column` is 'SA(<period>)', it will default to 'SA'
    """
    return _props_of(column).get('alias', (column,))


def get_help(column: str) -> str:
    """
    Return the help (description) of the given column name, or ''.
    If `column` is 'SA(<period>)', it will default to 'SA'
    """
    return _props_of(column).get('help', "")


def get_dtype(column: str) -> Union[Dtype, None]:
    """
    Return the data type of the given column name, as `Dtype` Enum item,
    or None if the column has no known data type. If the return value is
    `Dtype.CATEGORY`, get more info via `get_categorical_dtype(column)`.
    If `column` is 'SA(<period>)', it will default to 'SA'
    """
    dtype = _get_dtype(column)
    if isinstance(dtype, pd.CategoricalDtype):
        return Dtype.CATEGORY
    return dtype


def get_categorical_dtype(column: str) -> Union[pd.CategoricalDtype, None]:
    """
    Return the pandas CategoricalDtype, a data type for categorical data, for
    the given  To get the possible categories, use the `.categories` attribute
    of the returned object. Return None if the column data type is not categorical.
    If `column` is 'SA(<period>)', it will default to 'SA'
    """
    dtype = _get_dtype(column)
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype
    return None


def _get_dtype(column: str) -> Union[pd.CategoricalDtype, Dtype, None]:
    return _props_of(column).get('dtype', None)


def _props_of(column: str) -> dict:
    _data = _load_flatfile_columns_registry()
    props = _data.get(column, None)
    if props is None and sa_period(column) is not None:
        props = _data.get('SA', None)
    return props or {}


def get_dtype_of(
        obj: Union[IndexOpsMixin, np.dtype, str, float, int, datetime, bool]
) -> Union[Dtype, None]:
    """
    Get the dtype of the given pandas array, dtype or Python scalar. If
    `Dtype.CATEGORY` is returned, then `obj.dtype` is assured to be
    a pandas `CategoricalDtype` object with all categories the same dtype,
    which can be retrieved by calling again: `get_dtype_of(obj.dtype.categories)`

    Examples:
    ```
    get_dtype_of(pd.Series(...))
    get_dtype_of(pd.CategoricalDtype())
    get_dtype_of(pd.CategoricalDtype().categories)
    get_dtype_of(dataframe.index)
    get_dtype_of(datetime.utcnow())
    get_dtype_of(5.5)
    ```

    :param obj: any pandas numpy collection e.g. Series / Index, or a
        recognized Python scalar (float, int, bool, str)
    """
    # check bool / int / float in this order to avoid potential subclass issues:
    if pd.api.types.is_bool_dtype(obj) or isinstance(obj, bool):
        # pd.Series([True, False]).astype('category') ends being here, but
        # we should return categorical dtype (this happens only with booleans):
        if isinstance(getattr(obj, 'dtype', None), pd.CategoricalDtype):
            return Dtype.CATEGORY
        return Dtype.BOOL
    if pd.api.types.is_integer_dtype(obj) or isinstance(obj, int):
        return Dtype.INT
    if pd.api.types.is_float_dtype(obj) or isinstance(obj, float):
        return Dtype.FLOAT
    if pd.api.types.is_datetime64_any_dtype(obj) or isinstance(obj, datetime):
        return Dtype.DATETIME
    if isinstance(obj, pd.CategoricalDtype):
        return Dtype.CATEGORY
    if isinstance(getattr(obj, 'dtype', None), pd.CategoricalDtype):
        if get_dtype_of(obj.dtype.categories) is None:  # noqa
            return None  # mixed categories , return no dtype
        return Dtype.CATEGORY
    if pd.api.types.is_string_dtype(obj) or isinstance(obj, str):
        return Dtype.STR

    # Final check for data with str and Nones, whose dtype (np.dtype('O')) equals the
    # dtype of only-string Series, but for which `pd.api.types.is_string_dtype` is False:
    obj_dtype = None
    if getattr(obj, 'dtype', None) == np.dtype('O') and pd.api.types.is_list_like(obj):
        # check element-wise (very inefficient but unavoidable). Return Dtype.STR
        # if at least 1 element is str and all others are None:
        for item in obj:
            if item is None:
                continue
            if not pd.api.types.is_string_dtype(item):
                return None
            obj_dtype = Dtype.STR

    return obj_dtype


def cast_to_dtype(
        value: Any,
        dtype: Union[Dtype, pd.CategoricalDtype],
        mixed_dtype_categorical='raise'
) -> Any:
    """Cast the given value to the given dtype, raise ValueError if unsuccessful

    :param value: pandas Series/Index or Python scalar
    :param dtype: the base `Dtype`, or a pandas CategoricalDtype. In the latter
        case, if categories are not of the same dtype, a `ValueError` is raised
    :param mixed_dtype_categorical: what to do when `dtype=Dtype.CATEGORY`, i.e.,
        with the categories to be inferred and not explicitly given, and `value`
        (if array-like) contains mixed dtypes (e.g. float and strings).
        Then pass None to ignore and return `value` as it is, 'raise'
        (the default) to raise ValueError, and 'coerce' to cast all items to string
    """
    categories: Union[pd.CategoricalDtype, None] = None
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype
        dtype = get_dtype_of(dtype.categories.dtype)

    actual_base_dtype = get_dtype_of(value)
    actual_categories: Union[pd.CategoricalDtype, None] = None
    if isinstance(getattr(value, 'dtype', None), pd.CategoricalDtype):
        actual_categories = value.dtype
        actual_base_dtype = get_dtype_of(value.dtype.categories)

    if dtype is None:
        raise ValueError('cannot cast column to dtype None. If you passed '
                         'categorical dtype, check that all categories '
                         'dtypes are equal')

    if categories and actual_categories:
        if set(categories.categories) != set(actual_categories.categories):
            raise ValueError('Value mismatch with provided categorical values')

    is_pd = isinstance(value, IndexOpsMixin)  # common interface for Series / Index
    if not is_pd:
        values = pd.Series(value)
    else:
        values = value

    if dtype != actual_base_dtype:
        if dtype == Dtype.FLOAT:
            values = values.astype(float)
        elif dtype == Dtype.INT:
            values = values.astype(int)
        elif dtype == Dtype.BOOL:
            # bool is too relaxed, e.g. [3, 'x'].astype(bool) works but it shouldn't
            values = values.astype(int)
            if set(pd.unique(values)) - {0, 1}:
                raise ValueError('not a boolean')
            values = values.astype(bool)
        elif dtype == Dtype.DATETIME:
            try:
                values = pd.to_datetime(values)
            except (ParserError, ValueError) as exc:
                raise ValueError(str(exc))
        elif dtype == Dtype.STR:
            values = values.astype(str)
        elif dtype == Dtype.CATEGORY:
            values_ = values.astype('category')
            mixed_dtype = get_dtype_of(values_.cat.categories) is None
            if mixed_dtype and mixed_dtype_categorical == 'raise':
                raise ValueError('mixed dtype in categories')
            elif mixed_dtype and mixed_dtype_categorical == 'coerce':
                values = values.astype(str).astype('category')
            else:
                values = values_

    if categories is not None:
        # pandas converts invalid values (not in categories) to NA, we want to raise:
        is_na = pd.isna(values)
        values = values.astype(categories)
        is_na_after = pd.isna(values)
        ok = len(is_na) == len(is_na_after) and (is_na == is_na_after).all()  # noqa
        if not ok:
            raise ValueError('Value mismatch with provided categorical values')

    if is_pd:
        return values
    if pd.api.types.is_list_like(value):
        # for ref, although passing numpy arrays is not explicitly supported, numpy
        # arrays with no shape (e.g.np.array(5)) are *not* falling in this if
        return values.to_list()
    return values.item()


# YAML file path:
_flatfile_columns_path = join(dirname(__file__), 'columns_registry.yaml')
_flatfile_columns_yaml_cache: Optional[dict] = None


def _load_flatfile_columns_registry() -> dict[str, dict[str, Any]]:
    """Load the columns registry from the associated YAML file into a Python dict"""
    global _flatfile_columns_yaml_cache
    if _flatfile_columns_yaml_cache is None:
        _flatfile_columns_yaml_cache = {}
        with open(_flatfile_columns_path) as fpt:
            for col_name, props in yaml_load(fpt, SafeLoader).items():
                props = _harmonize_col_props(col_name, props)
                # add all aliases mapped to the relative properties:
                for c_name in props['alias']:
                    _flatfile_columns_yaml_cache[c_name] = props
    return _flatfile_columns_yaml_cache


def _harmonize_col_props(name: str, props: dict):
    """Harmonize the values of a column property dict"""
    aliases = props.get('alias', [])
    if isinstance(aliases, str):
        aliases = [aliases]
    props['alias'] = (name,) + tuple(aliases)
    if 'type' in props:
        props['type'] = Type[props['type'].upper()]
    dtype: Union[None, Dtype, pd.CategoricalDtype] = None
    if 'dtype' in props:
        if isinstance(props['dtype'], str):
            props['dtype'] = dtype = Dtype[props['dtype'].upper()]
        else:
            props['dtype'] = dtype = pd.CategoricalDtype(props['dtype'])
    if 'default' in props:
        props['default'] = cast_to_dtype(props['default'], dtype)
    for k in ("<", "<=", ">", ">="):
        if k in props:
            props[k] = cast_to_dtype(props[k], dtype)
    return props


