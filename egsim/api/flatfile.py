import re
from typing import Union, Callable, Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from openquake.hazardlib import imt
from smtk.sm_utils import DEFAULT_MSR
from smtk.residuals import context_db
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14


############
# Flatfile #
############

def read_flatfile(filepath_or_buffer: str,
                  sep: str = None,
                  col_mapping: dict[str, str] = None,
                  usecols: Union[list[str], Callable[[str], bool]] = None,
                  dtype: dict[str, Union[str, list, tuple]] = None,
                  defaults: dict[str, Any] = None,
                  **kwargs):
    """
    Read a flat file into pandas DataFrame from a given CSV file

    :param filepath_or_buffer: str, path object or file-like object of the CSV
        formatted data. If string, compressed files are also supported
        and inferred from the extension (e.g. 'gzip', 'zip')
    :param sep: the separator (or delimiter). None means 'infer' (it might
        take more time)
    :param col_mapping: dict mapping CSV column names to Flat file column names.
        CSV column names not found in the dict keys are left unchanged.
        **CSV COLUMNS WILL BE RENAMED AS FIRST STEP**: when not otherwise
        indicated, all arguments below will work on flat file column names, i.e.
        the CSV columns renamed via this mapping
    :param dtype: dict of *flat file column names* mapped to the data type:
        either 'int', 'bool', 'float', 'str', 'datetime', 'category'` or list/tuple.
        'category', list or tuples are for data that can take only a limited amount
        of possible values and should be used mostly with string data as it might
        save a lot of memory. With "category", pandas will infer the number of
        categories from the data, whereas a list/tuple defines the possible
        categories, if known beforehand: in this case data values not found
        are converted to missing values (NA) and then replaced by a default, if
        set in `defaults` for the given column.
        Columns of type 'int' and 'bool' do not support NA data and must have a
        default in `defaults`, otherwise NA data will be replaced with 0 for int
        and False for bool.
        `dtype`=None (the default) means that pandas will try to infer the data
        type of each column (see `read_csv` documentation and `na_values` to see
        what it's considered NA).
    :param usecols: flat file column names to load, as list or callable accepting
        a flat file column name and returning True/False
    :param defaults: a dict of flat file column names mapped to the default
        value for missing/NA data. Defaults will be set AFTER the underlying
        `pandas.read_csv` is called, on the returned dataframe before returning
        it. None means: do not replace any NA data. Note however that if int and
        bool columns are specified in `dtype`, then a default is set for those
        columns anyway (0 for int, False for bool), because those data types do
        not support NA in numpy/pandas
    :param kwargs: additional keyword arguments not provided above that can be
        passed to `pandas.read_csv`. 'header', 'delim_whitespace' and 'names'
        should not be provided as they might be overwritten by this function

    :return: pandas DataFrame representing a Flat file
    """
    kwargs['sep'] = sep
    kwargs.setdefault('encoding', 'utf-8-sig')

    # CSV columns can be renamed via `read_csv(..., names=[...], header=0, ...)`
    # or simply by calling afterwards `dataframe.rename(columns={...})`. Both
    # approaches have drawbacks: in the first case we need to open the file
    # twice, in the latter we have to modify all arguments, such as `dtype`,
    # working on renamed columns.
    # We opt for the former for simplicity, as we need to open the file twice
    # also in case `sep` is not given (infer CSV separator). Note that we should
    # never need to open the file twice for user uploaded flat files: `col_mapping`
    # should not be given (it makes no sense) and we could simply recommend
    # passing explicitly `sep`

    if sep is None:
        kwargs |= _infer_csv_sep(filepath_or_buffer, col_mapping is not None)
    elif col_mapping:
        kwargs['names'] = _read_csv_header(filepath_or_buffer, sep)

    if col_mapping is not None:
        kwargs['header'] = 0
        # replace names with the new names of the mapping (if found) or leave
        # the name as it is if a mapping is not found:
        kwargs['names'] = [col_mapping.get(n, n) for n in kwargs['names']]

    # initialize the defaults dict if None and needs to be populated:
    if defaults is None:
        defaults = {}

    # Check which column is an int or bool, as those columns do not support NA
    # (e.g. a empty CSV cell for a boolean column raises)
    dtype_, dtype_ib, datetime_cols = {}, {}, kwargs.pop('parse_dates', [])
    for col, dtyp in dtype.items():
        if dtyp in ('bool', 'int'):
            # Move the dtype to dtype_ib:
            dtype_ib[col] = dtyp
            # Replace dtype with float in order to safely read NA:
            dtype_[col] = 'float'
            # Check that a default is set and is of type float, otherwise pandas
            # might perform useless data conversions to object. As such, when a
            # default is unset, provide 0, as eventually float(0) = float(False)
            defaults[col] = float(defaults.get(col, 0))
        elif dtyp == 'datetime':
            datetime_cols.append(col)
        elif isinstance(dtyp, (list, tuple)):
            dtype_[col] = pd.CategoricalDtype(dtyp)  # noqa
        else:
            dtype_[col] = dtyp

    dfr = pd.read_csv(filepath_or_buffer, dtype=dtype_,
                      parse_dates=datetime_cols or None,
                      usecols=usecols, **kwargs)

    for col, def_val in defaults.items():
        if col not in dfr.columns:
            continue
        dfr.loc[dfr[col].isna(), col] = def_val
        if col in dtype_ib:
            dfr[col] = dfr[col].astype(dtype_ib[col])

    return dfr


def _infer_csv_sep(filepath_or_buffer: str, return_col_names=False) -> dict[str, Any]:
    """Prepares the CSV for reading by inspecting the header and inferring the
    separator `sep`, if the latter is None.

    :param filepath_or_buffer: str, path object or file-like object. If it has
        the attribute `seek` (e.g., file like object like `io.BytesIO`, then
        `filepath_or_buffer.seek(0)` is called before returning (reset the
        file-like object position at the start of the stream)
    :return: the arguments needed for pd.read_csv as dict (e.g. `{'sep': ','}`).
        if `return_colnames` is True, the dict also contains the key 'names'
        with the CSV column header names
    """
    params = {}
    # infer separator: pandas suggests to use the engine='python' argument,
    # but this takes approx 4.5 seconds with the ESM flatfile 2018
    # whereas the method below is around 1.5 (load headers and count).
    # So, try to read the headers only with comma and semicolon, and chose the
    # one producing more columns:
    comma_cols = _read_csv_header(filepath_or_buffer, sep=',')
    semicolon_cols = _read_csv_header(filepath_or_buffer, sep=';')
    if len(comma_cols) > 1 and len(comma_cols) >= len(semicolon_cols):
        params['sep'] = ','
        names = comma_cols.tolist()
    elif len(semicolon_cols) > 1:
        params['sep'] = ';'
        names = semicolon_cols.tolist()
    else:
        # try with spaces:
        space_cols = _read_csv_header(filepath_or_buffer, sep=r'\s+')
        if len(space_cols) > max(len(comma_cols), len(semicolon_cols)):
            params['sep'] = r'\s+'
            names = space_cols.tolist()
        else:
            raise ValueError('CSV separator could not be inferred by trying '
                             '",", ";" and "\\s+" (whitespaces)')

    if return_col_names:
        params['names'] = names

    return params


def _read_csv_header(filepath_or_buffer, sep: str, reset_if_stream=True) -> pd.Index:
    """Reads the csv header and return a pandas Index (iterable) of column names

    :param filepath_or_buffer: str, path object or file-like object. If it has
        the attribute `seek` (e.g., file like object like `io.BytesIO`, then
        `filepath_or_buffer.seek(0)` is called before returning (reset the
        file-like object position at the start of the stream)
    """
    columns = pd.read_csv(filepath_or_buffer, nrows=0, sep=sep).columns
    if reset_if_stream and hasattr(filepath_or_buffer, 'seek'):
        filepath_or_buffer.seek(0)
    return columns


############################################
# GroundMotionTable/ Residuals calculation #
############################################


class Flatfile(context_db.ContextDB):
    """This abstract-like class represents a Database (DB) of data capable of
    yielding Contexts and Observations suitable for Residual analysis (see
    argument `ctx_database` of :meth:`gmpe_residuals.Residuals.get_residuals`)

    Concrete subclasses of `ContextDB` must implement three abstract methods
    (e.g. :class:`smtk.sm_database.GroundMotionDatabase`):
     - get_event_and_records(self)
     - update_context(self, ctx, records, nodal_plane_index=1)
     - get_observations(self, imtx, records, component="Geometric")
       (which is called only if `imts` is given in :meth:`self.get_contexts`)

    Please refer to the functions docstring for further details
    """

    def __init__(self, filepath_buffer_or_dfr, **kwargs):
        """
        Initializes a new GroundMotionTable.

        :param filepath_buffer_or_dfr: str, path object, file-like object or
            pandas DataFrame.
        :param filepath: string denoting the HDF file path or a BytesIO
        """
        if isinstance(filepath_buffer_or_dfr, pd.DataFrame):
            self._data = filepath_buffer_or_dfr
        else:
            self._data = self.read_data(filepath_buffer_or_dfr, **kwargs)
        sa_columns_and_periods = ((col, imt.from_string(col.upper()).period)
                                  for col in self._data.columns
                                  if col.upper().startswith("SA("))
        sa_columns_and_periods = sorted(sa_columns_and_periods, key=lambda _: _[1])
        self.sa_columns = [_[0] for _ in sa_columns_and_periods]
        self.sa_periods = [_[1] for _ in sa_columns_and_periods]
        self.filter_expression = None

    def read_data(self, filepath_or_buffer, **kwargs):
        """
        :param filepath_or_buffer: str, path object or file-like object
        """
        raise NotImplementedError('')

    def filter(self, expression: str):
        """Filters this Flatfile via a selection expression on the columns.
        E.g.
        """
        self.filter_expression = self._format_filter_expression
        return self
        # new_dfr = self._data.query(condition)
        # return self.__class__(new_dfr)
        # gmdb = GroundMotionTable(self.filepath, self.dbname)
        # gmdb._condition = condition  # pylint: disable=protected-access
        # return gmdb

    @staticmethod
    def _format_filter_expression(filter_expression):
        """Reformat the filter expression by changing notna(column) to
        column == column, and isna(column) into column != column. Also change
        true/false to True/False. Equality must be given as ==
        """
        filter_expression = re.sub(r'isna\((.*?)\)', r"\1!=\1",
                                   filter_expression, re.IGNORECASE)
        filter_expression = re.sub(r'notna\((.*?)\)', r"\1==\1",
                                   filter_expression, re.IGNORECASE)
        filter_expression = re.sub(r'\btrue\b', "true", filter_expression,
                                   re.IGNORECASE)
        filter_expression = re.sub(r'\bfalse\b', "false", filter_expression,
                                   re.IGNORECASE)
        return filter_expression

    #########################################
    # IMPLEMENTS ContextDB ABSTRACT METHODS #
    #########################################

    def get_event_and_records(self):
        dfr = self._data
        if self.filter_expression:
            dfr = dfr.query(self.filter_expression)
        return dfr.groupby(['event_id'])

    def get_observations(self, imtx, records, component="Geometric"):

        if imtx.lower().startswith("sa("):
            sa_periods = self.sa_periods
            target_period = imt.from_string(imtx).period
            spectrum = np.log10(records[self.sa_columns])
            # we need to interpolate row wise
            # build the interpolation function
            interp = interp1d(sa_periods, spectrum, axis=1)
            # and interpolate
            values = 10 ** interp(target_period)
        else:
            try:
                values = records[imtx].values
            except KeyError:
                # imtx not found, try ignoring case: first `lower()`, more likely
                # (IMTs mainly upper case), then `upper()`, otherwise: raise
                for _ in (imtx.lower() if imtx.lower() != imtx else None,
                          imtx.upper() if imtx.upper() != imtx else None):
                    if _ is not None:
                        try:
                            values = records[_].values
                            break
                        except KeyError:
                            pass
                else:
                    raise ValueError("'%s' is not a recognized IMT" % imtx) from None
        return values.copy()

    def update_context(self, ctx, records, nodal_plane_index=1):
        self._update_rupture_context(ctx, records, nodal_plane_index)
        self._update_distances_context(ctx, records)
        self._update_sites_context(ctx, records)

    def _update_rupture_context(self, ctx, records, nodal_plane_index=1):
        """see `self.update_context`"""

        record = records.iloc[0]  # pandas Series
        ctx.mag = record['magnitude']
        ctx.strike = record['strike']
        ctx.dip = record['dip']
        ctx.rake = record['rake']
        ctx.hypo_depth = record['hypocenter_depth']

        ztor = record['depth_top_of_rupture'].values.copy()
        isna = pd.isna(ztor)
        if isna.any():
            ztor[isna] = ctx.hypo_depth[isna]
        ctx.ztor = ztor

        rup_width = record['rupture_width'].values.copy()
        isna = pd.isna(rup_width)
        if isna.any():
            rup_width[isna] = np.sqrt(DEFAULT_MSR.get_median_area(ctx.mag[isna], 0))
        ctx.width = rup_width

        ctx.hypo_lat = record['event_latitude']
        ctx.hypo_lon = record['event_longitude']
        ctx.hypo_loc = np.full((len(record), 2), 0.5)

    def _update_distances_context(self, ctx, records):
        """see `self.update_context`"""

        # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
        # is a hack! Need feedback on how to fix
        ctx.repi = records['repi']
        ctx.rhypo = records['rhypo']

        ctx.rcdpp = np.full((len(records),), 0.0)
        ctx.rvolc = np.full((len(records),), 0.0)
        ctx.azimuth = records['azimuth']

        dists = records['rjb'].values.copy()
        isna = pd.isna(dists)
        if isna.any():
            dists[isna] = records.loc[isna, 'repi']
        ctx.rjb = dists

        dists = records['rrup'].values.copy()
        isna = pd.isna(dists)
        if isna.any():
            dists[isna] = records.loc[isna, 'rhypo']
        ctx.rrup = dists

        dists = records['rx'].values.copy()
        isna = pd.isna(dists)
        if isna.any():
            dists[isna] = -records.loc[isna, 'repi']
        ctx.rx = dists

        dists = records['ry0'].values.copy()
        isna = pd.isna(dists)
        if isna.any():
            dists[isna] = records.loc[isna, 'repi']
        ctx.ry0 = dists

    def _update_sites_context(self, ctx, records):
        """see `self.update_context`"""

        ctx.lons = records['station_longitude'].values
        ctx.lats = records['station_latitude'].values
        ctx.depths = records['station_elevation'].values * -1.0E-3 if \
            'station_elevation' in records.columns else np.full((len(records),), 0.0)
        vs30 = records['vs30'].values
        ctx.vs30 = vs30
        ctx.vs30measured = records['vs30measured']
        ctx.z1pt0 = records['z1'].values if 'z1' in records.columns else \
            vs30_to_z1pt0_cy14(vs30)
        ctx.z2pt5 = records['z2pt5'].values if 'z2pt5' in records.columns else \
            vs30_to_z2pt5_cb14(vs30)
        ctx.backarc = records['backarc']


class UserFlatfile(Flatfile):

    def __init__(self, filepath_or_buffer, sep=None,
                 dtype: dict[str, Union[str, list, tuple]] = None,
                 defaults: dict[str, Any] = None):
        Flatfile. __init__(self, filepath_or_buffer, sep=sep, dtype=dtype,
                           defaults=defaults)

    def read_data(self, filepath_or_buffer, *, sep=None,
                  dtype: dict[str, Union[str, list, tuple]] = None,
                  defaults: dict[str, Any] = None):
        return read_flatfile(filepath_or_buffer, sep=sep, dtype=dtype,
                             defaults=defaults)


class PredefinedFlatfile(Flatfile):

    def __init__(self, filepath_or_buffer, dbname=None):
        """
        :param dbname: string denoting the database name. If None, the HDF file
            must contain a single database
        """
        Flatfile. __init__(self, filepath_or_buffer, dbname=dbname)

    def read_data(self, filepath_buffer, *, dbname=None):
        return pd.read_hdf(filepath_buffer, key=dbname)
