"""
Base Form for to model-to-data operations i.e. flatfile handling
"""
from typing import Iterable, Sequence

import pandas as pd
from django.forms import Form, ModelChoiceField
from django.forms.fields import CharField, FileField

from egsim.smtk import (ground_motion_properties_required_by,
                        intensity_measures_defined_for, registered_imts)
from egsim.smtk.flatfile import (read_flatfile, get_dtype_of, ColumnsRegistry,
                                 query as flatfile_query)
from egsim.api import models
from egsim.api.forms import EgsimBaseForm


# Let's provide uploaded flatfile Field in a separate Form as the Field is not
# strictly JSON-encodable (info here: https://stackoverflow.com/a/4083908) and
# should be kept private/hidden by default:
class _UploadedFlatfile(Form):
    flatfile = FileField(required=False,
                         allow_empty_file=False,
                         error_messages={
                            'empty': 'the submitted file is empty'
                         })


class FlatfileForm(EgsimBaseForm):
    """Base Form for handling Flatfiles"""

    # Custom API param names (see doc of `EgsimBaseForm._field2params` for details):
    _field2params: dict[str, tuple[str]] = {
        'selexpr': ('data-query', 'selection-expression')
    }
    flatfile = ModelChoiceField(
        queryset=models.Flatfile.queryset('name', 'media_root_path'),
        to_field_name="name",
        label='Flatfile',
        required=False)
    selexpr = CharField(required=False, label='Filter flatfile records via a '
                                              'query string')

    def __init__(self, data, files=None, **kwargs):
        # set now `self._u_ff`, in case `self.clean` is called in `super.__init__` below:
        self._u_ff = None if files is None else _UploadedFlatfile(files=files)
        super().__init__(data=data, **kwargs)

    def clean(self):
        """Call `super.clean()` and handle the flatfile"""
        u_form = self._u_ff

        # Handle flatfiles conflicts first. Note: with no selection from the web GUI we
        # have data['flatfile'] = None
        if u_form is not None and self.data.get('flatfile', None):
            self.add_error("flatfile", 'select a flatfile by name or upload one, '
                                       'not both')
        elif u_form is None and not self.data.get('flatfile', None):
            # note: with no selection from the web GUI we have data['flatfile'] = None
            self.add_error("flatfile",  self.ErrCode.required)

        cleaned_data = super().clean()

        if self.has_error('flatfile'):
            return cleaned_data

        u_flatfile = None  # None or bytes object

        if u_form is not None:
            if not u_form.is_valid():
                self._errors = u_form._errors
                return cleaned_data
            # the files dict[str, UploadedFile] should have only one item
            # in any case, get the first value:
            u_flatfile = u_form.files[next(iter(u_form.files))]  # Django Uploaded file
            u_flatfile = u_flatfile.file  # ByesIO or similar

        if u_flatfile is None:
            # cleaned_data["flatfile"] is a models.Flatfile instance:
            dataframe = cleaned_data["flatfile"].read_from_filepath()
        else:
            # u_ff = cleaned_data[key_u]
            try:
                # u_flatfile is a Django TemporaryUploadedFile or InMemoryUploadedFile
                # (the former if file size > configurable threshold
                # (https://stackoverflow.com/a/10758350):
                dataframe = read_flatfile(u_flatfile)
            except Exception as exc:
                # Use 'flatfile' as error key: users can not be confused
                # (see __init__), and also 'flatfile' is also the exposed key
                # for the `files` argument in requests
                self.add_error("flatfile", str(exc))
                return cleaned_data  # no need to further process

        # replace the flatfile parameter with the pandas dataframe:
        cleaned_data['flatfile'] = dataframe

        key = 'selexpr'
        selexpr = cleaned_data.get(key, None)
        if selexpr:
            try:
                cleaned_data['flatfile'] = flatfile_query(dataframe, selexpr).copy()
            except Exception as exc:
                # add_error removes also the field from self.cleaned_data:
                self.add_error(key, str(exc))

        return cleaned_data

    @staticmethod
    def get_columns_info(flatfile: pd.DataFrame) -> list[dict]:
        """Return a list of dicts representing a column:
        {"name":str, "type":str, "dtype":str, "help":str}.
        Columns that are defined in the flatfile and are also default columns
        registered in this program will not be returned if their data type does not match
        """
        columns = []
        for col in flatfile.columns:
            actual_dtype = getattr(get_dtype_of(flatfile[col]), 'value', None)  # str
            expected_dtype = ColumnsRegistry.get_dtype(col)
            c_info = FlatfileForm.get_registered_column_info(col)
            if not expected_dtype:  # not registered column, set the actual column dtype
                c_info['dtype'] = actual_dtype or ""
                c_info['help'] = ''  # for safety
            elif not c_info['dtype']:  # registered columns with no dtype set.
                # Use the actual dtype, if present
                if c_dtype:
                    c_info['dtype'] = c_dtype.name
            else:  # registered columns with dtype set. Check:
                if c_dtype != c_info['dtype']:
                    if not isinstance(c_dtype, pd.CategoricalDtype) and \
                            not isinstance(c_info['dtype'], pd.CategoricalDtype):
                        continue
                if isinstance(c_info['dtype'], pd.CategoricalDtype):
                    c_info['dtype'] = 'category'
            columns.append(c_info)
        return columns

    @staticmethod
    def get_registered_column_info(column: str):
        """Return a dict representing the given registered
        flatfile column:
            {"name":str, "type":str, "dtype":str, "help":str}
        """
        ret = {
            'name': column,
            'type': getattr(ColumnsRegistry.get_type(column), 'value', ""),
            'dtype': "",
            'help': ""
        }
        if ret['type'] is not None:
            ret['dtype'] = str(ColumnsRegistry.get_dtype(column) or "")
            ret['help'] = str(ColumnsRegistry.get_help(column) or "")
        return ret


def get_gsims_from_flatfile(flatfile_columns: Sequence[str]) -> Iterable[str]:
    """Yield the GSIM names supported by the given flatfile"""
    ff_cols = set('SA' if _.startswith('SA(') else _ for _ in flatfile_columns)
    imt_cols = ff_cols & set(registered_imts)
    ff_cols -= imt_cols
    for name in models.Gsim.names():
        imts = intensity_measures_defined_for(name)
        if not imts.intersection(imt_cols):
            continue
        if all(set(ColumnsRegistry.get_aliases(p)) & ff_cols
               for p in ground_motion_properties_required_by(name)):
            yield name
