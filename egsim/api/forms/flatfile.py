"""
Base Form for to model-to-data operations i.e. flatfile handling
"""
from typing import Optional
import pandas as pd
from django.core.files.uploadedfile import TemporaryUploadedFile

from django.forms import Form
from django.forms.fields import CharField, FileField

from egsim.smtk import (ground_motion_properties_required_by,
                        intensity_measures_defined_for, get_sa_limits)
from egsim.smtk.flatfile import (read_flatfile, get_dtype_of, FlatfileMetadata,
                                 query as flatfile_query, EVENT_ID_COLUMN_NAME,
                                 FlatfileError, FlatfileQueryError,
                                 IncompatibleColumnError)
from egsim.api import models
from egsim.api.forms import EgsimBaseForm, APIForm, GsimForm, split_pars


# Let's provide uploaded flatfile Field in a separate Form as the Field is not
# strictly JSON-encodable (info here: https://stackoverflow.com/a/4083908) and
# should be kept private/hidden by default:
class _UploadedFlatfile(Form):
    flatfile = FileField(
        required=False,
        allow_empty_file=False,
        error_messages={
            'empty': 'the submitted file is empty'
        }
    )


class FlatfileForm(EgsimBaseForm):
    """Base Form for handling Flatfiles"""

    # Custom API param names (see doc of `EgsimBaseForm._field2params` for details):
    _field2params: dict[str, tuple[str]] = {
        'selexpr': ('flatfile-query', 'data-query', 'selection-expression'),
        'flatfile': ('flatfile', 'data')
    }
    flatfile = CharField(
        required=False,
        help_text="The flatfile (pre- or user-defined) containing observed ground "
                  "motion properties and intensity measures, in CSV or HDF format"
    )  # Note: with a ModelChoiceField the benefits of handling validation are outweighed
    # by the fixes needed here and there to make values JSON serializable, so we opt for
    # a CharField + custom validation in `clean`
    selexpr = CharField(
        required=False,
        help_text='Filter flatfile records (rows) matching query expressions applied '
                  'on the columns, e.g.: "(mag > 6) & (rrup < 10)" (&=and, |=or)'
    )

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
            self.add_error("flatfile",  self.ErrMsg.required)

        cleaned_data = super().clean()

        if self.has_error('flatfile'):
            return cleaned_data

        u_flatfile = None  # None or bytes object

        if u_form is not None:
            if not u_form.is_valid():
                self._errors = u_form._errors
                return cleaned_data
            # u_form.files is a MultiValueDict or a dict (I guss the latter when
            # we do provide a flatfile). We do not care about the keys as long as
            # there is just one key:
            ff_keys = list(u_form.files.keys())
            if len(ff_keys) != 1:
                self.add_error("flatfile", f"only one flatfile should be uploaded "
                                           f"(found {len(ff_keys)})")
                return cleaned_data
            # Get our uploaded file (Django UploadedFile object, for ref see
            # https://docs.djangoproject.com/en/5.0/ref/files/uploads/):
            uploaded_flatfile = u_form.files[ff_keys[0]]
            if isinstance(uploaded_flatfile, TemporaryUploadedFile):
                # File on disk (Django TemporaryUploadedFile object), get the path:
                u_flatfile = uploaded_flatfile.temporary_file_path()
            else:
                # in-memory file (Django UploadedFile object), get the Python
                # file-like object:
                u_flatfile = uploaded_flatfile.file
            # Note: as of pandas 2.2.2, HDF does not support reading from stream
            # or buffer. As such, we force every uploaded flatfile to be a
            # TemporaryUploadedFile (via settings.FILE_UPLOAD_MAX_MEMORY_SIZE = 0),
            # and in-memory files are used only in some tests

        if u_flatfile is None:  # predefined flatfile
            flatfile_db_obj = models.Flatfile.queryset('name', 'filepath').\
                filter(name=cleaned_data['flatfile']).first()
            if flatfile_db_obj is None:
                self.add_error("flatfile", self.ErrMsg.invalid_choice)
                return cleaned_data
            # cleaned_data["flatfile"] is a models.Flatfile instance:
            dataframe = flatfile_db_obj.read_from_filepath()
        else:  # uploaded (user-defined) flatfile
            try:
                # u_flatfile is a Django TemporaryUploadedFile or InMemoryUploadedFile
                # (the former if file size > configurable threshold
                # (https://stackoverflow.com/a/10758350):
                dataframe = read_flatfile(u_flatfile)
            except IncompatibleColumnError as ice:
                self.add_error('flatfile', f'column names conflict {str(ice)}')
                return cleaned_data
            except FlatfileError as err:
                self.add_error("flatfile", str(err))
                return cleaned_data  # no need to further process

        # replace the flatfile parameter with the pandas dataframe:
        cleaned_data['flatfile'] = dataframe

        key = 'selexpr'
        selexpr = cleaned_data.get(key, None)
        if selexpr:
            try:
                cleaned_data['flatfile'] = flatfile_query(dataframe, selexpr).copy()
            except FlatfileQueryError as exc:
                # add_error removes also the field from self.cleaned_data:
                self.add_error(key, str(exc))

        return cleaned_data


class FlatfileValidationForm(APIForm, FlatfileForm):
    """Form for flatfile validation, on success
    return info from a given uploaded flatfile"""

    def output(self) -> Optional[dict]:
        """
        Compute and return the output from the input data (`self.cleaned_data`),
        which is a dict with all flatfile columns info.
        This method must be called after checking that `self.is_valid()` is True.

        :return: any Python object (e.g., a JSON-serializable dict)
        """
        # return human-readable column metadata from its values (dataframe[col]).
        cleaned_data = self.cleaned_data
        dataframe = cleaned_data['flatfile']
        columns = [
            get_hr_flatfile_column_meta(col, dataframe[col])
            for col in sorted(dataframe.columns)
        ]

        return {'columns': columns}


class FlatfileMetadataInfoForm(GsimForm, APIForm):
    """Form for querying the necessary metadata columns from a given selection
    of models"""

    def clean(self):
        unique_imts = FlatfileMetadata.get_intensity_measures()

        for m_name, model in self.cleaned_data['gsim'].items():
            imts = intensity_measures_defined_for(model)
            unique_imts &= set(imts)
            if not unique_imts:
                break

        if 'SA' in unique_imts:
            inf = float('inf')
            min_p, max_p = -inf, inf
            for m_name, model in self.cleaned_data['gsim'].items():
                p_bounds = get_sa_limits(model)
                if p_bounds is None:
                    # FIXME: we assume a model supporting SA with no period limits
                    #  is defined for all periods, but is it true?
                    continue
                min_p = max(min_p, p_bounds[0])
                max_p = min(max_p, p_bounds[1])
            if min_p > max_p:
                unique_imts -= {'SA'}
            elif -inf < min_p <= max_p < inf:
                self.cleaned_data['sa_period_limits'] = [min_p, max_p]

        if not unique_imts:
            self.add_error('gsim', 'No intensity measure defined for all models')

        self.cleaned_data['imt'] = sorted(unique_imts)
        return self.cleaned_data

    def output(self) -> dict:
        """Compute and return the output from the input data (`self.cleaned_data`).
        This method must be called after checking that `self.is_valid()` is True

        :return: any Python object (e.g., a JSON-serializable dict)
        """
        cleaned_data = self.cleaned_data
        gsims = list(cleaned_data['gsim'])

        required_columns = (ground_motion_properties_required_by(*gsims) |
                            {EVENT_ID_COLUMN_NAME})  # <- event id always required
        ff_columns = {FlatfileMetadata.get_aliases(c)[0] for c in required_columns}

        imts = cleaned_data['imt']

        columns = []
        sa_period_limits = cleaned_data.get('sa_period_limits', None)
        for col in sorted(ff_columns | set(imts)):
            columns.append(get_hr_flatfile_column_meta(col))
            if col == 'SA' and sa_period_limits is not None:
                columns[-1]['help'] = sa_hr_help(
                    gsims, columns[-1]['help'], sa_period_limits
                )

        return {'columns': columns}


def get_hr_flatfile_column_meta(name: str, values: Optional[pd.Series] = None) -> dict:
    """Return human-readable (hr) flatfile column metadata in the following `dict` form:
    {
        'name': str,
        'help': str,
        'dtype': str,
        'type': str
    }

    :param name: the flatfile column name
    :param values: the column data, ignored if `name` is a registered flatfile column.
        Otherwise, if provided, it will be used to infer the column metadata
    """
    c_type = ""
    c_help = ""
    c_dtype = None
    c_categories = []

    if FlatfileMetadata.has(name):
        c_dtype = FlatfileMetadata.get_dtype(name)
        cat_dtype = FlatfileMetadata.get_categorical_dtype(name)
        if cat_dtype is not None:
            # c_categories is a pandas CategoricalStype. So:
            c_dtype = get_dtype_of(cat_dtype.categories)
            c_categories = cat_dtype.categories.tolist()
        c_type = getattr(FlatfileMetadata.get_type(name), 'value', "")
        c_help = FlatfileMetadata.get_help(name) or ""
        c_aliases = FlatfileMetadata.get_aliases(name)
        if len(c_aliases) > 1:
            c_aliases = [n for n in c_aliases if n != name]
            c_aliases = (f"Alternative valid name{'s' if len(c_aliases) != 1 else ''}: "
                         f"{', '.join(c_aliases)}")
            if c_help:
                c_help += f". {c_aliases}"
            else:
                c_help = c_aliases
    elif values is not None:
        try:
            c_categories = values.cat.categories
            c_dtype = get_dtype_of(c_categories)
        except AttributeError:
            c_categories = []
            c_dtype = get_dtype_of(values)

    if c_dtype is not None:
        c_dtype = c_dtype.value
        if len(c_categories):
            if values is not None:  # custom values, compact categories info:
                c_dtype += f", categorical, {len(c_categories)} values"
            else:
                c_dtype += (f", categorical, to be chosen from "
                            f"{', '.join(str(c) for c in c_categories)}")
    else:
        c_dtype = ""

    return {
        'name': name,
        'type': c_type,
        'dtype': c_dtype,
        'help': c_help
    }


def sa_hr_help(gsims, sa_help: str, sa_p_limits: list[float]) -> str:
    """
    builds the SA field help, human readable (hr). The output will be HTML.
    `sa_help` should be text in flatfile_metadata for the field 'SA'
    """
    sa_p_min = sa_p_max = sa_p_limits[0]
    if len(sa_p_limits) > 1:
        sa_p_max = sa_p_limits[1]
    the_selected_model = 'the selected model'
    if len(gsims) > 1:
        the_selected_model = f'all {len(gsims)} selected models'
    new_text = (f'<b>The period range supported by {the_selected_model} '
                f'is [{sa_p_min}, {sa_p_max}] (endpoints included)</b>.'
                if sa_p_min < sa_p_max else
                f'<b>The only period supported by {the_selected_model} '
                f'is {sa_p_min}</b>.')
    help_pars = split_pars(sa_help)
    help_pars = help_pars[:2] + [new_text] + help_pars[2:]
    return " ".join(s.strip() for s in help_pars)