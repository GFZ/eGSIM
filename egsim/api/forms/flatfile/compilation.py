"""
Django Forms for eGSIM flatfile plot generator

@author: riccardo
"""
from typing import Iterable, Any

from django.db.models import Prefetch

from . import FlatfileForm
from .. import APIForm, MultipleChoiceWildcardField, get_gsim_choices

from .. import models
from ...flatfile import EVENT_ID_COL, EVENT_ID_DESC, EVENT_ID_DTYPE


class FlatfileColumnsForm(APIForm):
    """Form for flatfile compilation, return the necessary columns from a
    given list of Gsims"""

    # Set the public names of this Form Fields as `public_name: attribute_name`
    # mappings. Superclass mappings are merged into this one. An attribute name
    # can be keyed by several names, and will be keyed by itself anyway if not
    # done here (see `egsim.forms.EgsimFormMeta` for details)
    public_field_names = {}

    gsim = MultipleChoiceWildcardField(required=False, choices=get_gsim_choices,
                                       label='Ground Shaking Intensity Model(s)')


    @classmethod
    def process_data(cls, cleaned_data: dict) -> dict:
        """Process the input data `cleaned_data` returning the response data
        of this form upon user request.
        This method is called by `self.response_data` only if the form is valid.

        :param cleaned_data: the result of `self.cleaned_data`
        """
        qry = models.FlatfileColumn.objects  # noqa

        required = set()
        # Try to perform everything in a single more efficient query. Use
        # prefetch_related for this. It Looks like we need to assign the imts to a
        # new attribute, the attribute "Gsim.imts" does not work as expected
        if cleaned_data.get('gsim', []):
            required = set(qry.only('name').
                           filter(gsims__name__in=cleaned_data['gsim']).
                           values_list('name', flat=True))

        columns = {EVENT_ID_COL: {'help': EVENT_ID_DESC, 'dtype': EVENT_ID_DTYPE}}
        attrs = 'name', 'help', 'properties'
        for name, help, props in qry.only(*attrs).values_list(*attrs):
            columns[name] = {}
            if not required or name in required:
                columns[name] = {'help': help, 'dtype': props['dtype']}

        return columns

    @classmethod
    def csv_rows(cls, processed_data: dict) -> Iterable[Iterable[Any]]:
        """Yield CSV rows, where each row is an iterables of Python objects
        representing a cell value. Each row doesn't need to contain the same
        number of elements, the caller function `self.to_csv_buffer` will pad
        columns with Nones, in case (note that None is rendered as "", any other
        value using its string representation).

        :param processed_data: dict resulting from `self.process_data`
        """
        names = processed_data.keys()
        yield names
        yield (processed_data[n].get('help', '') for n in names)
        yield (processed_data[n].get('dtype', '') for n in names)
