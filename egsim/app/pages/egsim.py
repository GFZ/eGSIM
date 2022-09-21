"""Module for rendering the main page of the site (single page application)"""

from typing import Any, Type, Callable, Union
import json

from django.db.models import Prefetch
from django.forms import (Field, IntegerField, ModelChoiceField)
from django.forms.widgets import ChoiceWidget, Input

from .. import TAB, URLS
from ...api import models
from ...api.forms import EgsimBaseForm
from ...api.forms.flatfile import FlatfileForm
from ...api.forms.flatfile_compilation import FlatfileRequiredColumnsForm
from ...api.forms.flatfile.inspection import FlatfilePlotForm


def get_context(selected_menu=None, debug=True) -> dict:
    """The context to be injected in the template of the main HTML page"""

    # Tab components (one per tab, one per activated vue component)
    # (key, label and icon) (the last is bootstrap fontawesome name)
    components_tabs = [[_.name, _.title, _.icon] for _ in TAB]

    # this can be changed if needed:
    sel_component = TAB.home.name if not selected_menu else selected_menu

    # setup browser detection:
    allowed_browsers = {'Chrome': 49, 'Firefox': 45, 'Safari': 10}
    allowed_browsers_msg = ', '.join(f'{b}≥{v}' for b, v in allowed_browsers.items())
    invalid_browser_message = (f'Your browser or version number '
                               f'does not seem to match {allowed_browsers_msg}. '
                               f'This portal might not work as expected')

    # Get gsims and all related data (imts and warnings). Try to perform everything
    # in a single more efficient query. Use prefetch_related for this:
    gsims = []
    imts = Prefetch('imts', queryset=models.Imt.objects.only('name'))
    for gsim in models.Gsim.objects.only('name', 'warning').prefetch_related(imts):
        imt_names = [i for i in gsim.imts.values_list('name', flat=True)]
        gsims.append([gsim.name, imt_names, gsim.warning or ""])

    # get regionalization data (for selecting models on a map):
    regionalization = {
        'url': URLS.GET_GSIMS_FROM_REGION,
        'names': list(models.Regionalization.objects.values_list('name', flat=True))
    }

    # get predefined flatfiles info:
    flatfiles = [{'name': r.name, 'label': f'{r.name} ({r.display_name})', 'url': r.url,
                  'columns': FlatfileForm.get_flatfile_dtypes(FlatfileForm.read_flatfile_from_db(r), compact=True)}
                 for r in models.Flatfile.get_flatfiles(hidden=False)]

    # Get component props (core data needed for Vue rendering):
    components_props = get_components_properties(debug)

    return {
        'debug': debug,
        'sel_component': sel_component,
        'components': components_tabs,
        'component_props': json.dumps(components_props, separators=(',', ':')),
        'gsims': json.dumps(gsims, separators=(',', ':')),
        'flatfiles': flatfiles,
        'flatfile_upload_url': URLS.FLATFILE_INSPECTION,
        'regionalization': regionalization,
        'allowed_browsers': {k.lower(): v for k, v in allowed_browsers.items()},
        'invalid_browser_message': invalid_browser_message,
        'newpage_urls': {
            'api': URLS.API,
            'imprint': URLS.IMPRINT,
            'data_protection': URLS.DATA_PROTECTION,
            'ref_and_license': URLS.REF_AND_LICENSE
        }
    }


def get_components_properties(debugging=False) -> dict[str, dict[str, Any]]:
    """Return a dict with all the properties to be passed
    as VueJS components in the frontend

    :param debugging: if True, the components input elements will be setup
        with default values so that the frontend FORMS will be ready to
        test click buttons
    """
    def ignore_choices(field_att_name):
        return field_att_name in ('gsim', 'imt', 'flatfile')

    # properties to be passed to vuejs components.
    # If you change THE KEYS of the dict here you should change also the
    # javascript:
    components_props = {
        TAB.home.name: {
            'src': URLS.HOME_NO_MENU
        },
        TAB.trellis.name: {
            'form': form_to_json(TAB.trellis.formclass, ignore_choices),
            'url': TAB.trellis.urls[0],
            'urls': {
                'downloadRequest': f"{URLS.DOWNLOAD_REQUEST}/{TAB.trellis.name}/"
                                   f"{TAB.trellis.download_request_filename}",
                'downloadResponse': f"{URLS.DOWNLOAD_RESPONSE}/{TAB.trellis.name}/"
                                    f"{TAB.trellis.download_response_filename}"
            }
        },
        TAB.flatfile.name: {  # FIXME REMOVE
            'forms': [form_to_json(FlatfileRequiredColumnsForm, ignore_choices),
                      form_to_json(FlatfilePlotForm, ignore_choices)],
            'urls': [URLS.FLATFILE_REQUIRED_COLUMNS,
                     URLS.FLATFILE_PLOT]
        },
        TAB.residuals.name: {
            'form': form_to_json(TAB.residuals.formclass, ignore_choices),
            'url': TAB.residuals.urls[0],
            'urls': {
                'downloadRequest': f"{URLS.DOWNLOAD_REQUEST}/{TAB.residuals.name}/"
                                   f"{TAB.residuals.download_request_filename}",
                'downloadResponse': f"{URLS.DOWNLOAD_RESPONSE}/{TAB.residuals.name}/"
                                    f"{TAB.residuals.download_response_filename}"
            }
        },
        TAB.testing.name: {
            'form': form_to_json(TAB.testing.formclass, ignore_choices),
            'url': TAB.testing.urls[0],
            'urls': {
                'downloadRequest': f"{URLS.DOWNLOAD_REQUEST}/{TAB.testing.name}/"
                                   f"{TAB.testing.download_request_filename}",
                'downloadResponse': f"{URLS.DOWNLOAD_RESPONSE}/{TAB.testing.name}/"
                                    f"{TAB.testing.download_response_filename}"
            }
        }
    }

    # FlatfilePlotForm has x and y that must be represented as <select> but cannot
    # be implemented as ChoiceField, because their content is not static but
    # flatfile dependent. So
    plot_form = components_props[TAB.flatfile.name]['forms'][-1]
    plot_form['x']['type'] = 'select'
    plot_form['y']['type'] = 'select'
    # provide initial value:
    plot_form['x']['choices'] = [('', 'None: display histogram of Y values')]
    plot_form['y']['choices'] = [('', 'None: display histogram of X values')]

    if debugging:
        _setup_default_values(components_props)
    return components_props


def _setup_default_values(components_props: dict[str, dict[str, Any]]):
    """Set up some dict keys and sub-keys so that the frontend FORM is already
    filled with default values for easy testing
    """
    gsimnames = ['AkkarEtAlRjb2014', 'BindiEtAl2014Rjb', 'BooreEtAl2014',
                 'CauzziEtAl2014']
    val = 'value'
    trellis_form = components_props['trellis']['form']
    trellis_form['gsim'][val] = gsimnames
    trellis_form['imt'][val] = ['PGA']
    trellis_form['magnitude'][val] = "5:7"
    trellis_form['distance'][val] = "10 50 100"
    trellis_form['aspect'][val] = 1
    trellis_form['dip'][val] = 60
    trellis_form['plot_type'][val] = 's'

    residuals_form = components_props['residuals']['form']
    residuals_form['gsim'][val] = gsimnames
    residuals_form['imt'][val] = ['PGA', "SA(0.2)", "SA(1.0)", "SA(2.0)"]
    residuals_form['selexpr'][val] = "magnitude > 5"
    residuals_form['plot_type'][val] = 'res'

    testing_form = components_props['testing']['form']
    testing_form['gsim'][val] = gsimnames + ['AbrahamsonSilva2008']
    testing_form['imt'][val] = ['PGA', 'PGV', "0.2", "1.0", "2.0"]
    testing_form['fit_measure'][val] = ['res', 'lh']


def form_to_json(form: Union[Type[EgsimBaseForm], EgsimBaseForm],
                 ignore_choices: Callable[[str], bool] = None) -> dict[str, dict[Any]]:
    """Return a JSON-serializable dictionary of the Form Field names mapped to
    their properties, in order e.g. to be injected in HTML templates in order
    to render the Field as HTML component.

    :param form: EgsimBaseForm class or object (class instance)
    :param ignore_choices: callable accepting a string (field attribute name)
        and returning True or False. If False, the Field choices will not be
        loaded and the returned dict 'choices' key will be `[]`. Useful for
        avoiding time consuming long list loading
    """

    if ignore_choices is None:
        def ignore_choices(*a, **k):
            return False

    form_data = {}
    # keep track of Field done. Initialize the set below with the ignored fields:
    field_done = {'format', 'csv_sep', 'csv_dec'}
    # iterate over the field (public) names because we also have the attribute
    # name immediately available:
    for param_names, field_name, field in form.apifields():
        if field_name in field_done:
            continue
        field_done.add(field_name)
        field_dict = field_to_dict(field, ignore_choices=ignore_choices(field_name))
        field_dict |= dict(field_to_htmlelement_attrs(field), name=param_names[0])
        field_dict['error'] = ''
        form_data[field_name] = field_dict

    return form_data


def field_to_dict(field: Field, ignore_choices: bool = False) -> dict:
    """Convert a Field to a JSON serializable dict with keys:
    {
        'initial': field.initial,
        'help': (field.help_text or "").strip(),
        'label': (field.label or "").strip(),
        'is_hidden': False,
        'choices': field.choices
    }

    :param field: a Django Field
    :param ignore_choices: boolean. If True, 'chocies' will be not evaluated
        and set to `[]`. Useful with long lists for saving time and space
    """

    choices = []

    if not ignore_choices:
        choices = list(get_choices(field))

    return {
        'value': field.initial,
        'help': (field.help_text or "").strip(),
        'label': (field.label or "").strip(),
        # 'is_hidden': False,
        'choices': choices
    }


def get_choices(field: Field):
    """Yields tuples (value, label) corresponding to the field choices"""
    if isinstance(field, ModelChoiceField):
        # choices are ModeChoiceIteratorValue instances and are not
        # JSON serializable. Let's take their `value` attribute:
        for (val, label) in field.choices:
            yield val.value, label
    else:
        yield from getattr(field, 'choices', [])


def field_to_htmlelement_attrs(field: Field) -> dict:
    """Convert a Field to a JSON serializable dict with keys denoting the
    attributes of the associated HTML Element, e.g.:
    {'type', 'required', 'disabled', 'min' 'max', 'multiple'}
    and values inferred from the Field

    :param field: a Django Field
    """
    # Note: we could return the dict `field.widget.get_context` but we build our
    # own for several reasons, e.g.:
    # 1. Avoid loading all <option>s for Gsim and Imt (we could subclass
    #    `optgroups` in `widgets.SelectMultiple` and return [], but it's clumsy)
    # 2. Remove some attributes (e.g. checkbox with the 'checked' attribute are
    #    not compatible with VueJS v-model or v-checked)
    # 3. Some Select with single choice set their initial value as list  (e.g.
    #    ['value'] instead of 'value') and I guess VueJs prefers strings.

    # All in all, instead of complex patching we provide our code here:
    widget = field.widget
    attrs = {
        # 'hidden': widget.is_hidden,
        'required': field.required,
        'disabled': False
    }
    if isinstance(field, IntegerField):  # note: FloatField inherits from IntegerField
        if field.min_value is not None:
            attrs['min'] = field.min_value
        if field.max_value is not None:
            attrs['max'] = field.max_value
        # The step attribute seems to be needed by some browsers:
        if field.__class__.__name__ == IntegerField.__name__:
            attrs['step'] = '1'  # IntegerField
        else:  # FloatField or DecimalField.
            attrs['step'] = 'any'

    if isinstance(widget, ChoiceWidget):
        if widget.allow_multiple_selected:
            attrs['multiple'] = True
    elif isinstance(widget, Input):
        attrs['type'] = widget.input_type

    return attrs