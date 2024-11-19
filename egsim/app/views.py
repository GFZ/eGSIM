"""
Django views for the eGSIM app (web app with frontend)
"""
from io import BytesIO, StringIO
from os.path import splitext
from typing import Optional

from shapely.geometry import shape

from django.http import FileResponse, HttpResponseBase, HttpRequest
from django.shortcuts import render
from django.conf import settings

from ..api import models
from ..api.forms.flatfile import (FlatfileMetadataInfoForm,
                                  FlatfileValidationForm)
from ..api.forms import APIForm, EgsimBaseForm
from ..api.forms.residuals import ResidualsForm
from ..api.forms.scenarios import PredictionsForm
from ..api.views import MimeType, EgsimView
from .forms import PredictionsVisualizeForm, FlatfileVisualizeForm


img_ext = ('png', 'pdf', 'svg')
data_ext = ('hdf', 'csv')


class URLS:  # noqa
    """Define global URLs, to be used in both urls.py and injected in the web page"""

    # Form specific URLs
    PREDICTIONS = 'gui/egsim-predictions'  # <path>/<downloaded_file_basename>
    PREDICTIONS_VISUALIZE = 'gui/egsim-predictions-visualize'
    PREDICTIONS_PLOT_IMG = 'gui/egsim-predictions-plot'  # <path>/<downloaded_file_basename>  # noqa
    PREDICTIONS_RESPONSE_TUTORIAL = 'jupyter/predictions-response-tutorial.html'
    RESIDUALS = 'gui/egsim-residuals'  # <path>/<downloaded_file_basename>
    RESIDUALS_VISUALIZE = 'gui/egsim-residuals-visualize'
    RESIDUALS_PLOT_IMG = 'gui/egsim-residuals-plot'  # <path>/<downloaded_file_basename>
    RESIDUALS_RESPONSE_TUTORIAL = 'jupyter/residuals-response-tutorial.html'
    FLATFILE_VISUALIZE = 'gui/flatfile_visualization'
    FLATFILE_META_INFO = 'gui/get_flatfile_meta_info'
    FLATFILE_PLOT_IMG = 'gui/egsim-flatfile-plot'  # <path>/<downloaded_file_basename>
    # Misc (URLs shared between forms)
    GET_GSIMS_FROM_REGION = 'gui/get_models_from_region'
    FLATFILE_VALIDATE = 'gui/flatfile_validation'
    GET_GSIMS_INFO = 'gui/get_models_info'

    # webpage URLs:
    HOME_PAGE = 'home'
    DATA_PROTECTION_PAGE = 'https://www.gfz-potsdam.de/en/data-protection/'
    FLATFILE_META_INFO_PAGE = 'flatfile-metadata-info'
    FLATFILE_INSPECTION_PLOT_PAGE = 'flatfile-inspection-plot'
    IMPRINT_PAGE = "imprint"
    PREDICTIONS_PAGE = 'predictions'
    RESIDUALS_PAGE = 'residuals'
    REF_AND_LICENSE_PAGE = "ref_and_license"


def main(request, page=''):
    """view for the main page"""
    template = 'egsim.html'
    init_data = _get_init_data_json(settings.DEBUG) | \
        {'currentPage': page or URLS.HOME_PAGE}
    return render(request, template, context={'debug': settings.DEBUG,
                                              'init_data': init_data,
                                              'references': _get_references()})


def _get_init_data_json(debug=False) -> dict:
    """Return the JSON data to be passed to the browser at startup to initialize
    the page content

    :param debug: True or False, the value of the settings DEBUG flag
    """
    gsims = []
    imt_groups: dict[tuple, int] = {}  # noqa
    warning_groups: dict[str, int] = {}  # noqa
    for gsim in models.Gsim.queryset():
        # imt_names should be hashable and unique, so sort and make a tuple:
        imt_names = tuple(sorted(gsim.imts.split(" ")))
        imt_group_index = imt_groups.setdefault(imt_names, len(imt_groups))
        sa_limits = [gsim.min_sa_period, gsim.max_sa_period]
        if sa_limits[0] is None or sa_limits[1] is None:
            sa_limits = []
        model_warnings = []
        if gsim.unverified:
            model_warnings.append(models.Gsim.unverified.field.help_text)
        if gsim.experimental:
            model_warnings.append(models.Gsim.experimental.field.help_text)
        if gsim.adapted:
            model_warnings.append(models.Gsim.adapted.field.help_text)
        if model_warnings:
            warning_text = "; ".join(model_warnings)
            warning_group_index = warning_groups.setdefault(warning_text,
                                                            len(warning_groups))
            gsims.append([gsim.name, imt_group_index, sa_limits, warning_group_index])
        else:
            gsims.append([gsim.name, imt_group_index, sa_limits])

    # get regionalization data (for selecting models on a map):
    regionalizations = []
    for regx in models.Regionalization.queryset('name', 'url', 'media_root_path'):
        regionalizations.append({
            'name': regx.name,
            'bbox': _get_bbox(regx),  # tuple (min_lng, min_lat, max_lng, max_lat)
            'url': regx.url or ""
        })

    # get predefined flatfiles info:
    flatfiles = []
    for ffile in models.Flatfile.queryset(
            'name', 'display_name', 'url', 'media_root_path'):
        ff_form = FlatfileValidationForm({'flatfile': ffile.name})
        if ff_form.is_valid():
            flatfiles.append({
                'value': ffile.name,
                'name': ffile.name,
                'innerHTML': f'{ffile.name} ({ffile.display_name})',  # noqa
                'url': ffile.url,  # noqa
                'columns': ff_form.output()['columns']
            })

    predictions_form = PredictionsForm({
        'gsim': [],
        'imt': [],
        'magnitude': [],  # required
        'distance': [],  # required
        'format': 'hdf'
    })
    residuals_form = ResidualsForm({
        'gsim': [],
        'imt': [],
        'format': 'hdf'
    })
    if debug:
        predictions_form = PredictionsForm({
            'gsim': ['CauzziEtAl2014', 'BindiEtAl2014Rjb'],
            'imt': ['SA(0.05)', 'SA(0.075)'],  # default_imts,
            'magnitude': [4, 5, 6, 7],
            'distance': [1, 10, 100, 1000],
            'format': 'hdf'
        })
        residuals_form = ResidualsForm({
            'gsim': ['CauzziEtAl2014', 'BindiEtAl2014Rjb'],
            'imt': ['PGA', 'SA(0.1)'],
            'flatfile': 'esm2018',
            'flatfile-query': 'mag > 7',
            'format': 'hdf'
        })

    return {
        'pages': {  # tab key => url path (after the first slash)
            'predictions': URLS.PREDICTIONS_PAGE,
            'residuals': URLS.RESIDUALS_PAGE,
            'flatfile_meta_info': URLS.FLATFILE_META_INFO_PAGE,
            'flatfile_visualize': URLS.FLATFILE_INSPECTION_PLOT_PAGE,
            'ref_and_license': URLS.REF_AND_LICENSE_PAGE,
            'imprint': URLS.IMPRINT_PAGE,
            'home': URLS.HOME_PAGE,
            'data_protection': URLS.DATA_PROTECTION_PAGE
        },
        'urls': {
            'predictions': URLS.PREDICTIONS,
            'predictions_visualize': URLS.PREDICTIONS_VISUALIZE,
            'predictions_plot_img': [
                f'{URLS.PREDICTIONS_PLOT_IMG}.{ext}' for ext in img_ext
            ],
            'predictions_response_tutorial': URLS.PREDICTIONS_RESPONSE_TUTORIAL,
            'residuals': URLS.RESIDUALS,
            'residuals_visualize': URLS.RESIDUALS_VISUALIZE,
            'residuals_plot_img': [
                f'{URLS.RESIDUALS_PLOT_IMG}.{ext}' for ext in img_ext
            ],
            'get_gsim_info': URLS.GET_GSIMS_INFO,
            'residuals_response_tutorial': URLS.RESIDUALS_RESPONSE_TUTORIAL,
            'get_gsim_from_region': URLS.GET_GSIMS_FROM_REGION,
            'flatfile_meta_info': URLS.FLATFILE_META_INFO,
            'flatfile_visualize': URLS.FLATFILE_VISUALIZE,
            'flatfile_plot_img': [
                f'{URLS.FLATFILE_PLOT_IMG}.{ext}' for ext in img_ext
            ],
            'flatfile_validate': URLS.FLATFILE_VALIDATE,
        },
        'forms': {
            'predictions': form2dict(predictions_form),
            # in frontend, the form data below will be merged into forms.residuals above
            # (keys below will take priority):
            'predictions_plot': {'plot_type': 'm', 'format': 'json'},
            'residuals': form2dict(residuals_form),
            # in frontend, the form data below will be merged with forms.residuals above
            # (keys below will take priority):
            'residuals_plot': {'x': None, 'format': 'json'},
            'flatfile_meta_info': form2dict(
                FlatfileMetadataInfoForm({'gsim': [], 'imt': []})
            ),
            'flatfile_visualize': form2dict(FlatfileVisualizeForm({})),
            'misc': {
                'predictions': {
                    'msr': predictions_form.fields['msr'].choices,
                    'region': predictions_form.fields['region'].choices,
                    'help': {
                        PredictionsForm.param_names_of(n)[0]: f.help_text
                        for n, f in PredictionsForm.declared_fields.items()
                        if getattr(f, 'help_text', n).lower() != n.lower()
                    },
                    'tutorial_page_visible': False
                },
                'predictions_plot': {
                    'plot_types': PredictionsVisualizeForm.
                    declared_fields['plot_type'].choices
                },
                'flatfile_visualize': {
                    'selected_flatfile_fields': [],
                    'help': {
                        FlatfileVisualizeForm.param_names_of(n)[0]: f.help_text
                        for n, f in FlatfileVisualizeForm.declared_fields.items()
                        if getattr(f, 'help_text', n).lower() != n.lower()
                    }
                },
                'residuals': {
                    'selected_flatfile_fields': [],
                    'help': {
                        ResidualsForm.param_names_of(n)[0]: f.help_text
                        for n, f in ResidualsForm.declared_fields.items()
                        if getattr(f, 'help_text', n).lower() != n.lower()
                    },
                    'tutorial_page_visible': False
                },
                'flatfile_meta_info': {},
                'download_formats': data_ext
            }
        },
        'responses': {
            'predictions_plots': [],
            'residuals_plots': [],
            'flatfile_meta_info': None,
            'flatfile_visualize': [],
        },
        'gsims': gsims,
        # return the list of imts (imt_groups keys) in the right order:
        'imt_groups': sorted(imt_groups, key=imt_groups.get),
        # return the list of warnings (warning_groups keys) in the right order:
        'warning_groups': sorted(warning_groups, key=warning_groups.get),
        'flatfiles': flatfiles,
        'regionalizations': regionalizations
    }


def _get_bbox(reg: models.Regionalization) -> list[float]:
    """Return the bounds of all the regions coordinates in the given regionalization

    @param return: the 4-element list (minx, miny, maxx, maxy) i.e.
        (minLon, minLat, maxLon, maxLat)
    """
    feat_collection = reg.read_from_filepath()
    bounds = [180, 90, -180, -90]  # (minx, miny, maxx, maxy)
    for g in feat_collection['features']:
        bounds_ = shape(g['geometry']).bounds  # (minx, miny, maxx, maxy)
        bounds[0] = min(bounds[0], bounds_[0])
        bounds[1] = min(bounds[1], bounds_[1])
        bounds[2] = max(bounds[2], bounds_[2])
        bounds[3] = max(bounds[3], bounds_[3])
    return bounds


def _get_references():
    """Return the references of the data used by the program"""
    refs = {}
    for model_cls in [models.Regionalization, models.Flatfile]:
        for item in model_cls.queryset().values():
            url = item.get('url', '')
            if not url:
                url = item.get('doi', '')
                if url and not url.startswith('http'):
                    url = f'https://doi.org/{url}'
            if not url:
                continue
            name = item.get('display_name', None) or item['name']
            refs[name] = url
    return refs


def form2dict(form: EgsimBaseForm, compact=False) -> dict:
    """Return the `data` argument passed in the form
    constructor in a JSON serializable dict

    @param form: the EgsimBaseForm (Django Form subclass)
    @param compact: skip optional parameters, i.e. those whose value equals
        the default when missing
    """
    ret = {}
    for field_name, value in form.data.items():
        if compact:
            field = form.declared_fields.get(field_name, None)
            if field is None:
                continue
            is_field_optional = not field.required or field.initial is not None
            if field is not None and is_field_optional:
                if field.initial == value:
                    continue
        ret[form.param_name_of(field_name)] = value
    return ret


class PlotsImgDownloader(EgsimView):

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None) -> HttpResponseBase:
        """Process the response from a given request and the data / files
        extracted from it"""
        filename = request.path[request.path.rfind('/')+1:]
        img_format = splitext(filename)[1][1:].lower()
        try:
            content_type = getattr(MimeType, img_format)
        except AttributeError:
            return self.error_response(f'Invalid format "{img_format}"')

        from plotly import graph_objects as go, io as pio
        fig = go.Figure(data=data['data'], layout=data['layout'])
        # fix for https://github.com/plotly/plotly.py/issues/3469:
        pio.full_figure_for_development(fig, warn=False)
        img_bytes = fig.to_image(
            format=img_format, width=data['width'], height=data['height'], scale=5
        )
        return FileResponse(BytesIO(img_bytes), content_type=content_type,
                            filename=filename, as_attachment=True)


class PredictionsHtmlTutorial(EgsimView):

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None) -> HttpResponseBase:
        """Process the response from a given request and the data / files
        extracted from it"""
        from egsim.api.data.client.snippets.get_egsim_predictions import \
            get_egsim_predictions
        api_form = PredictionsForm({
            'gsim': ['CauzziEtAl2014', 'BindiEtAl2014Rjb'],
            'imt': ['PGA', 'SA(0.1)'],
            'magnitude': [4, 5, 6],
            'distance': [10, 100]
        })
        return get_html_tutorial(
            request, 'predictions', api_form, get_egsim_predictions
        )


class ResidualsHtmlTutorial(EgsimView):

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None) -> HttpResponseBase:
        """Process the response from a given request and the data / files
        extracted from it"""
        from egsim.api.data.client.snippets.get_egsim_residuals import \
            get_egsim_residuals
        api_form = ResidualsForm({
            'gsim': ['CauzziEtAl2014', 'BindiEtAl2014Rjb'],
            'imt': ['PGA', 'PGV'],
            'data-query': '(mag > 7) & (vs30 > 1100)',
            'flatfile': 'esm2018'
        })
        return get_html_tutorial(
            request, 'residuals', api_form, get_egsim_residuals
        )


def get_html_tutorial(
        request,
        key: str,
        api_form: APIForm,
        api_client_function
) -> HttpResponseBase:
    import re

    # create dataframe htm:
    s = StringIO()
    if api_form.is_valid():
        api_form.output().to_html(s, index=True,
                                  classes='table table-bordered table-light my-2',
                                  border=0,
                                  max_rows=3)

    if key == 'residuals':
        s.write('Or, if ranking=True:')
        api_form.cleaned_data['ranking'] = True
        api_form.output().to_html(s, index=True,
                                  classes='table table-bordered table-light my-2',
                                  border=0,
                                  max_rows=3)

    dataframe_html = re.sub(r"<t([dh])\s*>",
                            r"<t\1 style='white-space: nowrap;'>",
                            s.getvalue())

    # create explanation (from code snippet docstring):
    dataframe_info = api_client_function.__doc__
    dataframe_info = dataframe_info[dataframe_info.index('Returns:'):]
    dataframe_info = dataframe_info.split("\n")  # split strings
    dataframe_info = dataframe_info[3:]  # remove 1st 3 lines
    dataframe_info = [_.strip() for _ in dataframe_info]  # strip each line
    dataframe_info = ['</p><p>' if not _ else _ for _ in dataframe_info]
    if any('<p>' in _ for _ in dataframe_info):
        dataframe_info = ['<p>'] + dataframe_info + ['</p>']
    dataframe_info = "\n".join(dataframe_info)

    return render(request, 'downloaded-data-tutorial.html',
                  context={
                      'key': key,
                      'dataframe_html': dataframe_html,
                      'dataframe_info': dataframe_info
                  })
