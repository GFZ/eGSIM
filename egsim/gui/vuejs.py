from typing import Any, Type, Callable, Union
from . import TABS, URLS
from ..api.forms import EgsimBaseForm
from ..api.forms.tools import field_to_dict, field_to_htmlelement_attrs


def get_components_properties(debugging=False) -> dict[str, dict[str, Any]]:
    """Return a dict with all the properties to be passed
    as VueJS components in the frontend

    :param debugging: if True, the components input elements will be setup
        with default values so that the frontend FORMS will be ready to
        test click buttons
    """
    def ignore_choices(field_att_name):
        return field_att_name in ('gsim', 'imt')

    # properties to be passed to vuejs components.
    # If you change THE KEYS of the dict here you should change also the
    # javascript:
    components_props = {
        TABS.home.name: {
            'src': URLS.HOME_PAGE
        },
        TABS.trellis.name: {
            'form': form_to_vuejs(TABS.trellis.formclass, ignore_choices),
            'url': TABS.trellis.urls[0],
            'urls': {
                # the lists below must be made of elements of
                # the form [key, url]. For each element the JS library (VueJS)
                # will then create a POST data and issue a POST request
                # at the given url (see JS code for details).
                # Convention: If url is a JSON-serialized string representing
                # the dict: '{"file": <str>, "mimetype": <str>}'
                # then we will simply donwload the POST data without calling
                # the server.
                # Otherwise, when url denotes a Django view, remember
                # that the function should build a response with
                # response['Content-Disposition'] = 'attachment; filename=%s'
                # to tell the browser to download the content.
                # (it is just for safety, remember that we do not care because
                # we will download data in AJAX calls which do not care about
                # content disposition
                'downloadRequest': [
                    [
                        'json',
                        "{0}/{1}/{1}.config.json".format(URLS.DOWNLOAD_CFG,
                                                         TABS.trellis.name)
                    ],
                    [
                        'yaml',
                        "{0}/{1}/{1}.config.yaml".format(URLS.DOWNLOAD_CFG,
                                                         TABS.trellis.name)
                    ]
                ],
                'downloadResponse': [
                    [
                        'json',
                        '{"file": "%s.json", "mimetype": "application/json"}' %
                        TABS.trellis.name
                    ],
                    [
                        'text/csv',
                        "{0}/{1}/{1}.csv".format(URLS.DOWNLOAD_ASTEXT,
                                                 TABS.trellis.name)
                    ],
                    [
                        "text/csv, decimal comma",
                        "{0}/{1}/{1}.csv".format(URLS.DOWNLOAD_ASTEXT_EU,
                                                 TABS.trellis.name)
                    ],
                ],
                'downloadImage': [
                    [
                        'png (visible plots only)',
                        "%s/%s.png" % (URLS.DOWNLOAD_ASIMG, TABS.trellis.name)
                    ],
                    [
                        'pdf (visible plots only)',
                        "%s/%s.pdf" % (URLS.DOWNLOAD_ASIMG, TABS.trellis.name)
                    ],
                    [
                        'eps (visible plots only)',
                        "%s/%s.eps" % (URLS.DOWNLOAD_ASIMG, TABS.trellis.name)
                    ],
                    [
                        'svg (visible plots only)',
                        "%s/%s.svg" % (URLS.DOWNLOAD_ASIMG, TABS.trellis.name)
                    ]
                ]
            }
        },
        # KEY.GMDBPLOT: {  # FIXME REMOVE
        #     'form': to_vuejs_dict(GmdbPlotView.formclass()),
        #     'url': URLS.GMDBPLOT_RESTAPI
        # },
        TABS.residuals.name: {
            'form': form_to_vuejs(TABS.residuals.formclass, ignore_choices),
            'url': TABS.residuals.urls[0],
            'urls': {
                # download* below must be pairs of [key, url]. Each url
                # must return a
                # response['Content-Disposition'] = 'attachment; filename=%s'
                # url can also start with 'file:///' telling the frontend
                # to simply download the data
                'downloadRequest': [
                    [
                        'json',
                        "{0}/{1}/{1}.config.json".format(URLS.DOWNLOAD_CFG,
                                                         TABS.residuals.name)],
                    [
                        'yaml',
                        "{0}/{1}/{1}.config.yaml".format(URLS.DOWNLOAD_CFG,
                                                         TABS.residuals.name)
                    ],
                ],
                'downloadResponse': [
                    [
                        'json',
                        '{"file": "%s.json", "mimetype": "application/json"}' %
                        TABS.residuals.name
                    ],
                    [
                        'text/csv',
                        "{0}/{1}/{1}.csv".format(URLS.DOWNLOAD_ASTEXT,
                                                 TABS.residuals.name)
                    ],
                    [
                        "text/csv, decimal comma",
                        "{0}/{1}/{1}.csv".format(URLS.DOWNLOAD_ASTEXT_EU,
                                                 TABS.residuals.name)
                    ]
                ],
                'downloadImage': [
                    [
                        'png (visible plots only)',
                        "%s/%s.png" % (URLS.DOWNLOAD_ASIMG, TABS.residuals.name)
                    ],
                    [
                        'pdf (visible plots only)',
                        "%s/%s.pdf" % (URLS.DOWNLOAD_ASIMG, TABS.residuals.name)
                    ],
                    [
                        'eps (visible plots only)',
                        "%s/%s.eps" % (URLS.DOWNLOAD_ASIMG, TABS.residuals.name)
                    ],
                    [
                        'svg (visible plots only)',
                        "%s/%s.svg" % (URLS.DOWNLOAD_ASIMG, TABS.residuals.name)
                    ]
                ]
            }
        },
        TABS.testing.name: {
            'form': form_to_vuejs(TABS.testing.formclass, ignore_choices),
            'url': TABS.testing.urls[0],
            'urls': {
                # download* below must be pairs of [key, url]. Each url
                # must return a
                # response['Content-Disposition'] = 'attachment; filename=%s'
                # url can also start with 'file:///' telling the frontend
                # to simply download the data
                'downloadRequest': [
                    [
                        'json',
                        "{0}/{1}/{1}.config.json".format(URLS.DOWNLOAD_CFG,
                                                         TABS.testing.name)
                    ],
                    [
                        'yaml',
                        "{0}/{1}/{1}.config.yaml".format(URLS.DOWNLOAD_CFG,
                                                         TABS.testing.name)
                    ]
                ],
                'downloadResponse': [
                    [
                        'json',
                        '{"file": "%s.json", "mimetype": "application/json"}' %
                        TABS.testing.name
                    ],
                    [
                        'text/csv',
                        "{0}/{1}/{1}.csv".format(URLS.DOWNLOAD_ASTEXT,
                                                 TABS.testing.name)
                    ],
                    [
                        "text/csv, decimal comma",
                        "{0}/{1}/{1}.csv".format(URLS.DOWNLOAD_ASTEXT_EU,
                                                 TABS.testing.name)
                    ]
                ]
            }
        },
        TABS.doc.name: {
            'src': URLS.DOC_PAGE
        }
    }
    if debugging:
        _configure_values_for_testing(components_props)
    return components_props


def _configure_values_for_testing(components_props: dict[str, dict[str, Any]]):
    """Set up some dict keys and subkeys so that the frontend FORM is already
    filled with test values
    """
    gsimnames = ['AkkarEtAlRjb2014', 'BindiEtAl2014Rjb', 'BooreEtAl2014',
                 'CauzziEtAl2014']
    trellisformdict = components_props['trellis']['form']
    trellisformdict['gsim']['val'] = gsimnames
    trellisformdict['imt']['val'] = ['PGA']
    trellisformdict['magnitude']['val'] = "5:7"
    trellisformdict['distance']['val'] = "10 50 100"
    trellisformdict['aspect']['val'] = 1
    trellisformdict['dip']['val'] = 60
    trellisformdict['plot']['val'] = 's'

    residualsformdict = components_props['residuals']['form']
    residualsformdict['gsim']['val'] = gsimnames
    residualsformdict['imt']['val'] = ['PGA', "SA(0.2)", "SA(1.0)", "SA(2.0)"]
    # residualsformdict['sa_period']['val'] = "0.2 1.0 2.0"
    residualsformdict['selexpr']['val'] = "magnitude > 5"
    residualsformdict['plot']['val'] = 'res'

    testingformdict = components_props['testing']['form']
    testingformdict['gsim']['val'] = gsimnames + ['AbrahamsonSilva2008']
    testingformdict['imt']['val'] = ['PGA', 'PGV', "0.2", "1.0","2.0"]
    # testingformdict['sa_period']['val'] = "0.2 1.0 2.0"

    components_props['testing']['form']['mof']['val'] = ['res', 'lh',
                                                                 # 'llh',
                                                                 # 'mllh',
                                                                 # 'edr'
                                                                 ]


def form_to_vuejs(form: Union[Type[EgsimBaseForm], EgsimBaseForm],
                  ignore_choices: Callable[[str], bool] = None) -> dict:
    """Return a dictionary of field names mapped to their widget context.
     A widget context is in turn a dict with key and value pairs used to render
     the Field as HTML component.

    :param form: EgsimBaseForm class or object (class instance)
    :param ignore_choices: callable accepting a string (field attribute name)
        and returning True or False. If False, the Field choices will not be
        loaded and the returned dict 'choices' key will be `[]`. Useful for
        avoiding time consuming long list loading
    """

    if ignore_choices is None:
        ignore_choices = lambda _: False

    form_data = {}
    field_done = set()  # keep track of Field done using their attribute name
    for field_name, field_attname in form.public_field_names.items():
        if field_attname in field_done:
            continue
        field_done.add(field_attname)
        field = form.declared_fields[field_attname]
        field_dict = field_to_dict(field, ignore_choices=ignore_choices(field_attname))
        field_dict['attrs'] = dict(field_to_htmlelement_attrs(field), name=field_name)
        field_dict['val'] = None,
        field_dict['err'] = ''
        form_data[field_name] = field_dict

    return form_data