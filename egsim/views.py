'''
Created on 17 Jan 2018

@author: riccardo
'''
import os
import json
from collections import OrderedDict

from yaml.error import YAMLError

from django.http import JsonResponse
from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
from django.views.generic.base import View
from django.conf import settings

from egsim.middlewares import ExceptionHandlerMiddleware
from egsim.forms.forms import TrellisForm, GsimSelectionForm, GmdbForm, ResidualsForm, BaseForm
from egsim.core.utils import EGSIM
from egsim.core import smtk as egsim_smtk


_COMMON_PARAMS = {
    'project_name': 'eGSIM',
    'debug': settings.DEBUG,
    'menus': OrderedDict([('home', 'Home'), ('trsel', 'Gsim selection'),
                          ('trellis', 'Trellis plots'),
                          ('gmdb', 'Ground Motion database'),
                          ('residuals', 'Residuals'),
                          ('apidoc', 'API documentation'),]),
    }


# def index(request):
#     '''view for the index page. Defaults to the main view with menu="home"'''
#     return render(request, 'index.html', dict(_COMMON_PARAMS, menu='home'))


def main(request, menu):
    '''view for the main page'''
    return render(request, 'index.html', dict(_COMMON_PARAMS, menu=menu))


def home(request):
    '''view for the home page (iframe in browser)'''
    return render(request, 'home.html', _COMMON_PARAMS)


def get_tr_models(request):
    '''returns a JSON response with all tr(tectonic region) models for gsim selection'''
    return JsonResponse({'models': EGSIM.trmodels(),
                         'selected_model': list(EGSIM.trmodels().keys())[0]},
                        safe=False)

def apidoc(request):
    '''view for the home page (iframe in browser)'''
    return render(request, 'apidoc.html', dict(_COMMON_PARAMS,
                                               baseurl=request.META['HTTP_HOST']+"/query",
                                               trellis='trellis', residuals='residuals',
                                               gsimsel='gsims', test='testing',
                                               param=BaseForm.parnames(),
                                               form_trellis=TrellisForm.toHTML(),
                                               form_residuals=ResidualsForm.toHTML(),
                                               form_gsims=GsimSelectionForm.toHTML()))

def trsel(request):
    '''view returing the page forfor the gsim tectonic region
    selection'''
    return render(request, 'trsel.html', dict(_COMMON_PARAMS, post_url='../query/gsims'))


def trellis(request):
    '''view for the trellis page (iframe in browser)'''
    return render(request, 'trellis.html', dict(_COMMON_PARAMS, form=TrellisForm(),
                                                post_url='../query/trellis'))


def get_gmdbs(request):
    '''view for the residuals page (iframe in browser)'''
    return JsonResponse({'avalgmdbs': EGSIM.gmdb_names(),
                         'selectedgmdb': next(iter(EGSIM.gmdb_names()))},
                        safe=False)


def gmdb(request):
    '''view for the residuals page (iframe in browser)'''
    return render(request, 'gmdb.html', dict(_COMMON_PARAMS, form=GmdbForm(),
                                             post_url='../query/gmdbplot'))


def residuals(request):
    '''view for the residuals page (iframe in browser)'''
    return render(request, 'residuals.html', dict(_COMMON_PARAMS, form=ResidualsForm(),
                                                  post_url='../query/residuals'))


def loglikelihood(request):
    '''view for the log-likelihood page (iframe in browser)'''
    return render(request, 'loglikelihood.html', _COMMON_PARAMS)


# @api_view(['GET', 'POST'])
def get_init_params(request):  # @UnusedVariable pylint: disable=unused-argument
    """
    Returns input parameters for input selection. Called when app initializes
    """
    # FIXME: Referencing _gsims from BaseForm is quite hacky: it prevents re-calculating
    # the gsims list but there might be better soultions. NOTE: sessions need to much configuration
    # Cahce session are discouraged.:
    # https://docs.djangoproject.com/en/2.0/topics/http/sessions/#using-cached-sessions
    # so for the moment let's keep this hack
    return JsonResponse([gsim.asjson() for gsim in EGSIM.aval_gsims.values()], safe=False)


class EgsimQueryView(View):
    '''base view for every eGSIM view handling data request and returning data in response
    this is usually accomplished via a form in the web page or a POST reqeust from
    the a normal query in the standard API'''

    formclass = None
    EXCEPTION_CODE = 400
    VALIDATION_ERR_MSG = 'input validation error'

    def get(self, request):
        '''processes a get request'''
        ret = {key: val.split(',') if ',' in val else val for key, val in request.GET.items()}
        return self.response(ret)

    def post(self, request):
        '''processes a post request'''
        return self.response(request.body.decode('utf-8'))

    @classmethod
    def response(cls, obj):
        '''processes an input object `obj`, returning a response object.
        Calls `self.process` if the input is valid according to the Form's class `formclass`
        otherwise returns an appropriate json response with validation-error messages,
        or a json response with a gene'''
        try:
            form = cls.formclass.load(obj)
        except YAMLError as yerr:
            return ExceptionHandlerMiddleware.jsonerr_response(yerr, code=cls.EXCEPTION_CODE)

        if not form.is_valid():
            errors = cls.format_validation_errors(form.errors)
            return ExceptionHandlerMiddleware.jsonerr_response(cls.VALIDATION_ERR_MSG,
                                                               code=cls.EXCEPTION_CODE,
                                                               errors=errors)

        return JsonResponse(cls.process(form.cleaned_data), safe=False)

    @classmethod
    def process(cls, params):
        ''' core (abstract) method to be implemented in subclasses

        :param params: a dict of key-value paris which is assumed to be **well-formatted**:
            no check will be done on the dict: if the latter has to be validated (e.g., via
            a Django Form), **the validation should be run before this method and `params`
            should be the validated dict (e.g., as returned from `form.clean()`)**

        :return: a json-serializable object to be sent as successful response
        '''
        raise NotImplementedError()

    @classmethod
    def format_validation_errors(cls, errors):
        '''format the validation error returning the list of errors. Each item is a dict with
        keys:
             ```
             {'domain': <str>, 'message': <str>, 'code': <str>}
            ```
            :param errors: a django ErrorDict returned by the `Form.errors` property
        '''
        dic = json.loads(errors.as_json())
        errors = []
        for key, values in dic.items():
            for value in values:
                errors.append({'domain': key, 'message': value.get('message', ''),
                               'reason': value.get('code', '')})
        return errors


class TrellisView(EgsimQueryView):
    '''EgsimQueryView subclass for generating Trelli plots responses'''

    formclass = TrellisForm

    @classmethod
    def process(cls, params):
        return egsim_smtk.get_trellis(params)


class GsimsView(EgsimQueryView):

    formclass = GsimSelectionForm

    @classmethod
    def process(cls, params):
        return egsim_smtk.get_gsims(params)


class GmdbPlotView(EgsimQueryView):

    formclass = GmdbForm

    @classmethod
    def process(cls, params):
        return egsim_smtk.get_gmdbplot(params)


class ResidualsView(EgsimQueryView):

    formclass = ResidualsForm

    @classmethod
    def process(cls, params):
        return egsim_smtk.get_residuals(params)


# TESTS (FIXME: REMOVE?) #####################################################################

def test_err(request):
    raise ValueError('this is a test error!')


def trellis_test(request):
    '''view for the trellis (test) page (iframe in browser)'''
    return render(request, 'trellis_test.html', dict(_COMMON_PARAMS, form=TrellisForm(),
                                                     post_url='../query/trellis'))
