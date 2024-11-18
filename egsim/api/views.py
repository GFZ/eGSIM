"""Module with the views for the web API (no GUI)"""
from __future__ import annotations
from collections.abc import Callable, Iterable
from datetime import date, datetime
import re
from io import StringIO, BytesIO
from typing import Union, Type, Optional, IO, Any
from urllib.parse import quote as urlquote

import yaml
import pandas as pd
from django.http import (JsonResponse, HttpRequest, QueryDict, FileResponse,
                         HttpResponseBase)
from django.http.response import HttpResponse
from django.views.generic.base import View

from ..smtk.converters import dataframe2dict
from ..smtk.registry import gsim_info
from .forms import APIForm, EgsimBaseForm
from .forms.scenarios import PredictionsForm
from .forms.residuals import ResidualsForm


class MimeType:  # noqa
    """A collection of supported mime types (content_type in Django Response),
    loosely copied from mimetypes.types_map
    (https://docs.python.org/stable/library/mimetypes.html)
    """
    # NOTE: avoid Enums or alike, attributes below will be passed as arg `content_type`
    # to build Responses and must be pure str (subclasses NOT allowed!)
    csv = "text/csv"
    json = "application/json"
    hdf = "application/x-hdf"
    png = "image/png"
    pdf = "application/pdf"
    svg = "image/svg+xml"
    # GZIP = "application/gzip"


class EgsimView(View):

    # error codes for general client and server errors:
    CLIENT_ERR_CODE, SERVER_ERR_CODE = 400, 500

    def get(self, request: HttpRequest) -> HttpResponseBase:
        """Process a GET request and return a Django Response"""
        try:
            return self.response(request, data=self.parse_query_dict(request.GET))
        except Exception as exc:
            return self.handle_exception(exc, request)

    def post(self, request: HttpRequest) -> HttpResponseBase:
        """Process a POST request and return a Django Response"""
        try:
            if request.FILES:
                # request.content_type='multipart/form-data' (see link below for details)
                # https://docs.djangoproject.com/en/stable/ref/request-response/#django.http.HttpRequest.FILES  # noqa
                return self.response(request,
                                     data=self.parse_query_dict(request.POST),
                                     files=request.FILES)
            else:
                # request.content_type might be anything (most likely
                # 'application/json' or 'application/x-www-form-urlencoded')
                data = request.POST
                if data:  # the request contains form data
                    return self.response(request, data=self.parse_query_dict(data))
                # not form data, so assume we have JSON (stored in request.body):
                data = request.body
                return self.response(request,
                                     data=yaml.safe_load(StringIO(data.decode('utf-8'))))
        except Exception as exc:
            return self.handle_exception(exc, request)

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None) -> HttpResponseBase:
        """Return a Django HttpResponse from the given arguments extracted from a GET
        or POST request.

        :param request: the original HttpRequest
        :param data: the data extracted from the given request
        :param files: the files extracted from the given request, or None
        """
        raise NotImplementedError()

    def handle_exception(self, exc: Exception, request) -> HttpResponse:
        """Handles any exception raised in `self.get` or self.post` returning a
        server error with some info in the response body / content
        """
        msg = (
                f'Server error ({exc.__class__.__name__}) {exc}'.strip() +
                f'. Please contact the server administrator '
                f'if you think this error is due to a code bug'
        )
        return self.error_response(msg, status=self.SERVER_ERR_CODE)

    def error_response(self,
                       message: Union[str, Exception, bytes] = '',
                       **kwargs) -> HttpResponse:
        """
        Return a HttpResponse with status default to self.CLIENT_ERR_CODE
        and custom message. For custom status, provide the `status` keyword param.
        explicitly
        """
        kwargs.setdefault('status', self.CLIENT_ERR_CODE)
        return HttpResponse(content=str(message), **kwargs)

    def parse_query_dict(
            self,
            query_dict: QueryDict, *,
            nulls=("null",),
            literal_comma: Optional[set] = frozenset()
    ) -> dict[str, Union[str, list[str]]]:
        """parse the given query dict and returns a Python dict. This method parses
        GET and POST request data and can be overwritten in subclasses.

        :param query_dict: a QueryDict resulting from an `HttpRequest.POST` or
            `HttpRequest.GET`, with percent-encoded characters already decoded
        :param nulls: tuple/list/set of strings to be converted to None. Defaults
            to `("null", )`
        :param literal_comma: set (defaults to empty set) of the parameter names for
            which "," in the value has to be treated as a normal character (By default,
            a comma acts as multi-value separator)
        """
        ret = {}
        for param_name, values in query_dict.lists():
            if param_name not in literal_comma and any(',' in v for v in values):
                values = [v for val in values for v in re.split(r"\s*,\s*|\s+", val)]
            for i in range(len(values)):
                if values[i] in nulls:
                    values[i] = None
            ret[param_name] = values[0] if len(values) == 1 else values

        return ret


class NotFound(EgsimView):

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None) -> HttpResponse:
        return self.error_response(status=404)


class APIFormView(EgsimView):
    """Base view for every eGSIM API endpoint using an API Form to parse and process
    a request. Typical usage:

    1. For usage as view in `urls.py`: subclass and provide the relative `formclass`
    2. For usage inside a `views.py` function, to process data with a `APIForm`
       class `form_cls` (note: class not object):
       ```
       def myview(request):
           return APIFormView.as_view(formclass=form_cls)(request)
       ```
    """
    # The APIForm of this view, to be set in subclasses:
    formclass: Type[APIForm] = None

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None):
        """Return a HttpResponse from the given arguments. This method first creates
        a APIForm (from `self.formclass`) and puts the Form `output` into the
        returned Response body (or `Response.content`). On error, return
        an appropriate JSON response
        """
        rformat = data.pop('format', 'json')
        try:
            response_function = self.supported_formats()[rformat]
        except KeyError:
            return self.error_response(f'format: {EgsimBaseForm.ErrMsg.invalid.value}')

        form = self.formclass(data, files)
        if form.is_valid():
            obj = form.output()
            if form.is_valid():
                return response_function(self, obj, form)  # noqa
        return self.error_response(form.errors_json_data()['message'])

    @classmethod
    def supported_formats(cls) -> \
            dict[str, Callable[[APIForm, APIForm], HttpResponse]]:
        """Return a list of supported formats (content_types) by inspecting
        this class implemented methods. Each dict key is a MimeType attr name,
        mapped to a class method used to obtain the response data in that
        mime type"""
        formats = {}
        for a in dir(cls):
            if a.startswith('response_'):
                meth = getattr(cls, a)
                if callable(meth):
                    frmt = a.split('_', 1)[1]
                    if hasattr(MimeType, frmt):
                        formats[frmt] = meth
        return formats

    def response_json(self, form_output: Any, form: APIForm, **kwargs) -> JsonResponse:
        kwargs.setdefault('status', 200)
        return JsonResponse(form_output, **kwargs)


class SmtkView(APIFormView):
    """APIFormView for smtk (strong motion toolkit) output (e.g. Predictions or
    Residuals, set in the `formclass` class attribute"""

    def response_csv(self, form_output: pd.DataFrame, form: APIForm, **kwargs)\
            -> FileResponse:
        content = write_csv_to_buffer(form_output)
        content.seek(0)  # for safety
        kwargs.setdefault('content_type', MimeType.csv)
        kwargs.setdefault('status', 200)
        return FileResponse(content, **kwargs)

    def response_hdf(self, form_output: pd.DataFrame, form: APIForm, **kwargs)\
            -> FileResponse:
        content = write_hdf_to_buffer({'egsim': form_output})
        content.seek(0)  # for safety
        kwargs.setdefault('content_type', MimeType.hdf)
        kwargs.setdefault('status', 200)
        return FileResponse(content, **kwargs)

    def response_json(self, form_output: pd.DataFrame, form: APIForm, **kwargs) \
            -> JsonResponse:
        """Return a JSON response. This method is implemented for
        legacy code/tests and should be avoided whenever possible"""
        json_data = dataframe2dict(form_output, as_json=True, drop_empty_levels=True)
        kwargs.setdefault('status', 200)
        return JsonResponse(json_data, **kwargs)


class PredictionsView(SmtkView):
    """SmtkView subclass for predictions computation"""

    formclass = PredictionsForm


class ResidualsView(SmtkView):
    """SmtkView subclass for residuals computation"""

    formclass = ResidualsForm

    def response_json(self, form_output: pd.DataFrame, form: APIForm, **kwargs) \
            -> JsonResponse:
        """Return a JSON response. This method is overwritten because the JSON
        data differs if we computed measures of fit (param. `ranking=True`) or not
        """
        orient = 'dict' if form.cleaned_data['ranking'] else 'list'
        json_data = dataframe2dict(form_output, as_json=True,
                                   drop_empty_levels=True, orient=orient)
        kwargs.setdefault('status', 200)
        return JsonResponse(json_data, **kwargs)


# functions to read from BytesIO:
# (https://github.com/pandas-dev/pandas/issues/9246#issuecomment-74041497):


def write_hdf_to_buffer(frames: dict[str, pd.DataFrame], **kwargs) -> BytesIO:
    """Write in HDF format to a BytesIO the passed DataFrame(s)"""
    if any(k == 'table' for k in frames.keys()):
        raise ValueError('Key "table" invalid (https://stackoverflow.com/a/70467886)')
    with pd.HDFStore(
            "data.h5",  # apparently unused for in-memory data
            mode="w",
            driver="H5FD_CORE",  # create in-memory file
            driver_core_backing_store=0,  # prevent saving to file on close
            **kwargs) as out:
        for key, dfr in frames.items():
            out.put(key, dfr, format='table')
            # out[key] = df
        # https://www.pytables.org/cookbook/inmemory_hdf5_files.html
        return BytesIO(out._handle.get_file_image())  # noqa


def read_hdf_from_buffer(
        buffer: Union[bytes, IO], key: Optional[str] = None) -> pd.DataFrame:
    """Read from a BytesIO containing HDF data"""
    content = buffer if isinstance(buffer, bytes) else buffer.read()
    # https://www.pytables.org/cookbook/inmemory_hdf5_files.html
    with pd.HDFStore(
            "data.h5",  # apparently unused for in-memory data
            mode="r",
            driver="H5FD_CORE",  # create in-memory file
            driver_core_backing_store=0,  # for safety, just in case
            driver_core_image=content) as store:
        if key is None:
            keys = []
            for k in store.keys():
                if not any(k.startswith(_) for _ in keys):
                    keys.append(k)
                if len(keys) > 1:
                    break
            if len(keys) == 1:
                key = keys[0]
        # Note: top-level keys can be passed with or without leading slash:
        return store[key]


def write_csv_to_buffer(data: pd.DataFrame, **csv_kwargs) -> BytesIO:
    """Write in CSV format to a BytesIO the passed DataFrame(s)"""
    content = BytesIO()
    data.to_csv(content, **csv_kwargs)  # noqa
    return content


def read_csv_from_buffer(buffer: Union[bytes, IO],
                         header: Optional[Union[int, list[int]]] = None) -> pd.DataFrame:
    """
    Read from a file-like object containing CSV data.

    :param header: the header rows. Leave None or pass [0] explicitly for CSV with one
        row header, Pass a list (e.g. [0, 1, 2]) to indicate the indices of the rows
        to be used as  header (first row 0)
    """
    content = BytesIO(buffer) if isinstance(buffer, bytes) else buffer
    if header is None:
        header = [0]
    dframe = pd.read_csv(content, header=header, index_col=0)
    if header and len(header) > 1:  # multi-index, in case of "Unnamed:" column, replace:
        dframe.rename(columns=lambda c: "" if c.startswith("Unnamed:") else c,
                      inplace=True)
    return dframe


class ModelInfoView(EgsimView):

    def response(self,
                 request: HttpRequest,
                 data: dict,
                 files: Optional[dict] = None) -> HttpResponse:
        try:
            models = data['model']
            return JsonResponse({m: gsim_info(m) for m in models})
        except KeyError as exc:
            return self.error_response(exc)


# Default safe characters in `as_querystring`. Letters, digits are safe by default
# and don't need to be added. '_.-~' are safe but are added anyway for safety:
QUERY_STRING_SAFE_CHARS = "-_.~!*'()"


def as_querystring(
        data: Any,
        safe=QUERY_STRING_SAFE_CHARS,
        none_value='null',
        encoding: str = None,
        errors: str = None) -> str:
    """Return `data` as query string (URL portion after the '?' character) for GET
    requests. With the default set of input parameters, this function encodes strings
    exactly as JavaScript encodeURIComponent:
    https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent#description   # noqa
    Examples:
    ```
    as_querystring(' ') = "%20"
    as_querystring([1, 2]) = "1,2"
    as_querystring({'a': [1, 2], 'b': ' '}) = "a=1,2&b=%20"
    ```

    :param data: the data to be encoded. Containers (e.g. lists/tuple/dict) are
        allowed as long as they do not contain nested containers, because this
        would result in invalid URLs
    :param safe: string of safe characters that should not be encoded
    :param none_value: string to be used when encountering None (default 'null')
    :param encoding: used to deal with non-ASCII characters. See `urllib.parse.quote`
    :param errors: used to deal with non-ASCII characters. See `urllib.parse.quote`
    """  # noqa
    if data is None:
        return none_value
    if data is True or data is False:
        return str(data).lower()
    if isinstance(data, dict):
        return '&'.join(f'{f}={as_querystring(v)}' for f, v in data.items())
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return ','.join(f'{as_querystring(v)}' for v in data)

    if isinstance(data, (date, datetime)):
        data = data.isoformat(sep='T')
    elif not isinstance(data, (str, bytes)):
        data = str(data)
    return urlquote(data, safe=safe, encoding=encoding, errors=errors)
