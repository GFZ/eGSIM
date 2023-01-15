"""eGSIM Django Fields"""

from fnmatch import translate
import re
import json
import shlex
from itertools import chain, repeat
from typing import Collection, Any, Union, Sequence

import numpy as np
from openquake.hazardlib import imt
from django.core.exceptions import ValidationError

# Use this module as common namespace for all Fields. As such, DO NOT REMOVE UNUSED
# IMPORTS BELOW as they are imported throughout the project:
from django.forms.fields import (ChoiceField, FloatField, BooleanField, Field,
                                 CharField, MultipleChoiceField, FileField)
from django.forms.models import ModelChoiceField


def vectorize(value):
    """Return `value` if it is already an iterable, otherwise `[value]`.
    Note that :class:`str` and :class:`bytes` are considered scalars:
    ```
        vectorize(3) = vectorize([3]) = [3]
        vectorize('a') = vectorize(['a']) = ['a']
    ```
    """
    return [value] if isscalar(value) else value


def isscalar(value):
    """Return True if `value` is a scalar object, i.e. a :class:`str`, a
    :class:`bytes` or without the attribute '__iter__'. Example:
    ```
        isscalar(1) == isscalar('a') == True
        isscalar([1]) == isscalar(['a']) == False
    ```
    """
    return not hasattr(value, '__iter__') or isinstance(value, (str, bytes))


class ArrayField(CharField):
    """Django CharField subclass which parses and validates arrays given as
    string of text in JSON or Unix shell syntax (i.e., with space separated
    variables). An object of this class also accepts arrays given in the native
    Python type (e.g. `["a", 1]` instead of the string '["a", 1]')
    """
    def __init__(self, *, min_count=None, max_count=None,
                 min_value=None, max_value=None, **kwargs):
        """Initialize a new ArrayField

         :param min_count: numeric or None. The minimum number of elements of
            the parsed array. Raises ValueError if the array has less elements.
            None means ignore/do not check
         :param max_count: numeric or None. The maximum number of elements of
            the parsed array. See `min_count` for details
         :param min_value: object. The minimum value for the elements of the
            parsed array. None means ignore/do not check
         :param max_value: object. The maximum value for the elements of the
            parsed array. See `min_value` for details
         :param kwargs: keyword arguments forwarded to the Django super-class
        """
        # Parameters after “*” or “*identifier” are keyword-only parameters
        # and may only be passed used keyword arguments.
        super(ArrayField, self).__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count
        self.min_value = min_value
        self.max_value = max_value

    def to_python(self, value):
        if value is None:
            return None

        tokens = self.split(value) if isinstance(value, str) else value
        is_vector = not isscalar(tokens)

        values = []
        for val in self.parse_tokens(tokens if is_vector else [tokens]):
            if isscalar(val):
                values.append(val)
            else:
                is_vector = True  # force the return value to be list
                values.extend(val)

        # check lengths:
        try:
            self.checkrange(len(values), self.min_count, self.max_count)
        except ValidationError as v_err:
            # verr message starts with len(values), reformat it:
            raise ValidationError(f'number of elements {v_err.message}')

        # check bounds:
        min_v, max_v = self.min_value, self.max_value
        min_v = repeat(min_v) if isscalar(min_v) else chain(min_v, repeat(None))
        max_v = repeat(max_v) if isscalar(max_v) else chain(max_v, repeat(None))
        for val, min_val, max_val in zip(values, min_v, max_v):
            self.checkrange(val, min_val, max_val)

        return values[0] if (len(values) == 1 and not is_vector) else values

    def split(self, value: str):
        """Split the given value (str) into tokens according to json or shlex,
        in this order (json accepts arrays without brackets)
        """
        try:
            return json.loads(value)
        except Exception:  # noqa
            try:
                return shlex.split(value.strip())
            except Exception:
                raise ValidationError('Input syntax error')

    @classmethod
    def parse_tokens(cls, tokens: Collection[str]) -> Any:
        """Parse each token in `tokens` (calling self.parse(token) and yield the
        parsed token, which can be ANY value (also lists/tuples)
        """
        for val in tokens:
            try:
                yield cls.parse(val)
            except ValidationError:
                raise
            except Exception as exc:
                raise ValidationError("%s: %s" % (str(val), str(exc)))

    @classmethod
    def parse(cls, token: str) -> Any:
        """Parse token and return either an object or an iterable of objects.
        This method can safely raise any exception, if not ValidationError
        it will be wrapped into a suitable ValidationError
        """
        return token

    @staticmethod
    def checkrange(value, minval=None, maxval=None):
        """Check that the given value is in the range defined by `minval` and
        `maxval` (endpoints are included). None in `minval` and `maxval` mean:
        do not check. This method does not return any value but raises
        `ValidationError`` if value is not in the given range
        """
        toolow = (minval is not None and value < minval)
        toohigh = (maxval is not None and value > maxval)
        if toolow and toohigh:
            raise ValidationError(f'{value} not in [{minval}, {maxval}]')
        if toolow:
            raise ValidationError(f'{value} < {minval}')
        if toohigh:
            raise ValidationError(f'{value} > {maxval}')
        # if toolow and toohigh:
        #     raise ValidationError('%s not in [%s, %s]' %
        #                           (str(value), str(minval), str(maxval)))
        # if toolow:
        #     raise ValidationError('%s < %s' % (str(value), str(minval)))
        # if toohigh:
        #     raise ValidationError('%s > %s' % (str(value), str(maxval)))


class NArrayField(ArrayField):
    """ArrayField for sequences of numbers"""

    @classmethod
    def parse(cls, token):
        """Parse `token` into float.
        :param token: A python object denoting a token to be pared
        """
        # maybe already a number? try adn return
        try:
            return cls.float(token)
        except ValidationError:
            # raise if the input was not string: we surely can not deal it:
            if not isinstance(token, str) or (':' not in token):
                raise

        # token is a str with ':' in it. Let's try to parse it as matlab range:
        tokens = [_.strip() for _ in token.split(':')]
        if len(tokens) < 2 or len(tokens) > 3:
            raise ValidationError(f"Expected format '<start>:<end>' or "
                                  f"'<start>:<step>:<end>', found: {token}")

        start = cls.float(tokens[0])
        step = 1 if len(tokens) == 2 else cls.float(tokens[1])
        stop = cls.float(tokens[-1])
        rng = np.arange(start, stop, step, dtype=float)

        # round numbers to max number of decimals input:
        decimals = cls.max_decimals(tokens)
        if decimals is not None:
            if round(rng[-1].item() + step, decimals) == round(stop, decimals):
                rng = np.append(rng, stop)

            rng = np.round(rng, decimals=decimals)

            if decimals == 0:
                rng = rng.astype(int)

        return rng.tolist()

    @staticmethod
    def float(val):
        """Wrapper around the built-in `float` function.
        Raises ValidationError in case of errors"""
        try:
            return float(val)
        except ValueError:
            raise ValidationError(f"Not a number: {val}")
        except TypeError:
            raise ValidationError(f"Expected string(s) or number(s), "
                                  f"not {val.__class__}")

    @classmethod
    def max_decimals(cls, tokens: Collection[str]):
        """Return the maximum number of decimal digits necessary and sufficient
         to represent each token string without precision loss.
         Return None if the number could not be inferred.

        :param tokens: a sequence of strings representing numbers
        """
        decimals = 0
        for token in tokens:
            _decimals = cls.decimals(token)
            if _decimals is None:
                return None
            decimals = max(decimals, _decimals)
        # return 0 as we do not care for big numbers (they are int anyway)
        return decimals

    @classmethod
    def decimals(cls, token: str) -> Union[int, None]:
        """Return the number of decimal digits necessary and sufficient
         to represent the token string as float without precision loss.
         Return None if the number could not be inferred.

        :param token: a string representing a number,  e.g. '1', '11.5', '0.8e-11'
        """
        idx_dot = token.rfind('.')
        idx_exp = token.lower().find('e')
        if idx_dot > idx_exp > -1:
            return None
        # decimal digits inferred from exponent:
        dec_exp = 0
        if idx_exp > -1:
            try:
                dec_exp = -int(token[idx_exp+1:])
            except ValueError:
                return None
            token = token[:idx_exp]
        # decimal digits after the period and until 'e' or end of string:
        dec_dot = 0
        if idx_dot > -1:
            dec_dot = len(token[idx_dot+1:])
        return max(0, dec_dot + dec_exp)


class MultipleChoiceWildcardField(MultipleChoiceField):
    """Extension of Django MultipleChoiceField:
     - Accepts lists of strings or a single string
       (which will be converted to a 1-element list)
     - Accepts wildcard in strings in order to include all matching elements
    """
    # Reminder. The central validation method is `Field.clean`, which does the following:
    # def clean(self, value):
    #     value = self.to_python(value)
    #     self.validate(value)
    #       # in case of MultipleChoiceField, calls self.valid_value(v) for v in value
    #     self.run_validators(value)
    #     return value

    # override superclass default messages (provide shorter and better messages):
    default_error_messages = {
        "invalid_choice": "Value not found or misspelled: %(value)s",
        "invalid_list": "Enter a list of values",
    }

    def validate(self, value: Sequence[str]) -> None:
        """Validate the list of values. Overridden because the super
        method stops at the first validation error"""
        try:
            super().validate(value)
        except ValidationError as verr:
            # raise an error with ALL parameters invalid, not only the first one:
            if verr.code == 'invalid_choice':
                verr.params['value'] = ", ".join(v for v in value
                                                 if not self.valid_value(v))
            raise verr

    def to_python(self, value: Union[str, Sequence[Any]]) -> list[str]:
        """convert strings with wildcards to matching elements, and calls the
        super method with the converted value. For valid wildcard characters,
        see fnmatch in the Python documentation
        """
        # super call assures that value is a list/tuple and elements are strings
        value = super().to_python([value] if isinstance(value, str) else value)
        # Now store items to return as dict keys
        # (kind of overkill but we need fast search and to preserve insertion order):
        new_value = {}
        for val in value:
            if self.has_wildcards(val):
                reg = self.to_regex(val)
                for item, _ in self.choices:
                    if item not in new_value and reg.match(item):
                        new_value[item] = None  # the mapped value is irrelevant
            elif val not in new_value:
                new_value[val] = None  # the mapped value is irrelevant

        return list(new_value.keys())

    @staticmethod
    def has_wildcards(string) -> bool:
        return '*' in string or '?' in string or ('[' in string and ']' in string)

    @staticmethod
    def to_regex(wildcard_string) -> re.Pattern:
        """Convert string (a unix shell string, see
        https://docs.python.org/3/library/fnmatch.html) to regexp. The latter
        will match accounting for the case (ignore case off)
        """
        return re.compile(translate(wildcard_string))


class ImtField(MultipleChoiceWildcardField):
    """Field for IMT class selection. Provides a further validation for
    SA which is provided as (or with) periods (se meth:`valid_value`)
    """
    default_error_messages = {
        "invalid_sa_period": "Missing or invalid period: %(value)s"
    }

    def to_python(self, value: Union[str, Sequence[Any]]) -> list[str]:
        """Coerce value to a valid IMT string. Also, raise ValidationErrors from
        here, thus skipping self.validate() that would be called later and is usually
        responsible for that"""
        value = super().to_python(value)  # assure is a list without regexp(s)
        # Now normalize the IMTs. Store each normalized IMT ina dict key in order to
        # avoid duplicates whilst preserving order (Python sets dont' preserve it):
        new_val = {}
        for val in value:
            try:
                # Try to normalize the IMT (e.g. '0.2' -> 'SA(0.2)'):
                new_val[self.normalize_imt(val)] = None
            except (KeyError, ValueError):
                # val is invalid, skip (we will handle the error in `self.validate`)
                new_val[val] = None
        return list(new_val.keys())

    def validate(self, value: Sequence[str]) -> None:
        invalid_choices = []
        invalid_sa_period = []
        for val in value:
            try:
                # is IMT well written?:
                self.normalize_imt(val)  # noqa
                # is IMT supported by the program?
                if not self.valid_value(val):
                    raise KeyError()  # fallback below
            except KeyError:
                # `val` is invalid in OpenQuake, or not implemented in eGSIM (see above)
                invalid_choices.append(val)
            except ValueError:
                # val not a valid float (e.g. '0.t') or given as 'SA' (without period):
                invalid_sa_period.append(val)
        validation_errors = []
        if invalid_choices:
            validation_errors.append(ValidationError(
                self.error_messages["invalid_choice"],
                code="invalid_choice",
                params={"value": ", ".join(invalid_choices)},
            ))
        if invalid_sa_period:
            validation_errors.append(ValidationError(
                self.error_messages["invalid_sa_period"],
                code="invalid_sa_period",
                params={"value": ", ".join(invalid_sa_period)},
            ))
        if validation_errors:
            raise ValidationError(validation_errors)

    @staticmethod
    def normalize_imt(imt_string):
        """Checks and return a normalized version of the given imt as string,
        e.g. '0.1' -> 'SA(0.1)'. Raise KeyError (imt not implemented) or
        ValueError (SA period missing or invalid)"""
        return imt.from_string(imt_string.strip()).string  # noqa

    def valid_value(self, value):
        return super().valid_value('SA' if value.startswith('SA(') else value)


def get_field_docstring(field: Field, remove_html_tags=False):
    """Return a docstring from the given Form field `label` and `help_text`
    attributes. The returned string will have newlines replaced by spaces
    """
    field_label = getattr(field, 'label')
    field_help_text = getattr(field, 'help_text')

    label = (field_label or '') + \
            ('' if not field_help_text else f' ({field_help_text})')
    if label and remove_html_tags:
        # replace html tags, e.g.: "<a href='#'>X</a>" -> "X",
        # "V<sub>s30</sub>" -> "Vs30"
        _html_tags_re = re.compile('<(\\w+)(?: [^>]+|)>(.*?)</\\1>')
        # replace html characters with their content
        # (or empty str if no content):
        label = _html_tags_re.sub(r'\2', label)

    # replace newlines for safety:
    label = label.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

    return label
