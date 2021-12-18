"""eGSIM Django Fields"""

from fnmatch import translate
import re
import json
import shlex
from itertools import chain, repeat

import numpy as np
from openquake.hazardlib import imt
from django.core.exceptions import ValidationError

# import here all Fields used in this project to have a common namespace:
from django.forms.fields import (ChoiceField, FloatField, BooleanField,
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
    """Django CharField subclass which parses and validates string inputs given
    as array of elements in JSON (comma separated variables, with optional
    square brackets) or Unix shell (space separated variables) syntax.
    The type of the parsed elements depends on `self.parse(token)` which by
    default returns `token` but might be overridden by subclasses (see
    :class:`NArrayField`).
    As Form fields act also as validators, an object of this class can deal
    also with already parsed (e.g. via YAML) arrays
    """
    def __init__(self, *, min_count=None, max_count=None,
                 min_value=None, max_value=None, **kwargs):
        """Initialize a new ArrayField
         :param min_count: numeric or None. The minimum required count of the
             elements of the parsed array. Note that `min_length` is already
             defined in the super-class. If None (the default), parsed array
             can have any minimum length >=0.
         :param max_count: numeric or None. The maximum required count of the
             elements of the parsed array. Note that `max_length` is already
             defined in the super-class. If None (the default), parsed array
             can have any maximum length >=0.
         :param min_value: object. The minimum possible value for the
             elements of the parsed array. If None (the default) do not impose
             any minimum value. If iterable, sets the minimum required value
             element-wise (padding with None or slicing in case of lengths
             mismatch)
         :param max_value: object. Self-explanatory. Behaves the same as
             `min_value`
         :param kwargs: keyword arguments forwarded to the Django super-class.
        """
        # Parameters after “*” or “*identifier” are keyword-only parameters
        # and may only be passed used keyword arguments.
        super(ArrayField, self).__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count
        self.min_value = min_value
        self.max_value = max_value

    def to_python(self, value):
        # three scenarios: iterable: take iterable
        # non iterable: parse [value]
        # string: split value into iterable
        values = []
        is_vector = not isscalar(value)

        if value is not None:
            if not is_vector and isinstance(value, str):
                value = value.strip()
                is_vector = value[:1] == '['
                if is_vector != (value[-1:] == ']'):
                    raise ValidationError('unbalanced brackets')
                try:
                    value = json.loads(value if is_vector else "[%s]" % value)
                except Exception:  # noqa
                    try:
                        value = shlex.split(value[1:-1].strip() if is_vector
                                            else value)
                    except Exception:
                        raise ValidationError('Input syntax error')

            for val in vectorize(value):
                try:
                    vls = self.parse(val)
                except ValidationError:
                    raise
                except Exception as exc:
                    raise ValidationError("%s: %s" % (str(val), str(exc)))

                if isscalar(vls):
                    values.append(vls)
                else:
                    # force the return value to be list even if we have 1 elm:
                    is_vector = True
                    values.extend(vls)

            # check lengths:
            try:
                self.checkrange(len(values), self.min_count, self.max_count)
            except ValidationError as verr:
                # just re-format exception string and raise:
                # msg should be in the form '% not in ...', remove first '%s'
                msg = verr.message[verr.message.find(' '):]
                raise ValidationError('number of elements (%d) %s' %
                                      (len(values), msg))

            # check bounds:
            minval, maxval = self.min_value, self.max_value
            minval = [minval] * len(values) if isscalar(minval) else minval
            maxval = [maxval] * len(values) if isscalar(maxval) else maxval
            for numval, mnval, mxval in zip(values,
                                            chain(minval, repeat(None)),
                                            chain(maxval, repeat(None))):
                self.checkrange(numval, mnval, mxval)

        return values[0] if (len(values) == 1 and not is_vector) else values

    @classmethod
    def parse(cls, token):
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
            raise ValidationError('%s not in [%s, %s]' %
                                  (str(value), str(minval), str(maxval)))
        if toolow:
            raise ValidationError('%s < %s' % (str(value), str(minval)))
        if toohigh:
            raise ValidationError('%s > %s' % (str(value), str(maxval)))


class NArrayField(ArrayField):
    """ArrayField for sequences of numbers"""

    @staticmethod
    def float(val):
        """Wrapper around the built-in `float` function.
        Raises ValidationError in case of errors"""
        try:
            return float(val)
        except ValueError:
            raise ValidationError("Not a number: '%s'" % val)
        except TypeError:
            raise ValidationError(("input must be string(s) or number(s), "
                                   "not '%s'") % str(val))

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
            if not isinstance(token, str):
                raise

        # Let's try the only option left, i.e. token is a range in matlab
        # syntax, e.g.: "1:3" = [1,2,3],  "1:2:3" = [1,3]
        spl = [_.strip() for _ in token.split(':')]
        if len(spl) < 2 or len(spl) > 3:
            if ':' in token:
                raise ValidationError("Expected format '<start>:<end>' or "
                                      "'<start>:<step>:<end>', found: '%s'"
                                      % token)
            raise ValidationError("Unable to parse '%s'" % token)

        start, step, stop = \
            cls.float(spl[0]), 1 if len(spl) == 2 else \
            cls.float(spl[1]), cls.float(spl[-1])
        arange = np.arange(start, stop, step, dtype=float)

        # round numbers to max number of decimals input:
        decimals = cls.get_decimals(*spl)
        if decimals is not None:
            if round(arange[-1].item() + step, decimals) == \
                    round(stop, decimals):
                arange = np.append(arange, stop)

            arange = np.round(arange, decimals=decimals)
            if decimals == 0:
                arange = arange.astype(int)

        return arange.tolist()

    @classmethod
    def get_decimals(cls, *strings):
        """parse each string and returns the maximum number of decimals
        :param strings: a sequence of strings. Note that they do not need to
        be parsable as floats, this method searches for the dot and the
        letter 'E' (ignoring case)
        """
        decimals = 0
        try:
            for string in strings:
                idx_dec = string.find('.')
                idx_exp = string.lower().find('e')
                if idx_dec > idx_exp > -1:
                    raise ValueError()  # stop parsing
                # decimal digits after the period and until 'e' or end of string:
                dec1 = 0 if idx_dec < 0 else \
                    len(string[idx_dec+1: None if idx_exp < 0 else idx_exp])
                # decimal digits inferred from exponent:
                dec2 = 0 if idx_exp < 0 else -int(string[idx_exp+1:])
                # this string decimals are dec1 + dec2 (dec2 might be<0). Use
                # this string decimals if they are the maximum of all decimals:
                decimals = max([decimals, dec1 + dec2])
            # return 0 as we do not care for big numbers (they are int anyway)
            return decimals
        except ValueError:
            return None


class MultipleChoiceWildcardField(MultipleChoiceField):
    """Extension of Django MultipleChoiceField:
     - Accepts lists of strings or a single string
    (which will be converted to a 1-element list)
    - Accepts wildcard in strings in order to include all matching elements
    """

    def to_python(self, value):
        """convert strings with wildcards to matching elements, and calls the
        super method with the converted value. For valid wildcard characters,
        see https://docs.python.org/3.4/library/fnmatch.html
        """
        # value might be None, string, list
        if value and isinstance(value, str):
            value = [value]  # no need to call super
        else:
            # assure that value is a list/tuple:
            value = super(MultipleChoiceWildcardField, self).to_python(value)

        have_wildcard = {_: self.has_wildcards(_) for _ in value}
        if not any(have_wildcard.values()):
            return value

        # Convert wildcard strings:
        choices = [_[0] for _ in self.choices if _[0] not in value]
        new_value = []
        for val, has_wildcard in have_wildcard.items():
            if not has_wildcard:
                new_value.append(val)
                continue
            reg = self.to_regex(val)
            new_val = [c for c in choices if reg.match(c) and c not in new_value]
            new_value += new_val

        return new_value

    @staticmethod
    def has_wildcards(string):
        return '*' in string or '?' in string or ('[' in string and ']' in string)

    @staticmethod
    def to_regex(value):
        """Convert string (a unix shell string, see
        https://docs.python.org/3/library/fnmatch.html) to regexp. The latter
        will match accounting for the case (ignore case off)
        """
        return re.compile(translate(value))


class ImtField(MultipleChoiceWildcardField):
    """Field for IMT class selection. Provides a further validation for
    SA which is provided as (or with) periods (se meth:`valid_value`)
    """

    def to_python(self, value: list):
        """Coerce value to a valid IMT string"""
        value = super().to_python(value)  # assure is a list without regexp(s)
        for i, val in enumerate(value):
            # normalize val (e.g.  '0.2' -> 'SA(0.2)'):
            try:
                _ = imt.from_string(val.strip()).string
                if val != _:
                    value[i] = _
            except KeyError as kerr:
                # raised if `val` a non-imt string, e.g. 'abc'. Provide a better
                # message than simply "abc":
                raise ValidationError(f'invalid {str(kerr)}')
            except Exception as exc:
                raise ValidationError(str(exc))  # => register error in the form
        return value

    def valid_value(self, value: str):
        """Checks that value is in the choices"""
        # if we have 'SA(...)' we must validate 'SA':
        return super().valid_value('SA' if value.startswith('SA(') else value)