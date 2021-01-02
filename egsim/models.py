"""
Models for the django app

Currently they represent only read-only data from Tectonic Regionalizations
(provided by custom input files) and OpenQuake Imts, Gsims and Trt

Created on 5 Apr 2019

@author: riccardo
"""
import json
from enum import IntEnum

from django.core.management.commands import flush
from django.db import models
from django.db.models import Q
from django.db.models.aggregates import Count
from django.core import management
# primary keys are auto added if not present

from shapely.geometry import Point, shape, Polygon


class FlagEnum(IntEnum):
    """Flag-like enum **where all members are int** (e.g., they can be passed
    as int arguments to function, or combined using the the bitwise operators).

    Unlike the class:`enum.IntFlag`, this class automatically sets to each member
    value (1, 2, 4, ... see :meth:`FlagEnum.__new__`) and thus allows subclasses
    to configure each member with any tuple of additional values, usually set as
    attributes in the `__init__` method.
    """

    # Note: The builtin IntFlag enum does not entirely fit our requirements. Our
    # implementation follows https://stackoverflow.com/a/64040739
    # For further details how to access the Enum properties, see Python Enum doc

    def __new__(cls, *args):
        """Creates a new IntEnum member as int subclass with an automatic value"""
        value = 2 ** len(cls.__members__)  # create value with this member index
        member = int.__new__(cls, value)
        member._value_ = value  # mandatory otherwise superclass tries to set it
        return member

    @classmethod
    def choices(cls) -> tuple:
        """Return the `choices` argument for the associated
        :class:`models.IntegerField` (see below for applications)
        """
        return tuple((_, str(getattr(_, 'label', _.name))) for _ in cls)

    @classmethod
    def help_text(cls) -> str:
        """Return the `help_text` argument for the associated
        :class:`models.IntegerField` (see below for applications)
        """
        return " ".join("{:d} {:s}".format(*_) for _ in cls.choices())


class EntityType(FlagEnum):
    """Defines the database entities which might raise Errors to be reported
    in the :class:`Error` Model, e.g. `Error(entity_key=DbEntity.GSIM, ...)`
    """

    # Define FlagEnum members associated to a '<label>' str (see __init__):
    GSIM = 'Ground Shaking Intensity Model'
    IMT = 'Intensity Measure Type'
    TRT = 'Tectonic Region Type'

    def __init__(self, label):
        self.label = label


class Error(models.Model):
    """Model representing the Errors table. The table stores information
    during the creation of the database for diagnostic purposes
    (at the <URL>/admin address). Example: `Error(entity_key=GSIM.name, ...)`
    """
    entity_type = models.IntegerField(choices=tuple(EntityType.choices()),
                                      help_text=("The entity type. e.g. %s" %
                                                 EntityType.help_text()))
    entity_key = models.TextField(unique=True,
                                  help_text="The key uniquely identifying the "
                                            "entity, e.g. \"BindiEtAl2011\"")
    type = models.TextField(help_text="Error type, usually the class name of "
                                      "the Exception raised")
    message = models.TextField(help_text="Error message")

    def __str__(self):
        return '%s "%s": %s (%s)' % (DbEntity(self.entity_type).name,
                                     self.entity_key,
                                     self.type,
                                     self.message)


class Trt(models.Model):
    """Model representing the db table of the Tectonic Region Type(s)
    """
    key = models.TextField(unique=True)
    oq_attname = models.CharField(max_length=100, unique=True, default=None,
                                  null=True,
                                  help_text='corresponding TRT of OpenQuake '
                                            '(attribute name of the class '
                                            'openquake.hazardlib.const.TRT. '
                                            'NULL if not match is found)')
    # Instead of the JSON-like field below we should put the relationships
    # [Trt,  Trt alias(es)] in a dedicated `Model` class, but as aliases are supposed
    # to be used only in management commands, a class just for that (thus not
    # actually used in the API) seems an overhead and might also be confusing
    aliases_jsonlist = models.TextField(unique=True,
                                        help_text='Aliases/Pseudonyms of the '
                                                  'given Trt. Hidden-like field'
                                                  'that should be used in '
                                                  'management commands only')

    def __str__(self):
        return self.key


class Imt(models.Model):
    """Model representing the db table of the (OpenQuake) Intensity Measure
    Types
    """
    key = models.CharField(max_length=100, unique=True)
    needs_args = models.BooleanField(default=False, null=False)

    def __str__(self):
        return str(self.key)


#     REQUIRES_SITES_PARAMETERS = {'vs30', 'z1pt0', 'vs30measured'}
#
#     #: Required rupture parameters are magnitude, rake, dip, ztor, and width
#     #: (see table 2, page 1031)
#     REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake', 'dip', 'ztor', 'width'}
#
#     #: Required distance measures are Rrup, Rjb, Ry0 and Rx (see Table 2,
#     #: page 1031).
#     REQUIRES_DISTANCES = {'rrup', 'rjb', 'rx', 'ry0'}


class GsimParamType(FlagEnum):
    """Gsim parameter types (rupture, sites, distances)"""

    # Define FlagEnum members associated to a '<label>', <oq_attribute> (see __init__):
    SITE = 'Sites parameter', 'REQUIRES_SITES_PARAMETERS'
    RUPT = 'Rupture parameter', 'REQUIRES_RUPTURE_PARAMETERS'
    DIST = 'Distance measure', 'REQUIRES_DISTANCES'

    def __init__(self, label, oq_attname):
        self.label = label
        self.oq_attname = oq_attname


class GsimParameter(models.Model):
    key = models.TextField(null=False,
                           help_text="The parameter name (unique id)")
    type = models.IntegerField(choices=GsimParamType.choices(),
                               help_text=("The parameter type: " +
                                          GsimParamType.help_text()))

    def __str__(self):
        enum_member = GsimParamType(self.type)
        return '%s \"%s\"' % (enum_member.label, str(self.key))

    @staticmethod
    def gsim_params(gsim, *param_type: 'GsimParamType'):
        """Yield :class:`GsimParameter`s denoting the
        required parameters of  `gsim` (a GMPE instance or class).

        :param gsim: GMPE (instance or class)
        :param param_type: a list of members of :class:`GsimParamType`.
            This will return parameters of the provided types only.
            When not given, return Gsim parameters of all types
        """
        for member in param_type or GsimParamType:
            for prop_name in getattr(gsim, member.oq_attname, []):
                # (type should be int but members of this enum are also ints):
                yield GsimParameter(key=prop_name, type=member)


class Gsim(models.Model):
    """Model representing the db table of the (OpenQuake) Ground Shaking
    Intensity Models, or GMPE
    """
    # currently, the max length of the OQ gsims is 43 ...
    key = models.TextField(null=False, unique=True,
                           help_text='OpenQuake class name')
    imts = models.ManyToManyField(Imt, related_name='gsims',
                                  help_text='Intensity Measure Type(s)')
    oq_trt = models.ForeignKey(Trt, on_delete=models.CASCADE, null=True,
                               help_text='Tectonic Region type defined in '
                                         'OpenQuake')
    params = models.ManyToManyField(GsimParameter, related_name='gsims',
                                    help_text='Required parameter(s)')
    needs_args = models.BooleanField(default=False, null=False,
                                     help_text=('Whether __init__ method needs '
                                                'argument(s) in OpenQuake'))
    warning = models.TextField(default=None, null=True,
                               help_text='Optional usage warning(s)')

    def asjson(self):
        """Converts this object as a JSON-serializable tuple of strings:
        (gsim, imts, tectonic_region_type, warning) where arguments are all
        strings except 'imts' which is a tuple of strings
        """
        trt = self.oq_trt.key  # noqa
        imts = (_.key for _ in self.imts.all())  # noqa
        return self.key, tuple(imts), trt, self.warning or ''

    def __str__(self):
        return str(self.key)


class GeographicRegion(models.Model):
    """Model representing the db table of Geographic regions with associated
    Tectonic Region Type (TRT)
    """
    source_id = models.TextField(null=False,
                                 help_text="string ID of this data source "
                                           "(e.g., research project, model name)")
    geojson = models.TextField(null=False)
    trt = models.ForeignKey(Trt, on_delete=models.CASCADE, null=False,
                            help_text="tectonic region type (trt)")

    def __str__(self):
        return "Region %d (trt: %s, source_id: %s)" \
            % (self.id, self.source_id, self.trt.key)  # noqa


class GsimTrtRelation(models.Model):
    """Model representing the relationships between Trt and Gsim(s)
    """
    gsim = models.ForeignKey(Gsim, on_delete=models.CASCADE, null=False)
    trt = models.ForeignKey(Trt, on_delete=models.CASCADE, null=False)
    source_id = models.TextField(null=False,
                                 help_text="string ID of this data source "
                                           "(e.g., research project, model name)")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['gsim', 'trt', 'source_id'],
                                    name='unique(gsim,trt,source_id)')
        ]

    def __str__(self):
        return "%s selected on %s (source_id: %s)" % (str(self.gsim), str(self.trt),
                                                      str(self.source_id))


# utilities:

def empty_all():
    """Empties all tables, without removing them"""
    flush
    # https://stackoverflow.com/a/10606476
    # and
    # https://stackoverflow.com/a/2773195
    # Basically, the command below returns the database to the state it was in
    # immediately after 'syncdb' (deprecated, now 'migrate') was executed:
    management.call_command('flush', interactive=False)


def aval_gsims(asjsonlist=False):
    """Returns a list of available gsims.

    If asjsonlist=False (the default), the list elements are strings denoting
    the Gsim names (Model's attribute `gsim.key`).

    If asjsonlist is True, the list elements are json serializable tuples:
        (gsim.key, [gsim.imt1.key, .. gsim.imtN.key], gsim.trt, gsim.warning)
    where all tuple elements are strings.

    The gsims are returned sorted alphabetically
    """
    if not asjsonlist:
        return list(gsim_names())

    manager = Gsim.objects  # noqa
    # https://docs.djangoproject.com/en/2.2/ref/models/querysets/#select-related:
    queryset = manager.prefetch_related('imts').select_related('trt').\
        order_by('key')
    return [_.asjson() for _ in queryset.all()]


def aval_imts():
    """Returns a QuerySet of strings denoting the available imts"""
#     imtobjects = Imt.objects  # pylint: disable=no-member
#     # return imts mapped to at least one gsim
#     # (https://stackoverflow.com/a/12101599)
#     # for values_list, see: https://stackoverflow.com/a/37205928
#     return imtobjects.annotate(c=Count('gsims')).filter(c__gt=0).\
#         values_list('key', flat=True)
    return shared_imts(None)


def aval_trts(include_oq_name=False):
    """Returns a QuerySet of strings denoting the available Trts
    The Trts are returned sorted alphabetically by their keys"""
    trtobjects = Trt.objects  # noqa
    if include_oq_name:
        return trtobjects.order_by('key').values_list('key', 'oq_name')
    return trtobjects.order_by('key').values_list('key', flat=True)


def aval_trmodels(asjsonlist=False):
    """Returns the QueryList of models (strings) if asjsonlist is missing or
    False. Otherwise, returns QueryList of sub-lists:
        [model, type, geojson]
    where all list elements are strings (type is the associated Trt key.
    Geojson can be converted to a dict by calling as usual:
    `json.dumps(geojson)`)
    """
    trobjects = GeographicRegion.objects  # noqa
    if asjsonlist is True:
        return trobjects.values_list('model', 'type__key', 'geojson')
    return trobjects.order_by('model').values_list('model', flat=True).distinct()


def shared_imts(gsims):
    """Returns a QuerySet of strings with the the keys (=unique names) of the
    imts shared by all supplied gsims

    :param gsims: list of integers (gsim id), gsims instances, or
        strings denoting a Gsim key
    """
    # Do not expose publicly the fact that passing None returns all imts
    # defined for at least one gsim, as the user should use `aval_imts()`
    # instead
    imtobjects = Imt.objects  # noqa
    min_required_gsims = 1
    if gsims is not None:
        min_required_gsims = len(gsims)
        if not any(isinstance(_, str) for _ in gsims):
            imtobjects = imtobjects.filter(gsims__in=gsims)
        else:
            expr = or_(Q(gsims__key=_) if isinstance(_, str) else Q(gsims=_)
                       for _ in gsims)
            imtobjects = imtobjects.filter(expr)

    # return imts mapped to at least `min_required_gsims` gsim
    # (https://stackoverflow.com/a/12101599)
    # for values_list, see: https://stackoverflow.com/a/37205928
    # This assures that, if gsims is given, we return imts matching at
    # least all supplied gsims
    # (https://stackoverflow.com/a/8637972)
    return imtobjects.annotate(kount=Count('gsims')).\
        filter(kount__gte=min_required_gsims).values_list('key', flat=True)


def sharing_gsims(imts):
    """Returns a QuerySet of strings with the keys (=unique names)
    of the gsims defined for all supplied imts

    :param imts: list of integers (imt id), imts instances, or
        strings denoting an Imt key
    """
    return gsim_names(imts=imts, imts_match_all=True)


def gsim_names(gsims=None, imts=None, trts=None, tr_selector=None,
               imts_match_all=False):
    """Returns a QuerySet
    (https://docs.djangoproject.com/en/2.2/ref/models/querysets/)
    of strings denoting the Gsim names matching the given criteria.
    A Gsim name is the OpenQuake unique name (attribute `Gsim.key` in the
    associated ORM Model).
    A gsim is returned when it matches ALL the conditions
    specified by the provided arguments (missing arguments are skipped),
    which are:
    :param gsims: iterable of strings (Gsim.key) or Gsim instances, or
        both. If None (the default), no filter is applied. Otherwise,
        return Gsims whose name matches any of the provided Gsim(s)
    :param imts: iterable of strings (Imt.key) or Imt instances, or both.
        If None (the default), no filter is applied. Otherwise, return
        Gsims whose imts match any of the provided imt(s). If
        `imts_match_all`=True, return Gsims defined for (at least) all
        the provided imt(s)
    :param trts: iterable of strings (Trt.key) or Trt instances, or both.
        If None (the default), no filter is applied. Otherwise, return
        only Gsims defined for any of the provided trt(s)
    :param tr_selector: a Tectonic region selector (see class
        `TrSelector`) based on any tectonic regionalization defined on the
        database (see 'TectonicRegion' model) for filtering the search to
        a specific geographical point or rectangle. If None, nothing
        is filtered
    :param imts_match_all: boolean, ignored if `imts` is None or missing.
        When False (the default), any gsim defined for at least *one*
        of the provided imts will be taken. When True, any gsim defined
        for *all* provided imts will be taken.
    """
    # trt can be a trt instance, an int denoting the trt pkey,
    # or a string denoting the trt key field (unique)
    # in the expressions .filter(imts_in=[...]) the arguments in
    # list can be pkey (ints) or instances. We want to allow also
    # to pass imt keys (names). In this case we need to use
    # the Q function. For info see:
    # https://docs.djangoproject.com/en/2.2/topics/db/examples/many_to_one/
    # https://docs.djangoproject.com/en/2.2/topics/db/queries/

    gsimobjects = Gsim.objects  # noqa

    if gsims is not None:
        # For each gsim in gsims, gsim can be a string (the gsim key),
        # an instance or, of neither of the two, it is assumed to be an int
        # denoting the gsim.id. This is translated in this set of SQL OR
        # expressions:
        expr = or_(Q(key=_) if isinstance(_, str) else
                   Q(id=getattr(_, 'id', _)) for _ in gsims)
        gsimobjects = gsimobjects.filter(expr)

    if imts is not None:
        if not any(isinstance(_, str) for _ in imts):
            gsimobjects = gsimobjects.filter(imts__in=imts)
        else:
            expr = or_(Q(imts__key=_) if isinstance(_, str) else Q(imts=_)
                       for _ in imts)
            gsimobjects = gsimobjects.filter(expr)

        if imts_match_all:
            # https://stackoverflow.com/a/8637972
            gsimobjects = gsimobjects.annotate(num_imts=Count('imts')).\
                filter(num_imts__gte=len(imts))

    if tr_selector is not None:
        trts = tr_selector.get_trt_names(trts)
        if not trts:
            # https://docs.djangoproject.com/en/dev/ref/models/querysets/#none
            return gsimobjects.none()

    if trts is not None:
        if not any(isinstance(_, str) for _ in trts):
            gsimobjects = gsimobjects.filter(trt__in=trts)
        else:
            expr = or_(Q(trt__key=_) if isinstance(_, str) else Q(trt=_)
                       for _ in trts)
            gsimobjects = gsimobjects.filter(expr)

    gsimobjects = gsimobjects.order_by('key')

    # https://stackoverflow.com/a/37205928:
    return gsimobjects.distinct().values_list('key', flat=True)


def or_(q_expressions):
    """Concatenates the given Q expression with an 'OR'. Returns None if
    `q_expressions` (iterable) has no items"""
    expr = None
    for expr_chunk in q_expressions:
        if expr is None:
            expr = expr_chunk
        else:
            expr |= expr_chunk
    return expr


# These are the possible combinations

# trts OK
# model+point OK
# model+point+trts OK (intersection between trts and model's trts is returned)
# point NO (which model?)
# model NO (might be OK, but NO for simplicity)
# trts+point NO (which model?)
# trts+model NO (might be OK, but NO for simplicity)

class TrSelector:
    """This object allows selection of Trt (tectonic region types
    from a given tectonic regionalization (TR) and a specified point
    (or rectangle) defined on TR. The method get_trt_names will return all
    Trt(s) matching the given criteria
    """

    def __init__(self, tr_model, lon0, lat0, lon1=None, lat1=None):
        """Initializes this object. Call the method `get_trt_names` to return
        the Trt(s) matching this object.

        If specified, lon1 and lat1 will take the same values as lon0 and lat0,
        and will define a rectangle and all Trt(s) *intersecting* it will be
        returned. Otherwise, all Trt(s) *including* the point specified by
        lon0 and lat0 will be returned.

        :param tr_model: string. a tectonic regionalization model, must be
            present int the table 'TectonicRegions' under the 'model' column
        :param lon0: float, the latitude (in degrees) of the point
        :param lon0: float, the longitude (in degrees) of the point
        """
        self.tr_model = tr_model
        if lon1 is None or lat1 is None:
            self.shape = Point([lon0, lat0])
        else:
            self.shape = Polygon([(lon0, lat0), (lon0, lat1),
                                  (lon1, lat1), (lon1, lat0)])

    def get_trt_names(self, trts=None):
        tecregobjects = GeographicRegion.objects  # noqa
        geojsons = tecregobjects.filter(model=self.tr_model)
        if trts is not None:
            if not any(isinstance(_, str) for _ in trts):
                geojsons = geojsons.filter(type__in=trts)
            else:
                expr = or_(Q(type__key=_) if isinstance(_, str) else Q(type=_)
                           for _ in trts)
                geojsons = geojsons.filter(expr)

        matched_trts = {}
        ispoint = isinstance(self.shape, Point)
        for (geojson, trt) in geojsons.values_list('geojson', 'type__key'):
            if trt in matched_trts:
                continue
            tr_shape = shape(json.loads(geojson)['geometry'])
            check = tr_shape.contains if ispoint else tr_shape.intersects
            if check(self.shape):
                matched_trts[trt] = None

        return list(matched_trts.keys())
