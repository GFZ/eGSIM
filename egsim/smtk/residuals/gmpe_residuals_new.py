# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
"""
Module to get GMPE residuals - total, inter and intra
{'GMPE': {'IMT1': {'Total': [], 'Inter event': [], 'Intra event': []},
          'IMT2': { ... }}}
"""
from pandas import RangeIndex
from typing import Union, Iterable

from math import sqrt, ceil

import numpy as np
import pandas as pd
from openquake.hazardlib.gsim.base import GMPE
from scipy.special import erf
from scipy.stats import norm
from scipy.linalg import solve
from openquake.hazardlib import imt, const

from .. import check_gsim_list, get_gsim_name
from ..flatfile_new import EventContext, prepare_for_gsim_required_attribute, \
    MissingColumnError


def get_sa_limits(gsim: GMPE) -> Union[tuple[float, float], None]:
    pers = None
    for c in dir(gsim):
        if 'COEFFS' in c:
            pers = [sa.period for sa in getattr(gsim, c).sa_coeffs]
            break
    return (min(pers), max(pers)) if pers is not None else None


def check_sa_limits(gsim: GMPE, im: str):  # FIXME remove? is it used?
    """Return False if the period defined for the SA given in `im` does not match the
    SA limits defined for the given model (`gsim`). Return True in any other case
    (period within limits, gsim not defining any SA limit, `im` not representing SA)
    """
    if 'SA(' not in im:
        return True
    limits = get_sa_limits(gsim)
    if limits is None:
        return True
    period = imt.from_string(im).period
    return limits[0] < period < limits[1]


_EVENT_COLUMNS = ['event_id', 'event_latitude', 'event_longitude', 'event_depth',
                  'event_time']
_STATION_COLUMNS = ['station_id', 'station_latitude', 'station_longitude']


def yield_event_contexts(flatfile: pd.DataFrame) -> Iterable[EventContext]:
    """Group the flatfile by events, and yield `EventContext`s objects, one for
    each event"""
    # assure each row has a unique int id from 0 until row_count-1:
    if not isinstance(flatfile.index, RangeIndex):
        flatfile.reset_index(drop=True, inplace=True)

    # check event id column or use the event location to group events:
    try:
        # group flatfile by events. Use ev. id (_EVENT_COLUMNS[0]) or, when
        # no ID found, event spatio-temporal coordinates (_EVENT_COLUMNS[1:])
        ev_sub_flatfiles = flatfile.groupby(_EVENT_COLUMNS[0]
                                            if _EVENT_COLUMNS[0] in flatfile.columns
                                            else _EVENT_COLUMNS[1:])
    except KeyError:
        raise MissingColumnError(f'Missing flatfile event id column(s): Provide '
                                 f'"{_EVENT_COLUMNS[0]}" or ' +
                                 ", ".join('"{c}"' for c in _EVENT_COLUMNS[1:]))

    for ev_id, dfr in ev_sub_flatfiles:
        if not dfr.empty:  # for safety ...
            yield EventContext(dfr)


def get_residuals(gsims: list[str], imts: list[str],
                  flatfile: pd.DataFrame, nodal_plane_index=1,
                  component="Geometric", compute_lh=False,
                  normalise=True) -> pd.DataFrame:
    """
    Calculate the residuals for a set of ground motion records, returning a new
    DataFrame from `flatfile` with the column filled with residuals
    """
    gsims = check_gsim_list(gsims)
    flatfile = prepare_flatfile_for_residuals(flatfile, gsims.values())
    ev_columns:list[str] = _EVENT_COLUMNS[:1] if _EVENT_COLUMNS[0] in flatfile.columns \
        else _EVENT_COLUMNS[1:]
    expected = calculate_expected_motions(gsims.values(), imts, flatfile)
    new_flatfile = pd.concat([flatfile[ev_columns], expected], axis="columns")
    new_flatfile = calculate_residuals(gsims, imts, new_flatfile, normalise)
    if compute_lh:
        get_residuals_likelihood(gsims, imts, new_flatfile)
    return new_flatfile


def prepare_flatfile_for_residuals(flatfile: pd.DataFrame, gsims: Iterable[GMPE]):
    required = set()
    # code copied from openquake,hazardlib.contexts.ContextMaker.__init__:
    for gsim in gsims:
        # NB: REQUIRES_DISTANCES is empty when gsims = [FromFile]
        required.update(gsim.REQUIRES_DISTANCES | {'rrup'})
        required.update(gsim.REQUIRES_RUPTURE_PARAMETERS or [])
        required.update(gsim.REQUIRES_SITES_PARAMETERS or [])

    flatfile = flatfile.copy()
    for attr in required:
        prepare_for_gsim_required_attribute(flatfile, attr)

    return flatfile


def calculate_expected_motions(gsims: Iterable[GMPE], imt_names: list[str],
                               flatfile: pd.DataFrame) -> pd.DataFrame:
    expected:pd.DataFrame = pd.DataFrame(index=flatfile.index)
    for context in yield_event_contexts(flatfile):
        # Get the expected ground motions by event
        ev_expected = calculate_expected_motions_for_event(gsims, imt_names, context)
        # Put the above in the expected motions, index based:
        for col in ev_expected.columns:
            if col not in expected.columns:  # initialize column if needed
                expected[col] = np.nan
        # copy values:
        expected.loc[ev_expected.index, ev_expected.columns] = \
            ev_expected[ev_expected.columns]

    return expected


def calculate_expected_motions_for_event(gsims: Iterable[GMPE], imt_names: Iterable[str],
                                         ctx: EventContext) -> pd.DataFrame:
    """
    Calculate the expected ground motions from the context
    """
    expected:pd.DataFrame = pd.DataFrame(index=ctx.record_ids)
    label_of = {
        const.StdDev.TOTAL: c_labels.total,
        const.StdDev.INTER_EVENT: c_labels.inter_ev,
        const.StdDev.INTRA_EVENT: c_labels.intra_ev
    }
    # Period range for GSIM
    for gsim in gsims:
        types = gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES
        for imt_name in imt_names:
            imtx = imt.from_string(imt_name)
            if "SA(" in imt_name:
                period = imtx.period
                sa_period_limits = get_sa_limits(gsim)
                if sa_period_limits is not None and not \
                        (sa_period_limits[0] < period < sa_period_limits[1]):
                    continue
            mean, stddev = gsim.get_mean_and_stddevs(ctx, ctx, ctx, imtx, types)
            imt_name = str(imt)
            expected[column_label(gsim, imt_name, c_labels.mean)] = mean
            for std_type, values in zip(types, stddev):
                expected[column_label(gsim, imt_name, label_of[std_type])] = values
    return expected


def calculate_residuals(gsims: Iterable[str], imt_names: Iterable[str],
                        flatfile: pd.DataFrame, normalise=True) -> pd.DataFrame:
    """
    Calculate the residual terms
    """
    for gsim in gsims:
        for imtx in imt_names:
            obs = flatfile.get(imtx)
            if obs is None:
                continue
            mean = flatfile.get(column_label(gsim, imtx, c_labels.mean))
            total_stddev = flatfile.get(column_label(gsim, imtx, c_labels.total))
            if mean is None or total_stddev is None:
                continue
            obs = np.log(obs)
            flatfile[column_label(gsim, imtx, c_labels.total_res)] = \
                (obs - mean) / total_stddev
            inter_ev = flatfile.get(column_label(gsim, imtx, c_labels.inter_ev))
            intra_ev = flatfile.get(column_label(gsim, imtx, c_labels.intra_ev))
            if inter_ev is None or intra_ev is None:
                continue
            inter, intra = _get_random_effects_residuals(obs, mean, inter_ev,
                                                         intra_ev, normalise)
            flatfile[column_label(gsim, imtx, c_labels.inter_ev_res)] = inter
            flatfile[column_label(gsim, imtx, c_labels.intra_ev_res)] = intra

    return flatfile


def _get_random_effects_residuals(obs, mean, inter, intra, normalise=True):
    """
    Calculate the random effects residuals using the inter-event
    residual formula described in Abrahamson & Youngs (1992) Eq. 10
    """
    nvals = float(len(mean))
    inter_res = ((inter ** 2.) * sum(obs - mean)) /\
        (nvals * (inter ** 2.) + (intra ** 2.))
    intra_res = obs - (mean + inter_res)
    if normalise:
        return inter_res / inter, intra_res / intra
    return inter_res, intra_res


class c_labels:
    """Flatfile computed column labels (e.g. expected motion, residual)"""
    mean = "Mean"
    total = const.StdDev.TOTAL.capitalize()
    inter_ev = const.StdDev.INTER_EVENT.replace(" ", "-").capitalize()
    intra_ev = const.StdDev.INTRA_EVENT.replace(" ", "-").capitalize()
    expected_motion_column = {total, inter_ev, intra_ev}
    res = "residuals"
    total_res = total + f" {res}"
    inter_ev_res = inter_ev + f" {res}"
    intra_ev_res = intra_ev + f" {res}"
    residuals_columns = {total_res, inter_ev_res, intra_ev_res}
    lh = "LH"
    total_lh = total_res + f" {lh}"
    inter_ev_lh = inter_ev_res + f" {lh}"
    intra_ev_lh = intra_ev_res + f" {lh}"
    lh_columns = {total_res, inter_ev_res, intra_ev_res}


def column_label(gsim: Union[str, GMPE], imtx: str, c_label: str):
    """
    Return the column label for the given Gsim, imt and type (e.g. "Mean",
    "Total-residuals", see :ref:`labels`). If residuals is True, the column is
    supposed to denote a residual computed with the observed IMT stored in another column
    """
    if isinstance(gsim, GMPE):
        gsim = get_gsim_name(gsim)
    return f"{gsim} {imtx} {c_label}"


def get_computed_columns(gsims: Iterable[str],
                         imts: Iterable[str],
                         flatfile: pd.DataFrame):
    """Yield tuples from columns of `flatfile` denoting computed columns
    (expected motions, residuals) for the given `gsims` and `imts`. A computed column
    name has the form: `model + " "+ imt + " " + label` (See :ref:`column_label`).
    Each yielded tuple is:
        `column name, gsim name, imt name, residual label`
    """
    gsims = set(gsims)
    imts = set(imts)
    for col in flatfile.columns:
        chunks = col.split(" ", 2)
        if len(chunks) == 3 and chunks[0] in gsims and chunks[1] in imts:
            yield col, chunks[0], chunks[1], chunks[2]


def get_residuals_stats(gsim_names: Iterable[str], imt_names: Iterable[str],
                        flatfile: pd.DataFrame) -> \
        dict[str, dict[str, dict[str, float]]]:
    """
    Retrieve the mean and standard deviation values of the residuals

    :flatfile: the result of :ref:`get_residuals`
    """
    stats = {}
    for col, gsim, imtx, label in get_computed_columns(gsim_names, imt_names, flatfile):
        if label not in c_labels.residuals_columns:
            continue
        if gsim not in stats:
            stats[gsim] = {}
        if imtx not in stats[gsim]:
            stats[gsim][imtx] = {}
        stats[gsim][imtx]["Mean"] = flatfile[col].mean()
        stats[gsim][imtx]["Stddev"] = flatfile[col].std()
    return stats


def get_residuals_likelihood(gsims: Iterable[str], imts: Iterable[str],
                             flatfile: pd.DataFrame) -> \
        dict[str, dict[str, dict[str, float]]]:
    """
    Return the likelihood values for the residuals column found in `flatfile`
    (e.g. Total, inter- intra-event) according to Equation 9 of Scherbaum et al (2004)

    :param flatfile: a pandas DataFrame resulting from :ref:`get_residuals`.
        NOTE: the flatfile might be modified inplace (new likelihood columns added)
    """
    result = {}
    for col, gsim, imtx, label in get_computed_columns(gsims, imts, flatfile):
        if label in c_labels.residuals_columns:
            values = get_likelihood(flatfile[col])
            # get LH label corresponding to residuals:
            lh_label = c_labels.total_lh
            if label == c_labels.inter_ev_res:
                lh_label = c_labels.inter_ev_lh
            elif label == c_labels.intra_ev_res:
                lh_label = c_labels.intra_ev_lh
            # add new flatfile column:
            result[column_label(gsim, imtx, lh_label)] = values
        elif label in c_labels.lh_columns:
            values = flatfile[col]
        else:
            continue
        if gsim not in result:
            result[gsim] = {}
        if imtx not in result[gsim]:
            result[gsim][imtx] = {}
        p25, p50, p75 = np.nanpercentile(values, [25, 50, 75])
        result[f"{c_labels.lh} Median"] = p50
        result[f"{c_labels.lh} IQR"] = p75 - p25
    return result


def get_likelihood(values: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Returns the likelihood of the given values according to Equation 9 of
    Scherbaum et al (2004)
    """
    zvals = np.fabs(values)
    return 1.0 - erf(zvals / sqrt(2.))


def get_residuals_loglikelihood(gsims: Iterable[str], imts: Iterable[str],
                                flatfile: pd.DataFrame) -> \
        dict[str, dict[str, float]]:
    """
    Return the loglikelihood fit for the "Total residuals" columns found in `flatfile`
    using the loglikehood (LLH) function described in Scherbaum et al. (2009)
    Scherbaum, F., Delavaud, E., Riggelsen, C. (2009) "Model Selection in
    Seismic Hazard Analysis: An Information-Theoretic Perspective",
    Bulletin of the Seismological Society of America, 99(6), 3234-3247

    :param flatfile: a pandas DataFrame resulting from :ref:`get_residuals`
    """
    result = {}
    log_residuals = {}
    for col, gsim, imtx, label in get_computed_columns(gsims, imts, flatfile):
        if label != c_labels.total_res:
            continue
        asll = np.log2(norm.pdf(flatfile[col], 0., 1.0))
        if gsim not in result:
            result[gsim] = {}
        result[imtx] = -(1.0 / len(asll)) * np.sum(asll)
        log_residuals[gsim] = np.hstack([log_residuals[gsim], asll])
    for gsim, asll in log_residuals.items():
        result[gsim]["All"] = -(1.0 / len(asll)) * np.sum(asll)
    return result


def get_residuals_edr_values(gsims: Iterable[str], imts: Iterable[str],
                             flatfile: pd.DataFrame, bandwidth=0.01,
                             multiplier=3.0) -> \
        dict[str, dict[str, float]]:
    """
    Calculates the EDR values for each Gsim found in `flatfile` with computed residuals
    according to the Euclidean Distance Ranking method of Kale & Akkar (2013)

    Kale, O., and Akkar, S. (2013) "A New Procedure for Selecting and
    Ranking Ground Motion Predicion Equations (GMPEs): The Euclidean
    Distance-Based Ranking Method", Bulletin of the Seismological Society
    of America, 103(2A), 1069 - 1084.

    :param flatfile: a pandas DataFrame resulting from :ref:`get_residuals`
    :param float bandwidth: Discretisation width
    :param float multiplier: "Multiplier of standard deviation (equation 8 of Kale
        and Akkar)
    """
    result = {}
    for gsim in gsims:
        obs, expected, stddev = _get_edr_gsim_information(gsim, imts, flatfile)
        results = get_edr(obs, expected, stddev, bandwidth, multiplier)
        if gsim not in result:
            result[gsim] = {}
        result[gsim]["MDE Norm"] = float(results[0])
        result[gsim]["sqrt Kappa"] = float(results[1])
        result[gsim]["EDR"] = float(results[2])
    return result


def _get_edr_gsim_information(gsim: str, imts: Iterable[str], flatfile:pd.Dataframe) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the observed ground motions, expected and total standard
    deviation for the GMPE (aggregating over all IMS)

    Note: `get_residuals` must have been called on `flatfile` before this function
    """
    obs = np.array([], dtype=float)
    expected = np.array([], dtype=float)
    stddev = np.array([], dtype=float)
    for col, gsim, imtx, label in get_computed_columns([gsim], imts, flatfile):
        if label != c_labels.total:
            continue
        _stddev = flatfile[column_label(gsim, imtx, c_labels.total)]
        _obs = flatfile.get(imtx)
        _expected = flatfile.get(column_label(gsim, imtx, c_labels.mean))
        if _expected is not None and _obs is not None:
            obs = np.hstack([obs, np.log(flatfile[imtx])])
            expected = np.hstack([expected, _expected])
            stddev = np.hstack([stddev, _stddev])

    return obs, expected, stddev


def get_edr(obs: Union[np.ndarray, pd.Series],
            expected: Union[np.ndarray, pd.Series],
            stddev: Union[np.ndarray, pd.Series],
            bandwidth=0.01, multiplier=3.0) -> tuple[np.float, np.float, np.float]:
    """
    Calculated the Euclidean Distanced-Based Rank for a set of
    observed and expected values from a particular Gsim
    """
    finite = np.isfinite(obs) & np.isfinite(expected) & np.isfinite(stddev)
    if not finite.any():
        return np.nan, np.nan, np.nan
    elif not finite.all():
        obs, expected, stddev = obs[finite], expected[finite], stddev[finite]
    nvals = len(obs)
    min_d = bandwidth / 2.
    kappa = _get_edr_kappa(obs, expected)
    mu_d = obs - expected
    d1c = np.fabs(obs - (expected - (multiplier * stddev)))
    d2c = np.fabs(obs - (expected + (multiplier * stddev)))
    dc_max = ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
    num_d = len(np.arange(min_d, dc_max, bandwidth))
    mde = np.zeros(nvals)
    for iloc in range(0, num_d):
        d_val = (min_d + (float(iloc) * bandwidth)) * np.ones(nvals)
        d_1 = d_val - min_d
        d_2 = d_val + min_d
        p_1 = norm.cdf((d_1 - mu_d) / stddev) -\
            norm.cdf((-d_1 - mu_d) / stddev)
        p_2 = norm.cdf((d_2 - mu_d) / stddev) -\
            norm.cdf((-d_2 - mu_d) / stddev)
        mde += (p_2 - p_1) * d_val
    inv_n = 1.0 / float(nvals)
    mde_norm = np.sqrt(inv_n * np.sum(mde ** 2.))
    edr = np.sqrt(kappa * inv_n * np.sum(mde ** 2.))
    return mde_norm, np.sqrt(kappa), edr


def _get_edr_kappa(obs: Union[np.ndarray, pd.Series],
                   expected: Union[np.ndarray, pd.Series]) -> np.float:
    """
    Returns the correction factor kappa
    """
    mu_a = np.mean(obs)
    mu_y = np.mean(expected)
    b_1 = np.sum((obs - mu_a) * (expected - mu_y)) /\
        np.sum((obs - mu_a) ** 2.)
    b_0 = mu_y - b_1 * mu_a
    y_c = expected - ((b_0 + b_1 * obs) - obs)
    de_orig = np.sum((obs - expected) ** 2.)
    de_corr = np.sum((obs - y_c) ** 2.)
    return de_orig / de_corr


# FIXME: from here on multivariate_llh need to be fixed with GW


# Mak et al multivariate LLH functions
def get_residuals_multivariate_loglikelihood(gsim_names: list[str], imt_names: list[str],
                                          flatfile: pd.DataFrame, contexts=None):
    """
    Calculates the multivariate LLH for a set of GMPEs and IMTS according
    to the approach described in Mak et al. (2017)

    Mak, S., Clements, R. A. and Schorlemmer, D. (2017) "Empirical
    Evaluation of Hierarchical Ground-Motion Models: Score Uncertainty
    and Model Weighting", Bulletin of the Seismological Society of America,
    107(2), 949-965

    """
    raise NotImplementedError('this method is not supported and needs verification')
    result = pd.DataFrame(index=flatfile.index)
    for col, gsim, imt, label in residuals_columns(gsim_names, imt_names, flatfile):
        result[col + "-multivariate-loglikelihood"] = _get_multivariate_ll(
                    contexts, gsim, imt)
    return result


def _get_multivariate_ll(contexts, gmpe, imt):
    """
    Returns the multivariate loglikelihood, as described om equation 7 of
    Mak et al. (2017)
    """
    observations, v_mat, expected_mat, neqs, nrecs = _build_matrices(
        contexts, gmpe, imt)
    sign, logdetv = np.linalg.slogdet(v_mat)
    b_mat = observations - expected_mat
    # `solve` below needs only finite values (see doc), but unfortunately raises
    # in case. In order to silently skip non-finite values:
    # 1. Check v_mat (square matrix), by simply returning nan if any cell value
    # is nan (FIXME: handle better?):
    if not np.isfinite(v_mat).all():
        return np.nan
    # 2. Check b_mat (array) by removing finite values:
    b_finite = np.isfinite(b_mat)
    if not b_finite.all():
        if not b_finite.any():
            return np.nan
        b_mat = b_mat[b_finite]
        v_mat = v_mat[b_finite, :][:, b_finite]
    # call `solve(...check_finite=False)` because the check is already done.
    # The function shouldn't raise anymore:
    return (float(nrecs) * np.log(2.0 * np.pi) + logdetv +
            (b_mat.T.dot(solve(v_mat, b_mat, check_finite=False)))) / 2.


# The following methods are used for the MultivariateLLH function
def _build_matrices(contexts, gmpe, imtx):
    """
    Constructs the R and Z_G matrices (based on the implementation
    in the supplement to Mak et al (2017)
    """
    neqs = len(contexts)
    nrecs = sum(ctxt["Num. Sites"] for ctxt in contexts)

    r_mat = np.zeros(nrecs, dtype=float)
    z_g_mat = np.zeros([nrecs, neqs], dtype=float)
    expected_mat = np.zeros(nrecs, dtype=float)
    # Get observations
    observations = np.zeros(nrecs)
    i = 0
    # Determine the total number of records and pass the log of the
    # observations to the observations dictionary
    for ctxt in contexts:
        n_s = ctxt["Num. Sites"]
        observations[i:(i + n_s)] = np.log(ctxt["Observations"][imtx])
        i += n_s

    i = 0
    for j, ctxt in enumerate(contexts):
        if not("Intra event" in ctxt["Expected"][gmpe][imtx]) and\
                not("Inter event" in ctxt["Expected"][gmpe][imtx]):
            # Only the total sigma exists
            # Total sigma is used as intra-event sigma (from S. Mak)
            n_r = len(ctxt["Expected"][gmpe][imtx]["Total"])
            r_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imtx]["Total"]
            expected_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imtx]["Mean"]
            # Inter-event sigma is set to 0
            i += n_r
            continue
        n_r = len(ctxt["Expected"][gmpe][imtx]["Intra event"])
        r_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imtx]["Intra event"]
        # Get expected mean
        expected_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imtx]["Mean"]
        if len(ctxt["Expected"][gmpe][imtx]["Inter event"]) == 1:
            # Single inter event residual
            z_g_mat[i:(i + n_r), j] =\
                ctxt["Expected"][gmpe][imtx]["Inter event"][0]
        else:
            # inter-event residual given at a vector
            z_g_mat[i:(i + n_r), j] =\
                ctxt["Expected"][gmpe][imtx]["Inter event"]
        i += n_r

    v_mat = np.diag(r_mat ** 2.) + z_g_mat.dot(z_g_mat.T)
    return observations, v_mat, expected_mat, neqs, nrecs


# ============================================================


# GSIM_LIST = AVAILABLE_GSIMS
# GSIM_KEYS = set(GSIM_LIST)
#
# # SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
# SCALAR_IMTS = ["PGA", "PGV"]
# STDDEV_KEYS = ["Mean", "Total", "Inter event", "Intra event"]


GSIM_MODEL_DATA_TESTS = {
    "Residuals": lambda residuals, config:
        residuals.get_residual_statistics(),
    "Likelihood": lambda residuals, config: residuals.get_likelihood_values(),
    "LLH": lambda residuals, config: residuals.get_loglikelihood_values(
        config.get("LLH IMTs", [imt for imt in residuals.imts])),
    "MultivariateLLH": lambda residuals, config:
        residuals.get_multivariate_loglikelihood_values(),
    "EDR": lambda residuals, config: residuals.get_edr_values(
        config.get("bandwidth", 0.01), config.get("multiplier", 3.0))
}
