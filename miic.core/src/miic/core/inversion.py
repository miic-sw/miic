"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Nov 8, 2011
"""

# Main imports
import os
import numpy as np
from numpy.linalg import LinAlgError
from scipy.ndimage import correlate1d
from scipy import stats
from pandas import  DataFrame
from cPickle import Pickler
from copy import deepcopy

# ETS imports
try:
    BC_UI = True
    from traits.api import HasTraits, Int, Str, File, Float, Bool
    from traitsui.api import View, Item
except ImportError:
    BC_UI = False
    pass
    
# Local import
from miic.core.miic_utils import combinations, extract_stations_info, \
    dcs_check

#==============================================================================
# For backward compatibility
#==============================================================================
from miic.core.miic_utils import load_pickled_Series_DataFrame_Panel, \
    get_values_DataFrame, find_stations_name, \
    from_comb_to_stations_list, from_str_comb_to_list
    
if BC_UI:
    from miic.core.miic_utils import _load_pickled_Series_DataFrame_Panel_view, \
        _get_values_DataFrame_view
#------------------------------------------------------------------------------


def quantify_vchange_drop(vc_curve, time_win_len):
    """ Rolling interval (instead of point) derivative

    For each t0 it calculates the simple (x_t1 - x_t2) where t0 spans the
    whole timeserie vc_curve and x_t1 and x_t2 are its average on two time
    windows of length time_win_len symmetric respect to t0. At the border
    the time series is reflected so that any long term trend should not be
    altered.

    :type vc_curve: :class:`~numpy.ndarray`
    :param vc_curve: The velocity change curve (when 2D one timeseries on each
        column)
    :type time_win_len: int
    :param time_win_len: Time window length (on each side)

    :rtype: :class:`~numpy.ndarray`
    :param vdrop: Velocity change point drop on the chosen time basis
        (2*timw_win_len days)
    """
    rolling_mask = np.ones((2 * time_win_len,), dtype='f')
    rolling_mask[:time_win_len] = -1
    print rolling_mask
    vdrop = correlate1d(vc_curve, rolling_mask, axis=0, mode='reflect')\
         / time_win_len
    return vdrop


if BC_UI:
    class _quantify_vchange_drop_view(HasTraits):
        time_win_len = Int
        trait_view = View(Item('time_win_len'))


def hard_threshold_dv(dv, corr, threshold):
    """ Put dv[corr < threshould] to nan

    This function removes all the velocity_change values that are related with
    correlation values less then threshold to limit the study to those points
    in time that are reliable in terms of accuracy of the proposed model.

    :type dv: :class:`~pandas.DataFrame`
    :param dv: DataFrame that stores the velocity change
    :type corr: :class:`~pandas.DataFrame`
    :param corr: DataFrame that stores the correlation values
    :type threshold: float
    :param threshold: threshold for the correlation value to accept the
        velocity change has meaningful.

    :return: **dv**
    """
    dv.values[corr.values < threshold] = np.nan

    return dv


if BC_UI:
    class _hard_threshold_dv_view(HasTraits):
        threshold = Float(0.7)
        trait_view = View(Item('threshold'))


def resolution_matrix(n_stations):
    """ Simple resolution matrix creation

    It creates the simplest resolution matrix that describe an
    "Approximate resolution kernel by a double-delta function".
    This matrix as as much rows as possible combination pairs and
    as much columns as different stations and, on each row, just
    two values are 1 in the position of the corresponding pair of
    stations.

    :type n_stations: int
    :param n_stations: Number of stations available

    :rtype: 2D :class:`~numpy.ndarray`
    :return: **M**: Resolution matrix for the forward problem O=M*R where O
        is the vector of observed velocity change and R is the vector of real
        velocity change
    :rtype: 2D :class:`~numpy.ndarray`
    :return: **M_inv**: The term (M'*M)^(-1) * M' useful in the inverse model
        calculation
    """

    comb = combinations(n_stations, 'Full-corr')
    M = np.zeros((len(comb), n_stations), dtype='f')

    for (i, (j, k)) in enumerate(comb):
        M[i][[j, k]] = 1.0

    to_invert = np.dot(M.T, M)

    try:
        M_inv = np.dot(np.linalg.inv(to_invert), M.T)
    except LinAlgError:
        try:
            M_inv = np.dot(np.linalg.pinv(to_invert), M.T)
            M_inv[np.isnan(M_inv)] = 0
        except Exception:
            raise Exception

    return M, M_inv


if BC_UI:
    class _resolution_matrix_view(HasTraits):
        n_stations = Int
        trait_view = View(Item('n_stations'))


def invert(M, O, M_inv=None):
    """ Invert the simple forward problem O=MR

    In this case, where the matrix M is well-conditioned the inversion
    is done simple as R = (M'*M)^(-1) * M' * O

    :type M: 2D :class:`~numpy.ndarray`
    :param M: Resolution matrix for the forward problem O=M*R
    :type O: 1D :class:`~numpy.ndarray`
    :param O: Vector of observed velocity change
    :type M_inv: 2D :class:`~numpy.ndarray` (optional)
    :param M_inv: The term (M'*M)^(-1) * M' useful in the inverse
        model calculation

    :rtype:  1D :class:`~numpy.ndarray`
    :return: **R**: Vector of real velocity change
    """

    if M_inv is None:

        to_invert = np.dot(M.T, M)

        try:
            M_inv = np.dot(np.linalg.inv(to_invert), M.T)
        except LinAlgError:
            try:
                M_inv = np.dot(np.linalg.pinv(to_invert), M.T)
                M_inv[np.isnan(M_inv)] = 0
            except Exception:
                raise Exception

    R = np.dot(M_inv, O)

    return R


if BC_UI:
    class _invert_view(HasTraits):
        msg = Str("Inversion: R = (M'*M)^(-1) * M' * O ")
        trait_view = View(Item('msg', style='readonly'))


def reconstruction_error(O, M, R):
    """ Reconstruction error calculation

    For a general inverse problem of the form R = (M'*M)^(-1) * M' * O, it
    calculates the reconstruction error O - rec_O where rec_O = M * R.

    :type O: 1D :class:`~numpy.ndarray`
    :param O: Vector of observed velocity change
    :type M: 2D :class:`~numpy.ndarray`
    :param M: Resolution matrix for the forward problem O=M*R
    :type R: 1D :class:`~numpy.ndarray`
    :param R: Vector of real velocity change

    :rtype: 1D :class:`~numpy.ndarray`
    :return: **rec_error**: reconstruction error
    """
    rec_O = np.dot(M, R)
    rec_error = O - rec_O

    return rec_error


def resolution_matrix_reduced(str_comb, stations, stats_df=None):
    """ Simple resolution matrix creation

    It creates the simplest resolution matrix that describe an
    "Approximate resolution kernel by a double-delta function".
    This matrix as as much rows as possible combination pairs and
    as much columns as different stations and, on each row, just
    two values are 1 in the position of the corresponding pair of
    stations. This version accept a list of combinations in string
    form that are parsed and that defines the valid combinations.
    Its size is reduced to just those columns that are not null (all zero)
    so that the next calculations are optimized in this sense and
    to create an invertible matrix. Trace of which stations where
    adopted is kept outside this function.
    This function also returns the geographical information regarding
    the points in space involved in the inversion process.
    This is a feature that has been added to keep the interface consistent
    with other possible kernels that, in every case, should return the
    geographical information necessary to 'locate' the results of the
    inversion  process.

    :type str_comb: list of string
    :param str_comb: List of combinations to use
        (e.g. ['UV01-UV05','UV11-UV12']
    :type stations: list of string
    :param stations: Ordered list of stations name
    :type stats_df: :py:class:`~pandas.DataFrame`
    :param stats_df: Stats DataFrame

    :rtype: 2D :class:`~numpy.ndarray`
    :return: **M** : Resolution matrix for the forward problem O=M*R where O is
            the vector of observed velocity change and R is the vector of real
            velocity change.
        **points_geo_info**: DataFrame containing the geographical information
            about the points that are associated with R in the model O=M*R.
    """

    n_stations = len(stations)

    comb = from_str_comb_to_list(str_comb, stations)
    M = np.zeros((len(comb), n_stations), dtype='f')

    for (i, (j, k)) in enumerate(comb):
        M[i][[j - 1, k - 1]] = 1.0

    ss = np.sum(M, axis=0)

    M = M[:, ss != 0]

    M_inv = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)

    if stats_df is not None:
        points_geo_info = extract_stations_info(stats_df)
        return M, M_inv, points_geo_info
    else:
        return M, M_inv


def inversion_with_missed_data(dcs_dict=None,
                               dcs_filename=None,
                               threshold=0.6,
                               window='win-0',
                               save_dir=None):
    """ Simple inversion with hard thresholded velocity changes data.

    This function does the inversion to estimate the "real" velocity changes
    adopting just those observed velocity changes that are associated to
    correlation values grater then ``threshold``.

    :type dcs_dict:
    :type dcs_filename: string
    :param dcs_filename: Filename of stored dictionary that contains the
        observed velocity change dv Panel, its corresponding correlation Panel
        and the stats DataFrame with the meta-information about the stations
        where the data have been collected.
    :type corr_filename: string
    :param corr_filename: Filename of the observed correlation values stored in
        a DataFrame pickled object
    :type threshold: float
    :param threshold: Minimum correlation value necessary to assume the
        associated velocity change meaningful
    :type n_stations: int
    :param n_stations: Number of stations
    :type window: list of string
    :param window: If the velocity change were calculated on different time-lag
        windows, write here the window of interest (e.g. 'win-0')
    :type save_dir: string
    :param save_dir: If passed save the resulting dictionary in
        'adv_dict.pickle' file.

    :rtype: dictionary
    :return: **adv**: Dictionary with the following keys:
        *apparent_dv*: It is the calculated "Real" velocity change
            estimation sored in a DataFrame obj.
        *points_geo_info*: DataFrame containing the geographical information
            about all the stations involved in the inversion process.
        *residuals*: The residuals calculated, at each time, for a general
            inverse problem of the form R = (M'*M)^(-1) * M' * O, it consist of
            the reconstruction error O - rec_O where rec_O = M * R.

    .. rubric:: Notes

    ver. pandas >= 0.7 required

    """

    if dcs_dict is None:
        if dcs_filename is None:
            print "One of dcs_dict and dcs_filename MUST be not None"
            raise ValueError

        dcs_dict = load_pickled_Series_DataFrame_Panel(dcs_filename)

    check_state = dcs_check(dcs_dict)

    # Check if the dv dictionary is "correct"
    if check_state['is_incomplete']:
        print "Error: Incomplete dcs_dict"
        print "Possible errors:"
        for key in check_state:
            if key is not 'is_incomplete':
                print "%s: %s" % (key, check_state[key])
        raise ValueError

    dv = dcs_dict['dvP'][window]
    corr = dcs_dict['corrP'][window]
    stats = dcs_dict['stats']

    stations = find_stations_name(dv.columns)

    apparent_dv = DataFrame(index=dv.index, columns=stations)
    residuals = deepcopy(dv)
    points_geo_info = None

    for row_index, row in dv.iterrows():

        mask = abs(corr.ix[row_index]) > threshold

        cdv = row[mask].values

        list_stations_available = \
            from_comb_to_stations_list(dv.columns[mask], stations)

        x = len(list_stations_available)

        if x > 2:
            try:
                # It is reduntant to get the geographical information
                # at every iteration but it is kept in this way to have
                # a uniform interface that can suite even other kernels
                # respect to the simple one adopted here.
                M, _, points_geo_info = \
                    resolution_matrix_reduced(dv.columns[mask], \
                                              stations, stats)
            except np.linalg.LinAlgError:
                print "singular matrix"
                print M
                continue

            # Calculate the real dv for this specific time
            CR = invert(M, cdv)

            # Calculate the residuals
            cres = reconstruction_error(cdv, M, CR)

            # real_dv.ix[row_index, list(list_stations_available)] = CR
            apparent_dv.ix[row_index,
                           stations[list(list_stations_available)]] = CR

            residuals.ix[row_index, mask] = cres

    adv = {'apparent_dv': apparent_dv,
           'residuals': residuals}

    if points_geo_info is not None:
        adv.update({'points_geo_info': points_geo_info})

    if save_dir is not None:
        with open(os.path.join(save_dir, 'adv_dict.pickle'), 'w') as f_out:
            p = Pickler(f_out)
            p.dump(adv)

    return adv


if BC_UI:
    class _inversion_with_missed_data_macro_view(HasTraits):
        dcs_filename = File()
        threshold = Float(0.7)
        window = Str('win-0')
        save_as_file = Bool(False)
        save_file_name = File()
    
        trait_view = View(Item('dcs_filename'),
                          Item('threshold'),
                          Item('window'),
                          Item('save_as_file'),
                          Item('save_file_name', \
                               enabled_when='save_as_file==True'))
    

def compare_residuals_with_homogeneous_model(adv, dcs):
    """ Test significance of inversion model.

    Compare the residuals of a model with estimates local changes to
    a model with spatially homogeneous velocity changes. The function
    calculates the sums of squared residuals divided by their numbers
    of degrees of freedom (df). For the homogeneous model df is the
    number of measurements minus one. For the other model df is the number os
    measurements minus the number of model parameters. The function calculates
    the probability that the new model has the same variance as the homogeneous
    model. Using a confidence limit of 95% one can reject the null hypothesis
    that the variances are equal if the resulting probability is above 0.95.

    :type adv: dict of type inverted change
    :param adv: inverted velocity changes and residuals

    :type dcs: dict of type measured change
    :param dcs: velocity change, correlation and stats dictionary

    :rtype: pandas Dataframe
    :return: Dataframe containing in the columns the summed squared residuals
        for the inverted model and model0 i.e. model with homogeneous changes
        and the probability p
    """

    # Substract from every curve its mean to a void constant offset in the RSS
    dcs['dvP']['win-0'] = dcs['dvP']['win-0'] - dcs['dvP']['win-0'].mean(axis=0)
    # calculate residuals for homogeneous model
    mod0_res = dcs['dvP']['win-0'] - dcs['dvP']['win-0'].mean(axis=1)

    # copy residuals for new model
    new_res = adv['residuals']

    # number of degrees of freedom
    mod0_df = len(mod0_res.columns) - 1
    new_df = len(adv['residuals'].columns) - len(adv['apparent_dv'].columns)

    # estimate variances
    mod0_RSS = (mod0_res ** 2).sum(axis=1)
    new_RSS = (new_res ** 2).sum(axis=1)

    mod0_var = mod0_RSS / mod0_df
    new_var = new_RSS / new_df

    f_dat = mod0_var / new_var

    rv = stats.f.cdf(f_dat, mod0_df, new_df)

    compare = DataFrame({'mod0_RSS': mod0_RSS, 'new_RSS': new_RSS, 'p': rv})

    return compare



# EOF
