import numpy as np
from pygef import nap_to_depth
import re

WATER_PRESSURE = .00981


def grain_pressure(depth, gamma_sat, gamma, u2=None):
    """
    Determine the grain pressure over the depth.

    Parameters
    ----------
    depth : array
        Depth values in [m], returned by Pygef.
    gamma_sat : array
        Saturated gamma values in [MPa]. Determined by classification.
    gamma : array
        Gamma values in [MPa]. Determined by classification.
    u2 : array
        Water pressure in [MPa]
    Returns
    -------
    grain_pressure : array
        Grain pressure in [MPa]
    """
    h = np.diff(depth)
    h = np.r_[depth[0], h]
    if u2 is None:
        u2 = depth * WATER_PRESSURE
        u2[u2 < 0] = 0

    weights = h * gamma_sat

    # correct for soil above water level
    mask = u2 == 0
    weights[mask] = h[mask] * gamma[mask]

    return np.cumsum(weights) - u2


def join_cpt_with_classification(gef, layer_table):
    """
    Merge a layer table as defined in `tests/files/layer_table.csv` with a `cpt` from Pygef.

    Parameters
    ----------
    gef : ParseGEF object.
    layer_table : DataFrame

    Returns
    -------
    merged : DataFrame
        A dataframe with all the columns and rows available in the parsed cpt result from pygef and the classification
        results from the layer_table.

    """
    soil_properties = gef.df.merge(layer_table, how='left', left_on='depth', right_on='depth_btm')
    soil_properties = soil_properties.fillna(method='bfill').dropna()

    if 'u2' in gef.df.columns:
        u2 = soil_properties.u2.values
    elif gef.groundwater_level is not None:
        water_depth = nap_to_depth(gef.zid, gef.groundwater_level)
        u2 = (soil_properties.depth - water_depth) * WATER_PRESSURE
        u2[u2 < 0] = 0
    else:
        u2 = soil_properties.depth * WATER_PRESSURE

    soil_properties["grain_pressure"] = grain_pressure(soil_properties.depth.values,
                                                       soil_properties.gamma_sat.values, soil_properties.gamma.values,
                                                       u2)
    return soil_properties


def find_last_negative_friction_tipping_point(depth, soil_codes):
    """
    Find the first weak layer on top of the sand layers.

    Parameters
    ----------
    depth : array
        cpt's depth values in [m] sliced to the pile tip level.
    soil_codes : array
        soil_code values merged with the cpt, also sliced to the pile tip level.

    Returns
    -------
    depth : float
        Depth of the bottom of the weak layers on top of the sand layers.
    """
    m = re.compile(r'[VK]')
    weak_layer = np.array(list(map(lambda x: 1 if m.search(x) else 0, soil_codes)))
    idx = np.argwhere(weak_layer == 1).flatten()
    if len(idx) == 0:
        return depth[0]
    return depth[idx[-1]]


def find_positive_friction_tipping_point(depth, soil_codes):
    """
    Find the first weakish layer on top of the sand layers.

    Parameters
    ----------
    depth : array
        cpt's depth values in [m] sliced to the pile tip level.
    soil_codes : array
        soil_code values merged with the cpt, also sliced to the pile tip level.

    Returns
    -------
    depth : float
        Depth of the bottom of the weak layers on top of the sand layers.
    """
    m = re.compile(r'[ZG][kv]|[VK]')
    weakish_layer = np.array(list(map(lambda x: 1 if m.search(x) else 0, soil_codes)))
    idx = np.argwhere(weakish_layer == 1).flatten()
    if len(idx) == 0:
        return depth[0]
    return depth[idx[-1]]


def determine_pile_tip_level(depth, soil_codes, d_eq):
    if d_eq > 500:
        factor = 4
    else:
        factor = 8

    return find_positive_friction_tipping_point(depth, soil_codes) + d_eq * factor