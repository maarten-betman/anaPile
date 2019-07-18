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
    df = gef.df.assign(rounded_depth=gef.df.depth.values.round(1))
    layer_table = layer_table.assign(rounded_depth=layer_table.depth_btm.values.round(1))
    soil_properties = df.merge(layer_table, how='left', on='rounded_depth')
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
    return soil_properties.drop(columns=['rounded_depth'])


def find_last_negative_friction_tipping_point(depth, soil_code):
    """
    Find the first weak layer (organic main layer) on top of the sand layers.

    Parameters
    ----------
    depth : array
        cpt's depth values in [m] sliced to the pile tip level.
    soil_code : array
        soil_code values merged with the cpt, also sliced to the pile tip level.

    Returns
    -------
    depth : float
        Depth of the bottom of the weak layers on top of the sand layers.
    """
    m = re.compile(r'[VK]')
    weak_layer = np.array(list(map(lambda x: 1 if m.search(x) else 0, soil_code)))
    idx = np.argwhere(weak_layer == 1).flatten()
    if len(idx) == 0:
        return depth[0]
    return depth[idx[-1]]


def find_positive_friction_tipping_point(depth, soil_code):
    """
    Find the first weakish (sand, with organic sublayer or organic main layer) layer on top of the sand layers.

    Parameters
    ----------
    depth : array
        cpt's depth values in [m] sliced to the pile tip level.
    soil_code : array
        soil_code values merged with the cpt, also sliced to the pile tip level.

    Returns
    -------
    depth : float
        Depth of the bottom of the weak layers on top of the sand layers.
    """
    m = re.compile(r'[ZG][kv]|[VK]')
    weakish_layer = np.array(list(map(lambda x: 1 if m.search(x) else 0, soil_code)))
    idx = np.argwhere(weakish_layer == 1).flatten()
    if len(idx) == 0:
        return depth[0]
    return depth[idx[-1]]


def determine_pile_tip_level(depth, soil_code, d_eq):
    if d_eq > 500:
        factor = 4
    else:
        factor = 8

    return find_positive_friction_tipping_point(depth, soil_code) + d_eq * factor


def find_clean_sand_layers(thickness, soil_code, depth):
    m = re.compile(r'[ZG][kv]|[VK]')
    weakish_layer = np.array(list(map(lambda x: 1 if m.search(x) else 0, soil_code)))

    # indexes of boundaries
    layer_bounds = np.argwhere(np.r_[1, np.diff(weakish_layer)] != 0).flatten()

    sand_thickness = []
    top_sand_layer = []
    btm_sand_layer = []
    for start, end in np.roll(np.repeat(layer_bounds, 2).reshape(-1, 2), 1)[1:, :]:

        if 1 in weakish_layer[start: end]:
            continue
        sand_thickness.append(thickness[start: end].sum())
        top_sand_layer.append(depth[start])

    if 0 in weakish_layer[end:]:
        sand_thickness.append(thickness[end:].sum())
        top_sand_layer.append(depth[start])
        btm_sand_layer.append(depth[end])

    return np.array(sand_thickness), np.array(top_sand_layer), np.array(btm_sand_layer)