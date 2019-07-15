import numpy as np
from pygef import nap_to_depth

WATER_PRESSURE = .0098


def grain_pressure(depth, gamma_sat, u2=None):
    """
    Determine the grain pressure over the depth.

    Parameters
    ----------
    depth : array
        Depth values in [m], returned by Pygef.
    gamma_sat : array
        Saturated gamma values in [MPa]. Determined by classification.
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

    return np.cumsum(h * gamma_sat) - u2


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
    else:
        u2 = soil_properties.depth * WATER_PRESSURE

    soil_properties["sig'"] = grain_pressure(soil_properties.depth.values, soil_properties.gamma_sat.values, u2)
    return soil_properties
