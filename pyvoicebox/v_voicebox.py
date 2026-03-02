"""V_VOICEBOX - Global parameters for Voicebox functions.

Set/get global parameters used by other functions in the VOICEBOX toolbox.
"""

import os
import tempfile

_PP = None


def _init_defaults():
    """Initialize default parameter values."""
    return {
        'dir_temp': tempfile.gettempdir(),
        'dir_data': os.path.expanduser('~/data/speech'),
        'shorten': 'shorten',
        'flac': 'flac',
        'sfsbin': '',
        'sfssuffix': '',
        'memsize': 50e6,
        # DYPSA glottal closure identifier
        'dy_cpfrac': 0.3,
        'dy_cproj': 0.2,
        'dy_cspurt': -0.45,
        'dy_dopsp': 1,
        'dy_ewdly': 0.0008,
        'dy_ewlen': 0.003,
        'dy_ewtaper': 0.001,
        'dy_fwlen': 0.00045,
        'dy_fxmax': 500,
        'dy_fxmin': 50,
        'dy_fxminf': 60,
        'dy_gwlen': 0.0030,
        'dy_lpcdur': 0.020,
        'dy_lpcn': 2,
        'dy_lpcnf': 0.001,
        'dy_lpcstep': 0.010,
        'dy_nbest': 5,
        'dy_preemph': 50,
        'dy_spitch': 0.2,
        'dy_wener': 0.3,
        'dy_wpitch': 0.5,
        'dy_wslope': 0.1,
        'dy_wxcorr': 0.8,
        'dy_xwlen': 0.01,
    }


def v_voicebox(field=None, value=None):
    """Get or set global Voicebox parameters.

    Parameters
    ----------
    field : str, optional
        Parameter name. If None, returns all parameters.
    value : optional
        New value for the parameter.

    Returns
    -------
    Result depends on inputs:
        - No args: returns dict of all parameters
        - field only: returns value of that field (None if not found)
        - field and value: sets field and returns all parameters
    """
    global _PP
    if _PP is None:
        _PP = _init_defaults()

    if field is None:
        return dict(_PP)
    elif value is None:
        return _PP.get(field, None)
    else:
        if field not in _PP:
            raise ValueError(f"'{field}' is not a valid voicebox field name")
        _PP[field] = value
        return dict(_PP)
