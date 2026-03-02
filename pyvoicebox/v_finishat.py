"""V_FINISHAT - Print estimated finish time of a long computation."""

import time
import sys
from datetime import datetime, timedelta


_state = {}


def v_finishat(frac, tol=None, fmt=None):
    """Print estimated finish time of a long computation.

    Parameters
    ----------
    frac : float or ndarray
        Fraction of total computation completed (0 to 1).
        0 initializes the timer.
    tol : float, optional
        Tolerance in minutes before reprinting.
    fmt : str, optional
        Format string.

    Returns
    -------
    eta : str
        Estimated finish time string.
    """
    global _state

    if fmt is None:
        fmt = 'Estimated finish at {time} ({pct:.0f}% done, {remaining} remaining)'

    if frac <= 0:
        _state = {
            'tstart': time.time(),
            'oldt': 0,
            'oldnw': 0,
        }
        return 'Unknown'

    if 'tstart' not in _state:
        _state = {
            'tstart': time.time(),
            'oldt': 0,
            'oldnw': 0,
        }

    elapsed = time.time() - _state['tstart']
    if frac > 0:
        sectogo = (1.0 / frac - 1) * elapsed
    else:
        return 'Unknown'

    now = time.time()
    finish_time = datetime.now() + timedelta(seconds=sectogo)

    if tol is None:
        tol = max(0.1 * sectogo / 60, 1)

    oldt = _state.get('oldt', 0)
    oldnw = _state.get('oldnw', 0)

    if (oldt == 0 or
            abs(finish_time.timestamp() - oldt) > tol * 60 and (now - oldnw) > 10 or
            (now - oldnw) > 600):

        _state['oldt'] = finish_time.timestamp()
        _state['oldnw'] = now

        if finish_time.date() == datetime.now().date():
            eta = finish_time.strftime('%H:%M')
        else:
            eta = finish_time.strftime('%H:%M %d-%b-%Y')

        # Format remaining time
        if sectogo >= 3600:
            remaining = f'{sectogo / 3600:.1f} hr'
        elif sectogo >= 60:
            remaining = f'{sectogo / 60:.1f} min'
        else:
            remaining = f'{sectogo:.0f} sec'

        msg = fmt.format(time=eta, pct=frac * 100, remaining=remaining)
        print(msg, file=sys.stderr)
        return eta

    return ''
