"""V_HOSTIPINFO - Get host name and IP info using Python equivalents."""

import socket


def v_hostipinfo(m=''):
    """Get host name and IP address info.

    Parameters
    ----------
    m : str, optional
        Mode string. 'h' for hostname, 'i' for IP address.

    Returns
    -------
    info : str
        Hostname or IP address.
    """
    if not m or 'h' in m:
        return socket.gethostname()
    if 'i' in m:
        try:
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            return '127.0.0.1'
    return socket.gethostname()
