from numpy import array, fmod, sin

from .constants import T0

# Much of the following code is adapted from the USNO's "novas.c".


_sequence = ['tdb', 'tt', 'ut1', 'utc']
_sequence_indexes = {name: i for i, name in enumerate(_sequence)}

def _wrap(a):
    if hasattr(a, 'shape'):
        return a
    if hasattr(a, '__len__'):
        return array(a)
    return array((a,))

class JulianDate(object):
    """Julian date.

    Attributes:

    `tdb` - Barycentric Dynamical Time
    `tt`  - Terrestrial Time
    `ut1` - Universal Time
    `utc` - Coordinated Universal Time

    """
    def __init__(self, tdb=None, tt=None, ut1=None, utc=None, delta_t=0.0):

        self.delta_t = _wrap(delta_t)

        if tdb is not None:
            self.tdb = tdb = _wrap(tdb)
            self.shape = tdb.shape
        if tt is not None:
            self.tt = tt = _wrap(tt)
            self.shape = tt.shape
        if ut1 is not None:
            self.ut1 = ut1 = _wrap(ut1)
            self.shape = ut1.shape
        if utc is not None:
            self.utc = utc = _wrap(utc)
            self.shape = utc.shape

        if not self.__dict__:
            raise ValueError('you must supply either tdb= tt= ut1= or utc=')

    def __getattr__(self, name):
        delta_t = self.delta_t
        d = self.__dict__
        i = _sequence_indexes.get(name, None)
        if i is None:
            raise AttributeError('no such attribute {!r}'.format(name))

        _TDB, _TT, _UT1, _UTC = (0, 1, 2, 3)

        tdb = d.get('tdb')
        tt = d.get('tt')
        ut1 = d.get('ut1')
        utc = d.get('utc')  # wrong! TT -(const)-> TAI -(leap secs)-> UTC?

        if i >= _TT:
            if (tt is None) and (tdb is not None):
                self.tt = tt = tdb - tdb_minus_tt(tdb) / 86400.0
                if i == _TT:
                    return tt

            if i >= _UT1:
                if (ut1 is None) and (tt is not None):
                    self.ut1 = ut1 = tt - delta_t / 86400.
                    if i == _UT1:
                        return ut1

                if i == _UTC:
                    if (ut1 is not None):
                        die
                        utc = self.utc = 6.7
                        return utc

        if (ut1 is None) and (utc is not None):
            die
            ut1 = self.ut1 = 3.4
            if i == _UT1:
                return ut1

        if (tt is None) and (ut1 is not None):
            self.tt = tt = ut1 + self.delta_t / 86400.0
            if i == _TT:
                return tt

        self.tdb = tdb = tt + tdb_minus_tt(tt) / 86400.0
        return tdb

def julian_date(year, month=1, day=1, hour=0.0):
    janfeb = month < 3
    return (day - 32075
            + 1461 * (year + 4800 - janfeb) // 4
            + 367 * (month - 2 + janfeb * 12) // 12
            - 3 * ((year + 4900 - janfeb) // 100) // 4
            ) - 0.5 + hour / 24.0

def cal_date(jd):
    """Convert Julian Day `jd` into a Gregorian year, month, day, and hour."""
    jd = jd + 0.5

    hour = jd % 1.0 * 24.0
    k = int(jd) + 68569
    n = 4 * k // 146097;

    k = k - (146097 * n + 3) // 4
    m = 4000 * (k + 1) // 1461001
    k = k - 1461 * m // 4 + 31
    month = 80 * k // 2447
    day = k - 2447 * month // 80
    k = month // 11

    month = month + 2 - 12 * k
    year = 100 * (n - 49) + m + k

    return year, month, day, hour

def tdb_minus_tt(jd_tdb):
    """Computes TT corresponding to a TDB Julian date."""

    t = (jd_tdb - T0) / 36525.0

    # Expression given in USNO Circular 179, eq. 2.6.

    return (0.001657 * sin ( 628.3076 * t + 6.2401)
          + 0.000022 * sin ( 575.3385 * t + 4.2970)
          + 0.000014 * sin (1256.6152 * t + 6.1969)
          + 0.000005 * sin ( 606.9777 * t + 4.0212)
          + 0.000005 * sin (  52.9691 * t + 0.4444)
          + 0.000002 * sin (  21.3299 * t + 5.5431)
          + 0.000010 * t * sin ( 628.3076 * t + 4.2490))
