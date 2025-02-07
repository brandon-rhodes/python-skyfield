# -*- coding: utf-8 -*-
"""An interface between Skyfield and the Python ``sgp4`` library."""

from numpy import (
    array, concatenate, identity, multiply, ones_like, repeat,
)
from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec

from .constants import AU_KM, DAY_S, T0, tau
from .functions import _T, mxm, mxv, rot_x, rot_y, rot_z
from .searchlib import _find_discrete, find_maxima
from .timelib import compute_calendar_date
from .vectorlib import VectorFunction

_identity = identity(3)

class EarthSatellite(VectorFunction):
    """An Earth satellite loaded from a TLE file and propagated with SGP4.

    An earth satellite object is a Skyfield vector function, so you can
    either call its ``at()`` method to generate its position in the sky
    or else use addition and subtraction to combine it with other
    vectors.

    Satellite parameters are generally only accurate for a week or two
    around the *epoch* of the parameters, the date for which they were
    generated, which is available as an attribute:

    ``epoch``
        A Skyfield :class:`~skyfield.timelib.Time` giving the exact
        epoch moment for these satellite orbit parameters.
    ``name``
        Satellite name

    When building a satellite, use the arguments ``line1`` and ``line2``
    to provide the two data lines from a TLE file as separate strings.
    Optional ``name`` lets you give a name to the satellite, accessible
    later through the ``name`` attribute.  ``ts`` is a
    :class:`~skyfield.timelib.Timescale` object, used to generate the
    ``epoch`` value; if it is not provided, the satellite will use a
    built in ``Timescale`` object.

    If you are interested in the catalog entry details, the SGP4 model
    parameters for a particular satellite can be accessed through its
    ``model`` attribute:

    ``model.satnum``
        The unique satellite NORAD catalog number given in the TLE file.
    ``model.classification``
        Satellite classification, or else ``'U'`` for “Unknown”
    ``model.intldesg``
        International designator
    ``model.epochyr``
        Full four-digit year of this element set's epoch moment.
    ``model.epochdays``
        Fractional days into the year of the epoch moment.
    ``model.jdsatepoch``
        Julian date of the epoch (computed from ``epochyr`` and ``epochdays``).
    ``model.ndot``
        First time derivative of the mean motion (ignored by SGP4).
    ``model.nddot``
        Second time derivative of the mean motion (ignored by SGP4).
    ``model.bstar``
        Ballistic drag coefficient B* in inverse earth radii.
    ``model.ephtype``
        Ephemeris type (ignored by SGP4 as determination now automatic)
    ``model.elnum``
        Element number
    ``model.inclo``
        Inclination in radians.
    ``model.nodeo``
        Right ascension of ascending node in radians.
    ``model.ecco``
        Eccentricity.
    ``model.argpo``
        Argument of perigee in radians.
    ``model.mo``
        Mean anomaly in radians.
    ``model.no_kozai``
        Mean motion in radians per minute.
    ``model.revnum``
        Revolution number at epoch [Revs]

    """
    center = 399
    ts = None  # see __init__()

    def __init__(self, line1, line2, name=None, ts=None):
        if ts is None:
            ts = self.ts
            if ts is None:
                from .api import load  # avoid import loop
                ts = EarthSatellite.ts = load.timescale()

        self.name = None if name is None else name.strip()
        satrec = Satrec.twoline2rv(line1, line2)
        self.model = satrec

        two_digit_year = satrec.epochyr
        if two_digit_year < 57:
            year = two_digit_year + 2000
        else:
            year = two_digit_year + 1900

        self.epoch = ts.utc(year, 1, satrec.epochdays)

        self._setup(satrec)

    def _setup(self, satrec):
        # If only I had not made __init__() specific to TLE lines, but
        # had put them in an alternate construtor instead, this would
        # simply have lived in __init__().  Alas!  I was so young then.

        self.target = -100000 - satrec.satnum

    @classmethod
    def from_satrec(cls, satrec, ts):
        """Build an EarthSatellite from a raw sgp4 Satrec object.

        This lets you provide raw numeric orbital elements instead of
        the text of a TLE set.  See :ref:`from-satrec` for detais.

        """
        self = cls.__new__(cls)
        self.model = satrec
        self.name = None

        # TODO: once sgp4 starts filling in epochyr and epochdays in
        # sgp4init(), the separate epoch code here and in __init__() can
        # be unified to always use epochyr and epochdays.
        whole, fraction = divmod(satrec.jdsatepoch, 1.0)
        year, month, day = compute_calendar_date(whole)
        day += 0.5  # convert integer Julian day into Julian date float
        self.epoch = ts.utc(year, month, day + fraction + satrec.jdsatepochF)

        self._setup(satrec)
        return self

    @classmethod
    def from_omm(cls, ts, element_dict):
        """Build an EarthSatellite from OMM text fields.

        Provide a ``ts`` timescale object, and a Python dict of OMM
        field names and values.  The timescale is used to build the
        satellite's ``.epoch`` time.

        """
        self = cls.__new__(cls)
        self.name = element_dict.get('OBJECT_NAME', None)
        self.model = satrec = Satrec()
        omm.initialize(satrec, element_dict)
        self.epoch = ts._utc_jd(satrec.jdsatepoch, satrec.jdsatepochF)
        self._setup(satrec)
        return self

    def __str__(self):
        return self.target_name

    @property
    def target_name(self):
        return '{0}{1}catalog #{2} epoch {3}'.format(
            self.name or '',
            ' ' if self.name else '',
            self.model.satnum,
            self.epoch.utc_strftime(),
        )

    def _position_and_velocity_TEME_km(self, t):
        """Return the raw true equator mean equinox (TEME) vectors from SGP4.

        Returns a tuple of NumPy arrays ``([x y z], [xdot ydot zdot])``
        expressed in kilometers and kilometers per second.  Note that we
        assume the TLE epoch to be a UTC date, per AIAA 2006-6753.

        """
        sat = self.model
        jd = t.whole
        fraction = t.tai_fraction - t._leap_seconds() / DAY_S

        if getattr(jd, 'shape', None):
            e, r, v = sat.sgp4_array(jd, fraction)
            messages = [SGP4_ERRORS[error] if error else None for error in e]
            return r.T, v.T, messages
        else:
            error, position, velocity = sat.sgp4(jd, fraction)
            message = SGP4_ERRORS[error] if error else None
            return array(position), array(velocity), message

    def ITRF_position_velocity_error(self, t):
        """Deprecated: use the TEME and ITRS frame objects instead."""
        # TODO: can we teach frame objects to figure out that the
        # transform TEME -> ITRS can not only skip the t.M rotation, but
        # can also subtract the angles of their two competing z-axis
        # rotations and call rot_z() only once instead of twice?
        rTEME, vTEME, error = self._position_and_velocity_TEME_km(t)
        rTEME /= AU_KM
        vTEME /= AU_KM
        vTEME *= DAY_S
        rITRF, vITRF = TEME_to_ITRF(t.whole, rTEME, vTEME, 0.0, 0.0,
                                    t.ut1_fraction)
        return rITRF, vITRF, error

    def _at(self, t):
        """Compute this satellite's GCRS position and velocity at time `t`."""
        r, v, error = self._position_and_velocity_TEME_km(t)
        r /= AU_KM
        v /= AU_KM
        v *= DAY_S
        R = _T(TEME.rotation_at(t))
        r = mxv(R, r)
        v = mxv(R, v)
        return r, v, None, error

    def find_events(self, topos, t0, t1, altitude_degrees=0.0):
        """Return the times at which the satellite rises, culminates, and sets.

        Searches between ``t0`` and ``t1``, which should each be a
        Skyfield :class:`~skyfield.timelib.Time` object, for passes of
        this satellite above the location ``topos`` that reach at least
        ``altitude_degrees`` above the horizon.

        Returns a tuple ``(t, events)`` whose first element is a
        :class:`~skyfield.timelib.Time` array and whose second element
        is an array of events:

        * 0 — Satellite rose above ``altitude_degrees``.
        * 1 — Satellite culminated and started to descend again.
        * 2 — Satellite fell below ``altitude_degrees``.

        Note that multiple culminations in a row are possible when,
        without setting, the satellite reaches a second peak altitude
        after descending partway down the sky from the first one.

        """
        # First, we find the moments of maximum altitude over the time
        # period.  Some of these maxima will be negative, meaning the
        # satellite failed to crest the horizon.

        ts = t0.ts
        at = (self - topos).at
        half_second = 0.5 / DAY_S
        orbits_per_minute = self.model.no_kozai / tau
        orbits_per_day = 24 * 60 * orbits_per_minute

        # Note the protection against zero orbits_per_day.
        # TODO: why isn't 3 samples per orbit enough?
        step_days = 0.05 / max(orbits_per_day, 1.0)

        # Long-period satellites might rise each day not because of
        # their own motion, but because the Earth rotates under them, so
        # check position at least each quarter-day.  We might need to
        # tighten this even further if experience someday shows it
        # missing a pass of a particular satellite.
        if step_days > 0.25:
            step_days = 0.25

        def cheat(t):
            """Avoid computing expensive values that cancel out anyway."""
            t.gast = t.tt * 0.0
            t.M = t.MT = _identity

        def altitude_at(t):
            cheat(t)
            return at(t).altaz()[0].degrees

        altitude_at.step_days = step_days
        tmax, altitude = find_maxima(t0, t1, altitude_at, half_second, 12)
        if not tmax:
            return tmax, ones_like(tmax)

        # Next, filter out the maxima that are not high enough.

        keepers = altitude >= altitude_degrees
        jdmax = tmax.tt[keepers]
        ones = ones_like(jdmax, 'uint8')

        # Finally, find the rising and setting that bracket each maximum
        # altitude.  We guess that the satellite will be back below the
        # horizon in between each pair of adjancent maxima.

        def below_horizon_at(t):
            cheat(t)
            return at(t).altaz()[0].degrees < altitude_degrees

        # The `jdo` array are the times of maxima, with their averages
        # in between them.  The start and end times are thrown in too,
        # in case a rising or setting is lingering out between a maxima
        # and the ends of our range.  Could this perhaps still miss a
        # stubborn rising or setting near the ends?
        doublets = repeat(concatenate(((t0.tt,), jdmax, (t1.tt,))), 2)
        jdo = (doublets[:-1] + doublets[1:]) / 2.0

        trs, rs = _find_discrete(t0.ts, jdo, below_horizon_at, half_second/4, 8)

        jd = concatenate((jdmax, trs.tt))
        v = concatenate((ones, rs * 2))

        i = jd.argsort()
        return ts.tt_jd(jd[i]), v[i]

class TEME(object):
    """The satellite-specific True Equator Mean Equinox frame of reference.

    This TEME frame is used to measure right ascension and declination,
    and is the reference frame of the SGP4 Earth satellite orbit model.
    It is a bit quirky.  Instead measuring right ascension from the true
    vernal equinox point, it uses the ‘mean’ equniox that considers only
    precession but not nutation (the same equinox used for Greenwich
    Mean Sidereal Time).  This made the reference frame more tractable
    for the 1970s computers that first implemented SGP4.

    Defined in AIAA 2006-6753 Appendix C.  See :ref:`reference_frames`
    for a guide to using Skyfield reference frames like this one.

    """
    @staticmethod
    def rotation_at(t):
        theta, theta_dot = theta_GMST1982(t.whole, t.ut1_fraction)
        angle = theta - t.gast / 24.0 * tau
        return mxm(rot_z(angle), t.M)

    # TODO: Are there any applications that will need us to include the
    # tiny affect on velocity of the rate of change of the difference
    # between GMST1982 and GAST?

def theta_GMST1982(jd_ut1, fraction_ut1=0.0):
    """Return the angle of Greenwich Mean Standard Time 1982 given the JD.

    This angle defines the difference between the idiosyncratic True
    Equator Mean Equinox (TEME) frame of reference used by SGP4 and the
    more standard Pseudo Earth Fixed (PEF) frame of reference.  The UT1
    time should be provided as a Julian date.  Theta is returned in
    radians, and its velocity in radians per day of UT1 time.

    From AIAA 2006-6753 Appendix C.

    """
    t = (jd_ut1 - T0 + fraction_ut1) / 36525.0
    g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    theta = (jd_ut1 % 1.0 + fraction_ut1 + g / DAY_S % 1.0) % 1.0 * tau
    theta_dot = (1.0 + dg / (DAY_S * 36525.0)) * tau
    return theta, theta_dot

_zero_zero_minus_one = array((0.0, 0.0, -1.0))
_cross120 = array((1,2,0))
_cross201 = array((2,0,1))

def _cross(a, b):
    # Nearly 4x speedup over numpy cross(). TODO: Maybe move to .functions?
    return a[_cross120] * b[_cross201] - a[_cross201] * b[_cross120]

def TEME_to_ITRF(jd_ut1, rTEME, vTEME, xp=0.0, yp=0.0, fraction_ut1=0.0):
    """Deprecated: use the TEME and ITRS frame objects instead."""
    theta, theta_dot = theta_GMST1982(jd_ut1, fraction_ut1)
    angular_velocity = multiply.outer(_zero_zero_minus_one, theta_dot)

    R = rot_z(-theta)

    if len(rTEME.shape) == 1:
        rPEF = (R).dot(rTEME)
        vPEF = (R).dot(vTEME) + _cross(angular_velocity, rPEF)
    else:
        rPEF = mxv(R, rTEME)
        vPEF = mxv(R, vTEME) + _cross(angular_velocity, rPEF)

    if xp == 0.0 and yp == 0.0:
        rITRF = rPEF
        vITRF = vPEF
    else:
        W = (rot_x(yp)).dot(rot_y(xp))
        rITRF = (W).dot(rPEF)
        vITRF = (W).dot(vPEF)
    return rITRF, vITRF
