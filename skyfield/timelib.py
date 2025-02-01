# -*- coding: utf-8 -*-
import datetime as dt_module
import re
import sys
from collections import namedtuple
from datetime import date, datetime, timedelta
from numpy import (
    array, concatenate, cos, float64, int64, isnan, isinf, linspace,
    nan, ndarray, nonzero, pi, rollaxis, searchsorted, sin, where, zeros_like,

)
from time import strftime, struct_time
from ._compatibility import interp
from .constants import ASEC2RAD, B1950, DAY_S, T0, tau
from .curvelib import Splines, build_spline_given_ends
from .descriptorlib import reify
from .earthlib import sidereal_time, earth_rotation_angle
from .framelib import ICRS_to_J2000 as B
from .functions import (A, mxm, mxmxm, load_bundled_npy, rot_x, rot_y, rot_z,
                        _to_array, _reconcile)
from .nutationlib import (
    build_nutation_matrix, equation_of_the_equinoxes_complimentary_terms,
    iau2000a_radians, mean_obliquity,
)
from .precessionlib import compute_precession

_EMPTY_TUPLE = ()  # since we use the value as a sentinel
DAY_US = 86400000000.0
GREGORIAN_START = 2299161
GREGORIAN_START_ENGLAND = 2361222
_OLD_PYTHON = sys.version_info < (2, 7)
def _cat(*args): return concatenate(args, axis=1)

CalendarTuple = namedtuple('CalendarTuple', 'year month day hour minute second')

class CalendarArray(ndarray):
    @property
    def year(self): return self[0]
    @property
    def month(self): return self[1]
    @property
    def day(self): return self[2]
    @property
    def hour(self): return self[3]
    @property
    def minute(self): return self[4]
    @property
    def second(self): return self[5]

if hasattr(dt_module, 'timezone'):
    utc = dt_module.timezone.utc
else:
    class UTC(dt_module.tzinfo):
        'UTC'
        zero = timedelta(0)
        def utcoffset(self, dt):
            return self.zero
        def tzname(self, dt):
            return 'UTC'
        def dst(self, dt):
            return self.zero

    utc = UTC()

# Much of the following code is adapted from the USNO's "novas.c".

_time_zero = dt_module.time(tzinfo=utc)
MONTH_NAMES = A['0', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

tt_minus_tai = array(32.184 / DAY_S)

class Timescale(object):
    """The data necessary to express dates in different timescales.

    A `Timescale` provides time objects with the data tables they need
    to translate between different time scales: the schedule of UTC leap
    seconds, and the value of ∆T over time.  Most programs create a
    single `Timescale` which they use to build their `Time` objects:

    >>> from skyfield.api import load
    >>> ts = load.timescale()
    >>> t = ts.utc(1980, 3, 1, 9, 30)
    >>> t
    <Time tt=2444299.896425741>

    See :ref:`downloading-timescale-files` if you are interested in
    checking how recent the data is in the files loaded by the
    timescale.

    """
    polar_motion_table = None

    def __init__(self, delta_t_recent, leap_dates, leap_offsets):
        if callable(delta_t_recent):
            # Let a caller completely override our approach to ∆T by
            # passing a function of their own.
            self.delta_t_function = delta_t_recent
        else:
            self.delta_t_table = delta_t_recent  # so users can see it
            self.delta_t_function = build_delta_t(delta_t_recent)

        self.leap_dates, self.leap_offsets = leap_dates, leap_offsets
        self.J2000 = Time(self, float64(T0))
        self.B1950 = Time(self, float64(B1950))
        self.julian_calendar_cutoff = None

        # Our internal leap-second table has three columns:
        #
        # _leap_utc      Integer UTC seconds since JD 0 w/leap seconds missing.
        # _leap_offsets  Integer offset between UTC and TAI.
        # _leap_tai      Integer TAI seconds since JD 0.

        is_legacy_table = isinf(leap_dates[-1])
        if is_legacy_table:
            leap_dates = leap_dates[2:-1]
            leap_offsets = leap_offsets[3:]

        one_zero = [[-1,0]]
        self._leap_utc = (leap_dates[:,None] * DAY_S + one_zero).flatten()
        self._leap_offsets = (leap_offsets[:,None] + one_zero).flatten()
        self._leap_tai = self._leap_utc + self._leap_offsets

    def now(self):
        """Return the current date and time as a `Time` object.

        For the return value to be correct, your operating system time
        and timezone settings must be set so that the Python Standard
        Library constructor `datetime.now(timezone.utc)` returns a
        correct UTC date and time.

        """
        return self.from_datetime(datetime.now(utc))

    def from_datetime(self, datetime):
        """Return a `Time` for a Python ``datetime``.

        The ``datetime`` must be “timezone-aware”: it must have a time
        zone object as its ``tzinfo`` attribute instead of ``None``.

        .. versionadded:: 1.24

        """
        return self._utc(_datetime_to_utc_tuple(datetime))

    def from_datetimes(self, datetime_list):
        """Return a `Time` for a list of Python ``datetime`` objects.

        The ``datetime`` objects must each be “timezone-aware”: they
        must each have a time zone object as their ``tzinfo`` attribute
        instead of ``None``.

        .. versionadded:: 1.24

        """
        tuples = (_datetime_to_utc_tuple(d) for d in datetime_list)
        return self._utc(array(value) for value in zip(*tuples))

    def utc(self, year, month=1, day=1, hour=0, minute=0, second=0.0):
        """Build a `Time` from a UTC `calendar date`.

        .. versionadded:: 1.24
           Passing a Python ``datetime`` or a list of datetimes as the
           first argument has been deprecated (and was never supported
           for the other time scale methods).  Instead, use the methods
           :meth:`~skyfield.timelib.Timescale.from_datetime()` and
           :meth:`~skyfield.timelib.Timescale.from_datetimes()`.

        """
        # TODO: someday deprecate passing datetime objects here, as
        # there are now separate constructors for them.
        if isinstance(year, datetime):
            return self.from_datetime(year)
        if isinstance(year, date):
            return self.from_datetime(datetime.combine(year, _time_zero))
        if hasattr(year, '__len__') and isinstance(year[0], datetime):
            return self.from_datetimes(year)

        tup = year, month, day, hour, minute, second
        return self._utc(tup)

    def _utc(self, tup):
        # Build a Time from a UTC tuple, carefully preserving its exact
        # second number in the Time's hidden TAI seconds field.
        year, month, day, hour, minute, second = tup
        a = _to_array
        cutoff = self.julian_calendar_cutoff

        # Figure out exactly the TAI second number.
        seconds = (julian_day(a(year), a(month), a(day), cutoff) - 0.5) * DAY_S
        seconds, sfr = divmod(seconds, 1.0)  # in case there were any fractions
        seconds += interp(seconds, self._leap_utc, self._leap_offsets)
        more = a(hour) * 3600.0 + a(minute) * 60.0 + a(second)
        seconds2, sfr = divmod(sfr + more, 1.0)
        seconds += seconds2

        # For the other timescales, use the usual Julian date + fraction.
        whole, fraction = divmod(seconds, DAY_S)
        fraction += sfr
        fraction /= DAY_S

        t = Time(self, whole, fraction + tt_minus_tai)
        t.tai_fraction = fraction
        t._tai_seconds = seconds, sfr
        return t

    def _jd(self, year, month, day, hour, minute, second):
        a = _to_array
        cutoff = self.julian_calendar_cutoff
        whole = julian_day(a(year), a(month), a(day), cutoff) - 0.5
        fraction = (a(second) + a(minute) * 60.0 + a(hour) * 3600.0) / DAY_S
        return _reconcile(whole, fraction)

    def _cal(self, whole, fraction):
        return calendar_tuple(whole, fraction, self.julian_calendar_cutoff)

    def _strftime(self, format, jd, fraction, seconds_bump=None):
        # Python forces an unhappy choice upon us: either use the faster
        # time.strftime() and lose support for '%f', or use the slower
        # datetime.strftime() and crash if years are negative.  We take the
        # first option, but then patch '%f' support back in by secretly
        # passing the microseconds string as the time zone name.  After all,
        # the routines supported by this function never use time zones.
        # What could go wrong?

        offset, ms = _strftime_offset_seconds(format)
        fraction = fraction + offset / DAY_S
        year, month, day, hour, minute, second = self._cal(jd, fraction)
        z = year * 0

        # TODO: will this always produce the same whole number that
        # calendar_tuple() produces internally?  Or should we make a private
        # version of calendar_tuple() that returns it to us for use here?
        weekday = (fraction + 0.5 + _to_array(jd)).astype(int) % 7

        if _format_uses_day_of_year(format):
            start_of_year = julian_day(year, 1, 1, self.julian_calendar_cutoff)
            yday = (jd + fraction + 1.5 - start_of_year).astype(int)
        else:
            yday = z

        if ms:
            format = format[:ms.start()] + '%Z' + format[ms.end():]
            second = (second * 1e6).astype(int)
            second, usec = divmod(second, 1000000)
            if seconds_bump is not None:
                second += seconds_bump
            if getattr(jd, 'ndim', 0):
                u = ['%06d' % u for u in usec]
                tup = (year, month, day, hour, minute, second,
                       weekday, yday, z, u)
                return [strftime(format, struct_time(t)) for t in zip(*tup)]
            u = '%06d' % usec
            tup = year, month, day, hour, minute, second, weekday, yday, z, u
            return strftime(format, struct_time(tup))
        else:
            second = second.astype(int)
            if seconds_bump is not None:
                second += seconds_bump
            tup = year, month, day, hour, minute, second, weekday, yday, z
            if getattr(jd, 'ndim', 0):
                return [strftime(format, item) for item in zip(*tup)]
            return strftime(format, tup)

    def _utc_jd(self, whole, fraction):
        # Switch from days to seconds.
        seconds = whole * DAY_S
        seconds2, fraction_s = divmod(fraction * DAY_S, 1.0)
        seconds += seconds2

        # Add an integer number of leap seconds.
        seconds += interp(seconds, self._leap_utc, self._leap_offsets)

        # Switch back to days.
        whole, fraction = divmod(seconds, DAY_S)
        fraction += fraction_s
        fraction /= DAY_S
        return self.tai_jd(whole, fraction)

    def tai(self, year=None, month=1, day=1, hour=0, minute=0, second=0.0,
            jd=None):
        """Build a `Time` from an International Atomic Time `calendar date`.

        .. versionadded:: 1.6
           Passing a Julian date with ``jd=`` has been deprecated;
           instead, use :meth:`~skyfield.timelib.Timescale.tai_jd()`.

        """
        if jd is not None:
            return self.tai_jd(jd)  # deprecate someday
        whole, fraction = self._jd(year, month, day, hour, minute, second)
        t = Time(self, whole, fraction + tt_minus_tai)
        t.tai_fraction = fraction
        return t

    def tai_jd(self, jd, fraction=None):
        """Build a `Time` from an International Atomic Time Julian date."""
        jd, fraction = _normalize_jd_and_fraction(jd, fraction)
        t = Time(self, jd, fraction + tt_minus_tai)
        t.tai_fraction = fraction
        return t

    def tt(self, year=None, month=1, day=1, hour=0, minute=0, second=0.0,
           jd=None):
        """Build a `Time` from a Terrestrial Time `calendar date`.

        .. versionadded:: 1.6
           Passing a Julian date with ``jd=`` has been deprecated;
           instead, use :meth:`~skyfield.timelib.Timescale.tt_jd()`.

        """
        if jd is not None:
            return self.tt_jd(jd)  # deprecate someday
        whole, fraction = self._jd(year, month, day, hour, minute, second)
        return Time(self, whole, fraction)

    def tt_jd(self, jd, fraction=None):
        """Build a `Time` from a Terrestrial Time Julian date."""
        jd, fraction = _normalize_jd_and_fraction(jd, fraction)
        return Time(self, jd, fraction)

    def J(self, year):
        """Build a `Time` from a Terrestrial Time Julian year or array.

        Julian years are convenient uniform periods of exactly 365.25
        days of Terrestrial Time, centered on 2000 January 1 12h TT =
        Julian year 2000.0.

        """
        tt = _to_array(year) * 365.25 + 1721045.0
        return Time(self, tt, 0.0)

    def tdb(self, year=None, month=1, day=1, hour=0, minute=0, second=0.0,
            jd=None):
        """Build a `Time` from a Barycentric Dynamical Time `calendar date`.

        .. versionadded:: 1.6
           Passing a Julian date with ``jd=`` has been deprecated;
           instead, use :meth:`~skyfield.timelib.Timescale.tdb_jd()`.

        """
        if jd is not None:
            return self.tdb_jd(jd)  # deprecate someday
        whole, fraction = self._jd(year, month, day, hour, minute, second)
        jd = whole + fraction  # TODO: why do tests break if we pass separately
        return Time(self, jd, - tdb_minus_tt(jd) / DAY_S)

    def tdb_jd(self, jd, fraction=None):
        """Build a `Time` from a Barycentric Dynamical Time Julian date."""
        jd, fraction = _normalize_jd_and_fraction(jd, fraction)
        t = Time(self, jd, fraction - tdb_minus_tt(jd, fraction) / DAY_S)
        t.tdb_fraction = fraction
        return t

    def ut1(self, year=None, month=1, day=1, hour=0, minute=0, second=0.0,
            jd=None):
        """Build a `Time` from a UT1 Universal Time `calendar date`.

        .. versionadded:: 1.6
           Passing a Julian date with ``jd=`` has been deprecated;
           instead, use :meth:`~skyfield.timelib.Timescale.ut1_jd()`.

        """
        if jd is None:  # TODO: deprecate the jd parameter to this method
            whole, fraction = self._jd(year, month, day, hour, minute, second)
            jd = whole + fraction  # TODO: can we pass high precision on?
        return self.ut1_jd(jd)

    def ut1_jd(self, jd):
        """Build a `Time` from a UT1 Universal Time Julian date."""
        ut1 = _to_array(jd)

        # Estimate TT = UT1, to get a rough Delta T estimate.
        tt_approx = ut1
        delta_t_approx = self.delta_t_function(tt_approx)

        # Use the rough Delta T to make a much better estimate of TT,
        # then generate an even better Delta T.
        tt_approx = ut1 + delta_t_approx / DAY_S
        delta_t_approx = self.delta_t_function(tt_approx)

        # We can now estimate TT with an error of < 1e-9 seconds within
        # 10 centuries of either side of the present; for details, see:
        # https://github.com/skyfielders/astronomy-notebooks
        # and look for the notebook "error-in-timescale-ut1.ipynb".
        delta_t_approx /= DAY_S
        t = Time(self, ut1, delta_t_approx)
        t.ut1_fraction = 0.0 * ut1
        return t

    def from_astropy(self, t):
        """Build a Skyfield `Time` from an AstroPy time object."""
        return self.tt(jd=t.tt.jd)

    def linspace(self, t0, t1, num=50):
        """Return ``num`` times spaced uniformly between ``t0`` to ``t1``.

        This routine is named after, and powered by, the NumPy routine
        `linspace()`_.

        .. _linspace(): https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

        """
        whole0 = t0.whole
        frac0 = t0.tt_fraction
        whole1 = t1.whole
        frac1 = t1.tt_fraction
        return Time(
            self,
            linspace(whole0, whole1, num),
            linspace(frac0, frac1, num),
        )

class Time(object):
    """A single moment in history, or an array of several moments.

    Skyfield programs don’t usually instantiate this class directly, but
    instead build time objects using one of the timescale methods listed
    at `timescale-summary`.  If you do attempt the low-level operation
    of building a time object yourself, either leave ``tt_fraction`` at
    its default value of ``None`` — in which case Skyfield will assume
    the fraction is zero — or provide a ``tt_fraction`` array that has
    exactly the same dimensions as your ``tt`` array.

    """
    def __init__(self, ts, tt, tt_fraction=None):
        if tt_fraction is None:
            tt_fraction = zeros_like(tt)
        self.ts = ts
        self.whole = tt
        self.tt_fraction = tt_fraction
        self.shape = getattr(tt, 'shape', _EMPTY_TUPLE)

    def __len__(self):
        shape = self.shape
        if shape is _EMPTY_TUPLE:
            raise TypeError('this is a single Time, not an array')
        return shape[0]

    def __repr__(self):
        size = getattr(self.tt, 'size', -1)
        if size > 3:
            rstr = '<Time tt=[{0} ... {1}] len={2}>'
            return rstr.format(self.tt[0], self.tt[-1], size)
        else:
            return ('<Time tt={0}>'.format(self.tt)
                    .replace('[ ', '[').replace('  ', ' '))

    def __getitem__(self, index):
        if self.shape is _EMPTY_TUPLE:
            raise TypeError('this is a single Time, not an array')
        # TODO: also copy cached matrices?
        t = Time(self.ts, self.whole[index], self.tt_fraction[index])
        d = self.__dict__
        for name in 'tai_fraction', 'tdb_fraction', 'ut1_fraction':
            value = d.get(name)
            if value is not None:
                setattr(t, name, value[index])
        return t

    def astimezone(self, tz):
        """Convert to a Python ``datetime`` in a particular timezone ``tz``.

        If this time is an array, then an array of datetimes is returned
        instead of a single value.

        """
        dt, leap_second = self.astimezone_and_leap_second(tz)
        return dt

    def astimezone_and_leap_second(self, tz):
        """Convert to a Python ``datetime`` and leap second in a timezone.

        Convert this time to a Python ``datetime`` and a leap second::

            dt, leap_second = t.astimezone_and_leap_second(tz)

        The argument ``tz`` should be a ``datetime`` compatible
        timezone.

        The leap second value is provided because a Python ``datetime``
        can only number seconds ``0`` through ``59``, but leap seconds
        have a designation of at least ``60``.  The leap second return
        value will normally be ``0``, but will instead be ``1`` if the
        date and time are a UTC leap second.  Add the leap second value
        to the ``second`` field of the ``datetime`` to learn the real
        name of the second.

        If this time is an array, then an array of ``datetime`` objects
        and an array of leap second integers is returned, instead of a
        single value each.

        """
        dt, leap_second = self.utc_datetime_and_leap_second()
        normalize = getattr(tz, 'normalize', None)
        if self.shape and normalize is not None:
            dt = array([normalize(d.astimezone(tz)) for d in dt])
        elif self.shape:
            dt = array([d.astimezone(tz) for d in dt])
        elif normalize is not None:
            dt = normalize(dt.astimezone(tz))
        else:
            dt = dt.astimezone(tz)
        return dt, leap_second

    def toordinal(self):
        """Return the proleptic Gregorian ordinal of the UTC date.

        This method makes Skyfield `Time` objects compatible with Python
        `datetime`_ objects, which also provide a ``toordinal()``
        method.  Thanks to this method, a `Time` can often be used
        directly as a coordinate for a plot.

        """
        whole, fraction, is_leap_second = self._utc_seconds(0.0)
        return (whole - 148731076800.0 + fraction) / DAY_S

    def utc_datetime(self):
        """Convert to a Python ``datetime`` in UTC.

        If this time is an array, then a list of datetimes is returned
        instead of a single value.

        """
        dt, leap_second = self.utc_datetime_and_leap_second()
        return dt

    def utc_datetime_and_leap_second(self):
        """Convert to a Python ``datetime`` in UTC, plus a leap second value.

        Convert this time to a `datetime`_ object and a leap second::

            dt, leap_second = t.utc_datetime_and_leap_second()

        The leap second value is provided because a Python ``datetime``
        can only number seconds ``0`` through ``59``, but leap seconds
        have a designation of at least ``60``.  The leap second return
        value will normally be ``0``, but will instead be ``1`` if the
        date and time are a UTC leap second.  Add the leap second value
        to the ``second`` field of the ``datetime`` to learn the real
        name of the second.

        If this time is an array, then an array of ``datetime`` objects
        and an array of leap second integers is returned, instead of a
        single value each.

        """
        year, month, day, hour, minute, second = self._utc_tuple(0.5e-6)
        micro = (second * 1e6).astype(int)
        second, micro = divmod(micro, 1000000)
        leap_second = second // 60
        second -= leap_second  # fit within limited bounds of Python datetime
        if self.shape:
            zone = [utc] * self.shape[0]
            argsets = zip(year, month, day, hour, minute, second, micro, zone)
            d = []
            a = d.append
            try:  # placed outside the loop for efficiency
                for args in argsets:
                    a(datetime(*args))
            except ValueError as e:
                _upgrade_datetime_exception(args, e)
                raise
            d = array(d)
        else:
            args = year, month, day, hour, minute, second, micro, utc
            try:
                d = datetime(*args)
            except ValueError as e:
                _upgrade_datetime_exception(args, e)
                raise
        return d, leap_second

    def utc_iso(self, delimiter='T', places=0):
        """Convert to an ISO 8601 string like ``2014-01-18T01:35:38Z`` in UTC.

        If this time is an array of dates, then a sequence of strings is
        returned instead of a single string.

        """
        # "places" used to be the 1st argument, so continue to allow an
        # integer in that spot.  TODO: deprecate this in Skyfield 2.0
        # and remove it in 3.0.
        if isinstance(delimiter, int):
            places = delimiter
            delimiter = 'T'

        if places:
            power_of_ten = 10 ** places
            offset = 0.5 / power_of_ten
            year, month, day, hour, minute, second = self._utc_tuple(offset)
            second, fraction = divmod(second, 1.0)
            fraction *= power_of_ten
            format = '%04d-%02d-%02d{0}%02d:%02d:%02d.%0{1}dZ'.format(
                delimiter, places)
            args = (year, month, day, hour, minute, second, fraction)
        else:
            format = '%04d-%02d-%02d{0}%02d:%02d:%02dZ'.format(delimiter)
            args = self._utc_tuple(0.5)

        if self.shape:
            return [format % tup for tup in zip(*args)]
        else:
            return format % args

    def utc_jpl(self):
        """Convert to a string like ``A.D. 2014-Jan-18 01:35:37.5000 UTC``.

        Returns a string for this date and time in UTC, in the format
        used by the JPL HORIZONS system.  If this time is an array of
        dates, then a sequence of strings is returned instead of a
        single string.

        """
        year, month, day, hour, minute, second = self._utc_tuple(0.00005)
        second, fraction = divmod(second, 1.0)
        fraction *= 1e4
        bc = year < 1
        year = abs(year - bc)
        era = where(bc, 'B.C.', 'A.D.')
        format = '%s %04d-%s-%02d %02d:%02d:%02d.%04d UTC'
        args = (era, year, MONTH_NAMES[month], day,
                hour, minute, second, fraction)

        if self.shape:
            return [format % tup for tup in zip(*args)]
        else:
            return format % args

    def utc_strftime(self, format='%Y-%m-%d %H:%M:%S UTC'):
        """Format the UTC time using a Python datetime formatting string.

        This calls Python’s ``time.strftime()`` to format the date and
        time.  A single string is returned or else a whole array of
        strings, depending on whether this time object is an array.
        The most commonly used formats are:

        * ``%Y`` four-digit year, ``%y`` two-digit year
        * ``%m`` month number, ``%B`` name, ``%b`` abbreviation
        * ``%d`` day of month
        * ``%H`` hour
        * ``%M`` minute
        * ``%S`` second
        * ``%A`` day of week, ``%a`` its abbreviation

        The ``%Z`` and ``%z`` formats are not supported; instead, simply
        use the literal characters ``'UTC'`` in your format string.

        If the smallest time unit in your format is minutes or seconds,
        then the time is rounded to the nearest minute or second.
        Otherwise the value is truncated rather than rounded.

        """
        offset, uses_ms = _strftime_offset_seconds(format)
        year, month, day, hour, minute, second, jd = self._utc_tuple(offset, 1)
        start_of_year = julian_day(year, 1, 1, self.ts.julian_calendar_cutoff)
        weekday = jd % 7
        yday = jd + 1 - start_of_year
        return _strftime(format, year, month, day, hour, minute, second,
                         weekday, yday, uses_ms)

    def _utc_tuple(self, offset, return_jd=False):
        """Return UTC as (year, month, day, hour, minute, second.fraction).

        The `offset` in seconds is added to the UTC time before it is
        split into its components.  This is useful if the user is going
        to round the result before displaying it.  If the result is
        going to be displayed as seconds, for example, set `offset` to
        0.5 and then throw away the fraction; if the result is going to
        be displayed as minutes, set `offset` to 30.0 and then throw
        away the seconds; and so forth.

        """
        second, sfr, is_leap_second = self._utc_seconds(offset)
        second = second.astype(int64)
        second -= is_leap_second
        jd, second = divmod(second + 43200, 86400)
        cutoff = self.ts.julian_calendar_cutoff
        year, month, day = compute_calendar_date(jd, cutoff)
        minute, second = divmod(second, 60)
        hour, minute = divmod(minute, 60)
        second += is_leap_second
        if not return_jd:
            return year, month, day, hour, minute, second + sfr
        return year, month, day, hour, minute, second + sfr, jd

    def _utc_seconds(self, offset):
        """Return integer seconds since JD 0.0, plus a 0 ≤ fraction < 1."""
        seconds, fr = self._tai_seconds
        seconds2, fr = divmod(fr + offset, 1.0)
        seconds = seconds + seconds2  # not +=, which would modify cached array
        ts = self.ts
        tai_minus_utc = interp(seconds, ts._leap_tai, ts._leap_offsets)
        tai_minus_utc, is_leap_second = divmod(tai_minus_utc, 1.0)
        is_leap_second = is_leap_second > 0.0
        return seconds - tai_minus_utc, fr, is_leap_second

    @reify
    def _tai_seconds(self):
        # If UTC was supplied, this will already hold an exact value; otherwise:
        seconds, fr = divmod(self.whole * DAY_S, 1.0)
        seconds2, fr = divmod(fr + self.tai_fraction * DAY_S, 1.0)
        seconds += seconds2
        return seconds, fr

    def _leap_seconds(self):
        # TODO: should this be reified?
        ts = self.ts
        seconds, fr = self._tai_seconds
        return interp(seconds, ts._leap_tai, ts._leap_offsets)

    # Calendar tuples.

    def tai_calendar(self):
        """TAI as a (year, month, day, hour, minute, second) `calendar date`."""
        return self.ts._cal(self.whole, self.tai_fraction)

    def tt_calendar(self):
        """TT as a (year, month, day, hour, minute, second) `calendar date`."""
        return self.ts._cal(self.whole, self.tt_fraction)

    def tdb_calendar(self):
        """TDB as a (year, month, day, hour, minute, second) `calendar date`."""
        return self.ts._cal(self.whole, self.tdb_fraction)

    def ut1_calendar(self):
        """UT1 as a (year, month, day, hour, minute, second) `calendar date`."""
        return self.ts._cal(self.whole, self.ut1_fraction)

    # Date formatting.

    def tai_strftime(self, format='%Y-%m-%d %H:%M:%S TAI'):
        """Format TAI with a datetime strftime() format string."""
        return self.ts._strftime(format, self.whole, self.tai_fraction)

    def tt_strftime(self, format='%Y-%m-%d %H:%M:%S TT'):
        """Format TT with a datetime strftime() format string."""
        return self.ts._strftime(format, self.whole, self.tt_fraction)

    def tdb_strftime(self, format='%Y-%m-%d %H:%M:%S TDB'):
        """Format TDB with a datetime strftime() format string."""
        return self.ts._strftime(format, self.whole, self.tdb_fraction)

    def ut1_strftime(self, format='%Y-%m-%d %H:%M:%S UT1'):
        """Format UT1 with a datetime strftime() format string."""
        return self.ts._strftime(format, self.whole, self.ut1_fraction)

    # Convenient caching of several expensive functions of time.

    @reify
    def M(self):
        """3×3 rotation matrix: ICRS → equinox of this date."""

        # Compute N and P instead of asking for self.N and self.P to
        # avoid keeping copies of them, since Skyfield itself never uses
        # them again once M has been computed.  But we do check in case
        # a user has forced them to be built already.

        d = self.__dict__

        P = d.get('P')
        if P is None:
            P = self.precession_matrix()

        N = d.get('N')
        if N is None:
            N = self.nutation_matrix()

        return mxmxm(N, P, B)

    @reify
    def MT(self):
        """3×3 rotation matrix: equinox of this date → ICRS."""
        return rollaxis(self.M, 1)

    @reify
    def C(self):
        # Calculate the Equation of Origins in cycles
        eq_origins = (earth_rotation_angle(self.ut1) - self.gast / 24.0)
        R = rot_z(2 * pi * eq_origins)
        return mxm(R, self.M)

    @reify
    def CT(self):
        return rollaxis(self.C, 1)

    @reify
    def _nutation_angles_radians(self):
        # TODO: add psi and eps corrections support back in here, rather
        # than at points of use.
        return iau2000a_radians(self)

    def _nutation_angles(self, angles):
        # Sample code shared with early adopters suggested that setting
        # this attribute manually could avoid the expense of IAU 2000A,
        # so this setter continues to support the pattern.

        d_psi, d_eps = angles
        self._nutation_angles_radians = (
            d_psi / 1e7 * ASEC2RAD,
            d_eps / 1e7 * ASEC2RAD,
        )

    _nutation_angles = property(None, _nutation_angles)

    @reify
    def _mean_obliquity_radians(self):
        # Cached because it is used to compute both gast and N.
        return mean_obliquity(self.tdb) * ASEC2RAD

    # Conversion between timescales.

    @reify
    def J(self):
        """Return a floating point Julian year or array of years for this date.

        Julian years are convenient uniform periods of exactly 365.25
        days of Terrestrial Time, centered on 2000 January 1 12h TT =
        Julian year 2000.0.

        """
        return (self.whole - 1721045.0 + self.tt_fraction) / 365.25

    @reify
    def utc(self):
        """A tuple ``(year, month, day, hour, minute, second)`` in UTC."""
        utc = self._utc_tuple(0.0)
        return (array(utc).view(CalendarArray) if self.shape
                else CalendarTuple(*utc))

    @reify
    def tai_fraction(self):
        return self.tt_fraction - tt_minus_tai

    @reify
    def tdb_fraction(self):
        fr = self.tt_fraction
        return fr + tdb_minus_tt(self.whole, fr) / DAY_S

    @reify
    def ut1_fraction(self):
        return self.tt_fraction - self.delta_t / DAY_S

    @reify
    def delta_t(self):
        return self.ts.delta_t_function(self.tt)

    @reify
    def dut1(self):
        return 32.184 + self._leap_seconds() - self.delta_t

    @reify
    def gmst(self):
        """Greenwich Mean Sidereal Time (GMST) in hours."""
        return sidereal_time(self)

    @reify
    def gast(self):
        """Greenwich Apparent Sidereal Time (GAST) in hours."""
        d_psi, _ = self._nutation_angles_radians
        tt = self.tt
        # TODO: move this into an eqeq function?
        c_terms = equation_of_the_equinoxes_complimentary_terms(tt)
        eq_eq = d_psi * cos(self._mean_obliquity_radians) + c_terms
        return (self.gmst + eq_eq / tau * 24.0) % 24.0

    # Low-precision floats generated from internal float pairs.

    @property
    def tai(self):
        return self.whole + self.tai_fraction

    @property
    def tt(self):
        return self.whole + self.tt_fraction

    @property
    def tdb(self):
        return self.whole + self.tdb_fraction

    @property
    def ut1(self):
        return self.whole + self.ut1_fraction

    # Earth position as a function of time.

    def polar_motion_angles(self):
        table = self.ts.polar_motion_table
        if table is None:
            return 0.0, 0.0, 0.0
        sprime = -47.0e-6 * (self.tdb - T0) / 36525.0
        tt, x, y = table
        return sprime, interp(self.tt, tt, x), interp(self.tt, tt, y)

    def polar_motion_matrix(self):
        sprime, x, y = self.polar_motion_angles()
        return mxmxm(
            rot_x(y * ASEC2RAD),
            rot_y(x * ASEC2RAD),
            rot_z(-sprime * ASEC2RAD),
        )

    def nutation_matrix(self):
        """Compute the 3×3 nutation matrix N for this date."""
        d_psi, d_eps = self._nutation_angles_radians
        mean_obliquity = self._mean_obliquity_radians
        true_obliquity = mean_obliquity + d_eps
        return build_nutation_matrix(mean_obliquity, true_obliquity, d_psi)

    def precession_matrix(self):
        """Compute the 3×3 precession matrix P for this date."""
        return compute_precession(self.tdb)

    # Various dunders.

    def __eq__(self, other_time):
        if isinstance(other_time, Time):
            return self.__sub__(other_time) == 0.0
        return False

    def __lt__(self, other):
        return ((self.tt_fraction - other.tt_fraction)
                + (self.whole - other.whole)) < 0

    def __add__(self, other_time):
        if isinstance(other_time, timedelta):
            w = other_time.days
            f = other_time.seconds / DAY_S + other_time.microseconds / DAY_US
        elif isinstance(other_time, (int, float)):
            w, f = divmod(other_time, 1.0)
        else:
            return NotImplemented

        return self.ts.tt_jd(self.whole + w, self.tt_fraction + f)

    def __sub__(self, other_time):
        if isinstance(other_time, Time):
            return self.whole - other_time.whole + (
                self.tt_fraction - other_time.tt_fraction
            )
        elif isinstance(other_time, timedelta):
            w = other_time.days
            f = other_time.seconds / DAY_S + other_time.microseconds / DAY_US
        elif isinstance(other_time, (int, float)):
            w, f = divmod(other_time, 1.0)
        else:
            return NotImplemented

        return self.ts.tt_jd(self.whole - w, self.tt_fraction - f)

    def __hash__(self):
        # Someone wanted to use Time objects with functools.lru_cache so
        # we make this attempt to support hashability; beware that it
        # will return the same hash for very closely spaced times that
        # all round to the same floating point TT.
        return hash(self.tt)

    # Interoperability.

    def to_astropy(self):
        """Return an AstroPy object representing this time."""
        from astropy.time import Time
        return Time(self.whole, self.tt_fraction, format='jd', scale='tt')

    # Deprecated attributes that were once used internally, consuming
    # memory with matrices that are never used again by Skyfield once
    # t.M has been computed.

    P = reify(precession_matrix)
    N = reify(nutation_matrix)
    P.__doc__ = N.__doc__ = None  # omit from Sphinx documentation

    @reify
    def PT(self): return rollaxis(self.P, 1)
    @reify
    def NT(self): return rollaxis(self.N, 1)

def julian_day(year, month=1, day=1, julian_before=None):
    """Given a calendar date, return a Julian day integer.

    Uses the proleptic Gregorian calendar unless ``julian_before`` is
    set to a specific Julian day, in which case the Julian calendar is
    used for dates older than that.

    """
    # Support months <1 and >12 by overflowing cleanly into adjacent years.
    y, month = divmod(month - 1, 12)
    year = year + y
    month += 1

    # See the Explanatory Supplement to the Astronomical Almanac 15.11.
    janfeb = month <= 2
    g = year + 4716 - janfeb
    f = (month + 9) % 12
    e = 1461 * g // 4 + day - 1402
    J = e + (153 * f + 2) // 5

    mask = 1 if (julian_before is None) else (J >= julian_before)
    J += (38 - (g + 184) // 100 * 3 // 4) * mask
    return J

def julian_date(year, month=1, day=1, hour=0, minute=0, second=0.0):
    """Given a proleptic Gregorian calendar date and time, build a Julian date.

    The difference between a “Julian day” and a “Julian date” is that
    the “day” is the integer part, while the “date” includes a fraction
    indicating the time.

    """
    return julian_day(year, month, day) - 0.5 + (
        second + minute * 60.0 + hour * 3600.0) / DAY_S

def julian_date_of_besselian_epoch(b):
    return 2415020.31352 + (b - 1900.0) * 365.242198781

def compute_calendar_date(jd_integer, julian_before=None):
    """Convert Julian day ``jd_integer`` into a calendar (year, month, day).

    Uses the proleptic Gregorian calendar unless ``julian_before`` is
    set to a specific Julian day, in which case the Julian calendar is
    used for dates older than that.

    """
    use_gregorian = (julian_before is None) or (jd_integer >= julian_before)

    # See the Explanatory Supplement to the Astronomical Almanac 15.11.
    f = jd_integer + 1401
    f += use_gregorian * ((4 * jd_integer + 274277) // 146097 * 3 // 4 - 38)
    e = 4 * f + 3
    g = e % 1461 // 4
    h = 5 * g + 2
    day = h % 153 // 5 + 1
    month = (h // 153 + 2) % 12 + 1
    year = e // 1461 - 4716 + (12 + 2 - month) // 12
    return year, month, day

calendar_date = compute_calendar_date  # old name, in case anyone used it

def calendar_tuple(jd_float, fraction=0.0, julian_before=None):
    """Return a (year, month, day, hour, minute, second.fraction) tuple."""
    jd_float = _to_array(jd_float)
    whole1, fraction1 = divmod(jd_float, 1.0)
    whole2, fraction = divmod(fraction1 + fraction + 0.5, 1.0)
    whole = (whole1 + whole2).astype(int)
    year, month, day = compute_calendar_date(whole, julian_before)
    second = fraction * 86400.0
    minute, second = divmod(second, 60.0)
    minute = minute.astype(int)
    hour, minute = divmod(minute, 60)
    return year, month, day, hour, minute, second

def tdb_minus_tt(jd_tdb, fraction_tdb=0.0):
    """Computes how far TDB is in advance of TT, given TDB.

    Given that the two time scales never diverge by more than 2ms, TT
    can also be given as the argument to perform the conversion in the
    other direction.

    """
    t = (jd_tdb - T0 + fraction_tdb) / 36525.0

    # USNO Circular 179, eq. 2.6.
    return (0.001657 * sin ( 628.3076 * t + 6.2401)
          + 0.000022 * sin ( 575.3385 * t + 4.2970)
          + 0.000014 * sin (1256.6152 * t + 6.1969)
          + 0.000005 * sin ( 606.9777 * t + 4.0212)
          + 0.000005 * sin (  52.9691 * t + 0.4444)
          + 0.000002 * sin (  21.3299 * t + 5.5431)
          + 0.000010 * t * sin ( 628.3076 * t + 4.2490))

class DeltaT(object):
    def __init__(self, table_tt, table_delta_t, long_term_function):
        self.table_tt = table_tt
        self.table_delta_t = table_delta_t
        self.long_term_function = long_term_function

    def __call__(self, tt):
        delta_t = interp(tt, self.table_tt, self.table_delta_t, nan, nan)
        if getattr(delta_t, 'shape', None):
            nan_indexes = nonzero(isnan(delta_t))
            if nan_indexes:
                J = (tt[nan_indexes] - 1721045.0) / 365.25
                delta_t[nan_indexes] = self.long_term_function(J)
        else:
            if isnan(delta_t):
                J = (tt - 1721045.0) / 365.25
                delta_t = self.long_term_function(J)
        return delta_t

delta_t_parabola_stephenson_morrison_hohenkerk_2016 = Splines(
    [1825.0, 1925.0, 0.0, 32.5, 0.0, -320.0])

delta_t_parabola_morrison_stephenson_2004 = Splines(
    [1820.0, 1920.0, 0.0, 32.0, 0.0, -20.0])

def build_delta_t(delta_t_recent):
    """Return a function t→∆T, given recent real-world observations of ∆T.

    When asked for ∆T outside the range of the `delta_t_recent` table,
    Skyfield uses the splines for 720 BC – AD 2015 computed by Morrison,
    Stephenson, Hohenkerk, and Zawilski.  For dates that fall outside of
    the spines, we use the long-term parabola of Stephenson, Morrison,
    and Hohenkerk.

    """
    parabola = delta_t_parabola_stephenson_morrison_hohenkerk_2016
    s15_table = load_bundled_npy('delta_t.npz')['Table-S15.2020.txt']
    table_tt, table_delta_t = delta_t_recent

    p = parabola
    pd = parabola.derivative
    s = Splines(s15_table)
    sd = s.derivative

    long_term_parabola_width = p.upper[0] - p.lower[0]

    # How many years wide we make the splines that connect the tables to
    # the long-term parabola; tuned by hand until the derivative graphed
    # by `work_on_delta_t_discontinuities()` in `design/delta_t.py`
    # doesn't look too terrible.
    patch_width = 800.0

    # To the left of the Table-S15 splines, design a spline connecting
    # them to the long-term parabola.

    x1 = s.lower[0]  # For the current table, this = -720.0.
    x0 = x1 - patch_width
    left = build_spline_given_ends(x0, p(x0), pd(x0), x1, s(x1), sd(x1))

    # And to the left of that, put the pure long-term parabola.

    x1 = x0
    x0 = x1 - long_term_parabola_width
    far_left = build_spline_given_ends(x0, p(x0), pd(x0), x1, p(x1), pd(x1))

    # Truncate the splines table where the daily table starts, and
    # adjust the final spline to remove any discontinuity.

    x = (table_tt[0] - 1721045.0) / 365.25  # TT to J centuries

    i = searchsorted(s15_table[0], x)
    s15_table = s15_table[:,:i]

    desired_y = table_delta_t[0]
    current_y = s(x)
    x0, x1, a3, a2, a1, a0 = s15_table[:,-1]
    t = (x - x0) / (x1 - x0)
    a1 = a1 + (desired_y - current_y) / t  # adjust linear term
    s15_table[:,-1] = x0, x1, a3, a2, a1, a0

    # To the right of the recent ∆T table, design a spline connecting
    # smoothly to the long-term parabola.

    x0 = (table_tt[-1] - 1721045.0) / 365.25  # TT to J centuries
    x1 = (x0 + patch_width) // 100.0 * 100.0  # Choose multiple of 100 years
    y0 = table_delta_t[-1]

    lookback = min(366, len(table_delta_t))       # Slope of last year of ∆T.
    slope = (table_delta_t[-1] - table_delta_t[-lookback]) * lookback / 365.0
    right = build_spline_given_ends(x0, y0, slope, x1, p(x1), pd(x1))

    # At the far right, finish with the pure long-term parabola.

    x0 = x1
    x1 = x0 + long_term_parabola_width
    far_right = build_spline_given_ends(x0, p(x0), pd(x0), x1, p(x1), pd(x1))

    curve = Splines(_cat(
        array([far_left]).T,
        array([left]).T,
        s15_table,
        array([right]).T,
        array([far_right]).T,
    ))

    return DeltaT(table_tt, table_delta_t, curve)

def build_delta_t_table(delta_t_recent):
    """Deprecated: the Delta T interpolation table used in Skyfield <= 1.37."""
    bundled = _cat(
        load_bundled_npy('morrison_stephenson_deltat.npy')[:,:22],
        load_bundled_npy('historic_deltat.npy'),
    )
    recent_start_time = delta_t_recent[0,0]
    i = searchsorted(bundled[0], recent_start_time)
    century = 36524.0
    start_tt = bundled[0,0] - century
    start_J = Time(None, start_tt).J
    end_tt = delta_t_recent[0,-1] + century
    end_J = Time(None, end_tt).J
    return _cat(
        [[start_tt], [delta_t_parabola_morrison_stephenson_2004(start_J)]],
        bundled[:,:i],
        delta_t_recent,
        [[end_tt], [delta_t_parabola_morrison_stephenson_2004(end_J)]],
    )

_format_uses_milliseconds = re.compile(r'%[-_0^#EO]*f').search
_format_uses_seconds = re.compile(r'%[-_0^#EO]*[STXc]').search
_format_uses_minutes = re.compile(r'%[-_0^#EO]*[MR]').search
_format_uses_day_of_year = re.compile(r'%[-_0^#EO]*j').search

def _datetime_to_utc_tuple(dt):
    z = dt.tzinfo
    if z is None:
        raise ValueError(_naive_complaint)
    if z is not utc:
        dt = dt.astimezone(utc)
    return (dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

def _normalize_jd_and_fraction(jd, fraction):
    jd = _to_array(jd)
    if fraction is None:
        jd, fraction = divmod(jd, 1.0)
    else:
        jd, fraction = _reconcile(jd, _to_array(fraction))
    return jd, fraction

def _strftime_offset_seconds(format):
    uses_ms = _format_uses_milliseconds(format)
    if uses_ms:
        if _OLD_PYTHON:
            raise ValueError('strftime() "%f" not supported under Python 2')
        offset = 1e-16  # encourage .0 to not turn into .999999
    elif _format_uses_seconds(format):
        offset = 0.5
    elif _format_uses_minutes(format):
        offset = 30.0
    else:
        offset = 0.0
    return offset, uses_ms

def _strftime(format, year, month, day, hour, minute, second,
              weekday, yday, uses_ms):
    zero = year * 0

    # TODO: avoid computing yday if this is false:
    #_format_uses_day_of_year(format)

    if uses_ms:
        format = format[:uses_ms.start()] + '%Z' + format[uses_ms.end():]
        second = (second * 1e6).astype(int)
        second, usec = divmod(second, 1000000)
        if getattr(year, 'ndim', 0):
            u = ['%06d' % u for u in usec]
            tup = (year, month, day, hour, minute, second,
                   weekday, yday, zero, u)
            return [strftime(format, struct_time(t)) for t in zip(*tup)]
        u = '%06d' % usec
        tup = year, month, day, hour, minute, second, weekday, yday, zero, u
        return strftime(format, struct_time(tup))
    else:
        second = second.astype(int)
        tup = year, month, day, hour, minute, second, weekday, yday, zero
        if getattr(year, 'ndim', 0):
            return [strftime(format, item) for item in zip(*tup)]
        return strftime(format, tup)

def _upgrade_datetime_exception(args, e):
    year, month, day, hour, minute, second, micro, zone = args
    if year < 0:
        e.args = ("Python's datetime does not support negative"
                  ' years like the year {}'.format(year),)
    elif month == 2 and day == 29:
        e.args = ("Python's datetime does not support Julian leap"
                  ' days like {} February 29 that are missing from'
                  ' the Gregorian calendar'.format(year),)

_naive_complaint = """cannot interpret a datetime that lacks a timezone

You must either specify that your datetime is in UTC:

    from skyfield.api import utc
    d = datetime(..., tzinfo=utc)  # to build a new datetime
    d = d.replace(tzinfo=utc)      # to fix an existing datetime

Or use a timezone object like those provided by the third-party `pytz` library:

    from pytz import timezone
    eastern = timezone('US/Eastern')
    d = eastern.localize(datetime(2014, 1, 16, 1, 32, 9))"""
