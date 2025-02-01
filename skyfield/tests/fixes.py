"""Helpers for making Skyfield tests stable."""

import datetime as dt
import sys

from skyfield import earthlib
import skyfield.api
import skyfield.timelib

IS_32_BIT = (sys.maxsize == 0x7fffffff)
_real_datetime_class = dt.datetime

class datetime(dt.datetime):
    _fake_now = None
    @staticmethod
    def now():
        return datetime._fake_now

def setup(utc=(2015, 10, 11, 10)):
    # Importing Pandas raises a TypeError once our fake datetime is
    # installed.  So instead of waiting, let's import it now.
    __import__('pandas')

    fake_now = dt.datetime(*utc)
    skyfield.api.load = skyfield.iokit.Loader('.', verbose=False)
    datetime._fake_now = fake_now
    dt.datetime = datetime

def teardown():
    skyfield.api.load = skyfield.iokit.Loader('.')
    dt.datetime = _real_datetime_class

class low_precision_ERA(object):
    """Compute the Earth rotation angle with only a single float for UT1.

    Skyfield now uses two floats ``t.whole`` and ``t.ut1_fraction`` to
    represent the UT1 Julian date, supplying an additional 16 digits of
    precision.  For the Earth rotation angle, which moves very quickly
    per unit time compared to most other astronomical quantities, this
    knocks Skyfield out of agreement with other libraries like NOVAS and
    SOFA that round UT1 to a single float.  Various tests use this
    context manager to make Skyfield match the lower-precision output.

    """
    def __enter__(self):
        self.saved = earthlib.earth_rotation_angle
        earthlib.earth_rotation_angle = self.era
    def __exit__(self, *args):
        earthlib.earth_rotation_angle = self.saved
    def era(self, whole, fraction):
        rounded_single_float = whole + fraction
        return self.saved(rounded_single_float)
