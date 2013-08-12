"""Compare the output of Skyfield with the same routines from NOVAS."""

from itertools import product
from numpy import array, einsum
from unittest import TestCase

from skyfield import (coordinates, earthlib, framelib, nutationlib,
                      planets, precessionlib, starlib, timescales)

from ..constants import T0, DEG2RAD, AU_KM, TAU
from ..timescales import JulianDate

# Since some users might run these tests without having installed our
# test dependencies, we detect import errors and skip these tests if the
# resources they need are not available.

try:
    import de405
except ImportError:
    de405 = None

try:
    import novas
    import novas_de405
except ImportError:
    novas = None
else:
    import novas.compat as c
    import novas.compat.eph_manager

    jd_start, jd_end, number = c.eph_manager.ephem_open()  # needs novas_de405

    c_nutation = c.nutation
    import novas.compat.nutation  # overwrites nutation() function with module!

    TA = c.julian_date(1969, 7, 20, 20. + 18. / 60.)
    TB = c.julian_date(2012, 12, 21)

    D0 = 63.8285
    DA = 39.707
    DB = 66.8779

    P0 = (T0, D0)  # "pair 0"
    PA = (TA, DA)
    PB = (TB, DB)

arcminute = DEG2RAD / 60.0
arcsecond = arcminute / 60.0
meter = 1.0 / AU_KM

planet_codes = {
    'mercury': 1,
    'venus': 2,
    'mars': 4,
    'jupiter': 5,
    'saturn': 6,
    'uranus': 7,
    'neptune': 8,
    'pluto': 9,
    'sun': 10,
    'moon': 11,
    }

planets_to_test = planet_codes.keys()

# Defining fixtures
import pytest

@pytest.fixture(params=[P0, PA, PB])
def timepairs(request):
    return request.param

@pytest.fixture(params=planet_codes.items())
def planets_list(request):
    return request.param

emp = planets.Ephemeris(de405)

def eq(first, second, epsilon=None):
    """Test whether two floats are within `epsilon` of one another."""
    if hasattr(first, 'shape') or hasattr(second, 'shape'):
        failed = abs(first - second).max() > epsilon
    else:
        failed = abs(first - second) > epsilon
    if failed:
        appendix = ('\nbecause the difference is\n%r\ntimes too big'
                    % (abs(first - second) / epsilon)) if epsilon else ''
        raise AssertionError(
            '%r\ndoes not equal\n%r\nwithin the error bound\n%r%s'
            % (first, second, epsilon, appendix))


def test_new_star_deflected_by_jupiter(timepairs):
    """ Tests of generating a stellar position. """
    jd_tt = timepairs[0]
    star = c.make_cat_entry(
        star_name='Star', catalog='cat', star_num=101,
        ra=1.59132070233, dec=8.5958876464,
        pm_ra=0.0, pm_dec=0.0,
        parallax=0.0, rad_vel=0.0,
        )
    ra, dec = c.app_star(jd_tt, star)
        
    earth = emp.earth
    star = starlib.Star(
        ra=1.59132070233, dec=8.5958876464,
        pm_ra=0.0, pm_dec=0.0,
        parallax=0.0, radial_velocity=0.0,
        )
    jd = JulianDate(tt=jd_tt)
    g = star.observe_from(earth(jd)).apparent()
    
    eq(ra * TAU / 24.0, g.ra, 0.001 * arcsecond)
    eq(dec * TAU / 360.0, g.dec, 0.001 * arcsecond)

# Tests of generating a full position or coordinate.

def test_astro_planet(timepairs, planets_list):
    jd_tt = timepairs[0]
    planet_name = planets_list[0]
    planet_code = planets_list[1]
    
    obj = c.make_object(0, planet_code, 'planet', None)
    ra, dec, dis = c.astro_planet(jd_tt, obj)
    
    earth = emp.earth
    planet = getattr(emp, planet_name)
    jd = JulianDate(tt=jd_tt)
    g = planet.observe_from(earth(jd)).astrometric()
    
    eq(ra * TAU / 24.0, g.ra, 0.001 * arcsecond)
    eq(dec * TAU / 360.0, g.dec, 0.001 * arcsecond)
    eq(dis, g.distance, 0.001 * meter)


def test_app_planet(timepairs, planets_list):
    jd_tt = timepairs[0]
    planet_name = planets_list[0]
    planet_code = planets_list[1]
    
    obj = c.make_object(0, planet_code, 'planet', None)
    ra, dec, dis = c.app_planet(jd_tt, obj)

    earth = emp.earth
    planet = getattr(emp, planet_name)
    jd = JulianDate(tt=jd_tt)
    g = planet.observe_from(earth(jd)).apparent()

    eq(ra * TAU / 24.0, g.ra, 0.001 * arcsecond)
    eq(dec * TAU / 360.0, g.dec, 0.001 * arcsecond)
    eq(dis, g.distance, 0.001 * meter)

def test_topo_planet(timepairs, planets_list):
    position = c.make_on_surface(45.0, -75.0, 0.0, 10.0, 1010.0)
    ggr = coordinates.Topos('75 W', '45 N', 0.0,
                            temperature=10.0, pressure=1010.0)
    ggr.earth = emp.earth
    ggr.ephemeris = emp

    jd_tt = timepairs[0]
    delta_t = timepairs[1]
    planet_name = planets_list[0]
    planet_code = planets_list[1]

    obj = c.make_object(0, planet_code, 'planet', None)
    ra, dec, dis = c.topo_planet(jd_tt, delta_t, obj, position)

    planet = getattr(emp, planet_name)
    jd = JulianDate(tt=jd_tt, delta_t=delta_t)
    g = ggr(jd).observe(planet).apparent()

    eq(ra * TAU / 24.0, g.ra, 0.001 * arcsecond)
    eq(dec * TAU / 360.0, g.dec, 0.001 * arcsecond)
    eq(dis, g.distance, 0.001 * meter)


def test_new_horizontal(timepairs, planets_list):
    """ Tests of generating a full position in horizontal coordinates. Uses
        fixtures to iterate through date pairs and planets to generate
        individual tests.
    """
    jd_tt = timepairs[0]
    delta_t = timepairs[1]
    planet_name = planets_list[0]
    planet_code = planets_list[1]
    position = c.make_on_surface(45.0, -75.0, 0.0, 10.0, 1010.0)
    ggr = coordinates.Topos('75 W', '45 N', 0.0,
                            temperature=10.0, pressure=1010.0)
    ggr.earth = emp.earth
    ggr.ephemeris = emp
    xp = yp = 0.0

    # replaces the for loop
    obj = c.make_object(0, planet_code, 'planet', None)
    ra, dec, dis = c.topo_planet(jd_tt, delta_t, obj, position)
    jd_ut1 = jd_tt - delta_t / 86400.0
    (zd, az), (ra, dec) = c.equ2hor(
        jd_ut1, delta_t, xp, yp, position, ra, dec, ref_option=0)
    planet = getattr(emp, planet_name)
    jd = JulianDate(tt=jd_tt, delta_t=delta_t)
    h = ggr(jd).observe(planet).apparent().horizontal()
    
    eq(zd * TAU / 360.0, h.zd, 0.001 * arcsecond)
    eq(az * TAU / 360.0, h.az, 0.001 * arcsecond)
    eq(0.25 * TAU - zd * TAU / 360.0, h.alt, 0.001 * arcsecond)
    eq(dis, h.distance, 0.001 * meter)

# Tests for Basic Functions

def test_cal_date():
    for jd in 0.0, 2414988.5, 2415020.31352, 2442249.5, 2456335.2428472:
        assert c.cal_date(jd) == timescales.cal_date(jd)

def test_earth_rotation_angle():
    epsilon = 1e-12

    a0 = c.era(T0)
    aA = c.era(TA)
    aB = c.era(TB)

    t = array([T0, TA, TB])
    v = earthlib.earth_rotation_angle(t)
    eq(v, [a0, aA, aB], epsilon)

def test_earth_tilt():
    epsilon = 1e-9

    vars0 = c.e_tilt(T0)
    vars1 = c.e_tilt(TA)
    vars2 = c.e_tilt(TB)

    jd = JulianDate(tdb=[T0, TA, TB])
    v = nutationlib.earth_tilt(jd)
    for i in range(len(v)):
        eq(v[i], [vars0[i], vars1[i], vars2[i]], epsilon)

def test_equation_of_the_equinoxes_complimentary_terms():
    epsilon = 1e-23

    e0 = c.ee_ct(T0, 0.0, 0)
    eA = c.ee_ct(TA, 0.0, 0)
    eB = c.ee_ct(TB, 0.0, 0)

    t = array([T0, TA, TB])
    v = nutationlib.equation_of_the_equinoxes_complimentary_terms(t)
    eq(v, [e0, eA, eB], epsilon)

def test_frame_tie():
    epsilon = 1e-15
    v = array([1, 2, 3])

    eq(c.frame_tie(v, 0), v.dot(framelib.ICRS_to_J2000), epsilon)
    eq(c.frame_tie(v, -1), v.dot(framelib.J2000_to_ICRS), epsilon)

def test_fundamental_arguments():
    epsilon = 1e-12

    args0 = c.fund_args(jcentury(T0))
    argsA = c.fund_args(jcentury(TA))
    argsB = c.fund_args(jcentury(TB))

    t = array([T0, TA, TB])
    v = nutationlib.fundamental_arguments(jcentury(t))
    eq(v.T, [args0, argsA, argsB], epsilon)

def test_geocentric_position_and_velocity():
    epsilon = 1e-13

    delta_t = 0.0
    observer = c.make_observer_on_surface(45.0, -75.0, 0.0, 10.0, 1010.0)

    pos0, vel0 = c.geo_posvel(T0, delta_t, observer)
    posA, velA = c.geo_posvel(TA, delta_t, observer)

    topos = coordinates.Topos('75 W', '45 N', elevation=0.0,
                              temperature=10.0, pressure=1010.0)

    jd = JulianDate(tt=[T0, TA], delta_t=delta_t)
    posv, velv = earthlib.geocentric_position_and_velocity(topos, jd)
    eq(posv.T, [pos0, posA], epsilon)
    eq(velv.T, [vel0, velA], epsilon)

def test_iau2000a():
    epsilon = 1e-18

    psi0, eps0 = c.nutation.iau2000a(T0, 0.0)
    psiA, epsA = c.nutation.iau2000a(TA, 0.0)
    psiB, epsB = c.nutation.iau2000a(TB, 0.0)

    t = array([T0, TA, TB])
    psi, eps = nutationlib.iau2000a(t)
    eq(psi, [psi0, psiA, psiB], epsilon)
    eq(eps, [eps0, epsA, epsB], epsilon)

def test_julian_date():
    epsilon = 0.0
    for args in (
          (-4712, 1, 1, 0.0),
          (-4712, 3, 1, 0.0),
          (-4712, 12, 31, 0.5),
          (-241, 3, 25, 19.0),
          (530, 9, 27, 23.5),
          (1976, 3, 7, 12.5),
          (2000, 1, 1, 0.0),
          ):
        eq(c.julian_date(*args), timescales.julian_date(*args), epsilon)

def test_mean_obliq():
    epsilon = 0

    m0 = c.mean_obliq(T0)
    mA = c.mean_obliq(TA)
    mB = c.mean_obliq(TB)

    t = array([T0, TA, TB])
    v = nutationlib.mean_obliquity(t)
    eq(v, [m0, mA, mB], epsilon)

def test_nutation():
    epsilon = 1e-15
    v = array([1, 2, 3])

    v0 = c_nutation(T0, v, direction=0)
    va = c_nutation(TA, v, direction=0)
    vb = c_nutation(TB, v, direction=0)

    jd = JulianDate(tt=[T0, TA, TB])
    v = einsum('i,ijk->jk', v, nutationlib.compute_nutation(jd))

    eq(v0, v[:,0], epsilon)
    eq(va, v[:,1], epsilon)
    eq(vb, v[:,2], epsilon)

def test_precession():
    epsilon = 1e-15
    v = array([1, 2, 3])

    va = c.precession(T0, v, TA)
    vb = c.precession(T0, v, TB)

    ab = array([TA, TB])
    vab = einsum('i,ijk->jk', v, precessionlib.compute_precession(ab))

    eq(va, vab[:,0], epsilon)
    eq(vb, vab[:,1], epsilon)

def test_sidereal_time_with_zero_delta_t():
    epsilon = 1e-13

    delta_t = 0.0

    st0 = c.sidereal_time(T0, 0.0, delta_t, False, True)
    stA = c.sidereal_time(TA, 0.0, delta_t, False, True)
    stB = c.sidereal_time(TB, 0.0, delta_t, False, True)

    jd = JulianDate(ut1=[T0, TA, TB], delta_t=delta_t)
    v = earthlib.sidereal_time(jd)
    eq(v, [st0, stA, stB], epsilon)

def test_sidereal_time_with_nonzero_delta_t():
    epsilon = 1e-13

    st0 = c.sidereal_time(T0, 0.0, D0, False, True)
    stA = c.sidereal_time(TA, 0.0, DA, False, True)
    stB = c.sidereal_time(TB, 0.0, DB, False, True)

    jd = JulianDate(ut1=[T0, TA, TB], delta_t=[D0, DA, DB])
    v = earthlib.sidereal_time(jd)
    eq(v, [st0, stA, stB], epsilon)

def test_starvectors():
    epsilon = 1e-10

    p, v = c.starvectors(c.make_cat_entry(
            'POLARIS', 'HIP', 0, 2.530301028, 89.264109444,
            44.22, -11.75, 7.56, -17.4))

    star = starlib.Star(2.530301028, 89.264109444,
                        44.22, -11.75, 7.56, -17.4)

    eq(p, star._position.reshape(3), epsilon)
    eq(v, star._velocity.reshape(3), epsilon)

def test_terra():
    epsilon = 1e-18

    observer = c.make_on_surface(45.0, -75.0, 0.0, 10.0, 1010.0)

    class Topos(object):
        latitude = 45.0 * DEG2RAD
        longitude = -75.0 * DEG2RAD
        elevation = 0.0
    topos = Topos()

    pos0, vel0 = array(c.terra(observer, 11.0))
    pos1, vel1 = array(c.terra(observer, 23.9))

    posn, veln = earthlib.terra(topos, array([11.0, 23.9]))

    eq(pos0, posn[:,0], epsilon)
    eq(pos1, posn[:,1], epsilon)
    eq(vel0, veln[:,0], epsilon)
    eq(vel1, veln[:,1], epsilon)

def test_tdb2tt():
    epsilon = 1e-16

    tt0 = c.tdb2tt(T0)[1]
    ttA = c.tdb2tt(TA)[1]
    ttB = c.tdb2tt(TB)[1]

    t = array([T0, TA, TB])
    v = timescales.tdb_minus_tt(t)
    eq(v, [tt0, ttA, ttB], epsilon)

def jcentury(t):
    return (t - T0) / 36525.0


