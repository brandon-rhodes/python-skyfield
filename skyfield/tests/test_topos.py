from assay import assert_raises
from collections import namedtuple
from numpy import abs, arange, sqrt

from skyfield import constants
from skyfield.api import Distance, load, wgs84, wms
from skyfield.functions import length_of
from skyfield.positionlib import Apparent, Barycentric, ICRF
from skyfield.toposlib import ITRSPosition, iers2010

angle = (-15, 15, 35, 45)

def ts():
    yield load.timescale()

def t():
    ts = load.timescale()
    yield ts.utc(2020, 11, 3, 17, 5)
    yield ts.utc(2020, 11, 3, 17, [5, 5])

def test_latitude_longitude_elevation_str_and_repr():
    w = wgs84.latlon(36.7138, -112.2169, 2400.0)
    assert str(w) == ('WGS84 latitude +36.7138 N'
                      ' longitude -112.2169 E elevation 2400.0 m')
    assert repr(w) == ('<GeographicPosition WGS84 latitude +36.7138 N'
                       ' longitude -112.2169 E elevation 2400.0 m>')

    w = wgs84.latlon([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    assert str(w) == (
        'WGS84 latitude [+1.0000 +2.0000] N'
        ' longitude [3.0000 4.0000] E'
        ' elevation [5.0 6.0] m'
    )
    assert repr(w) == '<GeographicPosition {0}>'.format(w)

    w = wgs84.latlon(arange(6.0), arange(10.0, 16.0), arange(20.0, 26.0))
    assert str(w) == (
        'WGS84 latitude [+0.0000 +1.0000 ... +4.0000 +5.0000] N'
        ' longitude [10.0000 11.0000 ... 14.0000 15.0000] E'
        ' elevation [20.0 21.0 ... 24.0 25.0] m'
    )
    assert repr(w) == '<GeographicPosition {0}>'.format(w)

def test_raw_itrs_position():
    d = Distance(au=[1, 2, 3])
    p = ITRSPosition(d)
    ts = load.timescale()
    t = ts.utc(2020, 12, 16, 12, 59)
    p.at(t)

def test_wgs84_velocity_matches_actual_motion():
    # It looks like this is a sweet spot for accuracy: presumably a
    # short enough fraction of a second that the vector does not time to
    # change direction much, but long enough that the direction does not
    # get lost down in the noise.
    factor = 300.0

    ts = load.timescale()
    t = ts.utc(2019, 11, 2, 3, 53, [0, 1.0 / factor])
    jacob = wgs84.latlon(36.7138, -112.2169)
    p = jacob.at(t)
    velocity1 = p.position.km[:,1] - p.position.km[:,0]
    velocity2 = p.velocity.km_per_s[:,0]
    assert length_of(velocity2 - factor * velocity1) < 0.0007

def test_lst():
    ts = load.timescale()
    ts.delta_t_table = [-1e99, 1e99], [69.363285] * 2  # from finals2000A.all
    t = ts.utc(2020, 11, 27, 15, 34)
    top = wgs84.latlon(0.0, 0.0)
    expected = 20.0336663100  # see "authorities/horizons-lst"
    actual = top.lst_hours_at(t)
    difference_mas = (actual - expected) * 3600 * 15 * 1e3
    horizons_ra_offset_mas = 51.25
    difference_mas -= horizons_ra_offset_mas
    assert abs(difference_mas) < 1.0

def test_itrs_xyz_attribute_and_itrf_xyz_method():
    top = wgs84.latlon(45.0, 0.0, elevation_m=constants.AU_M - constants.ERAD)

    x, y, z = top.itrs_xyz.au
    assert abs(x - sqrt(0.5)) < 2e-7
    assert abs(y - 0.0) < 1e-14
    assert abs(z - sqrt(0.5)) < 2e-7

    ts = load.timescale()
    t = ts.utc(2019, 11, 2, 3, 53)
    x, y, z = top.at(t).itrf_xyz().au
    assert abs(x - sqrt(0.5)) < 1e-4
    assert abs(y - 0.0) < 1e-14
    assert abs(z - sqrt(0.5)) < 1e-4

def test_polar_motion_when_computing_topos_position(ts):
    xp_arcseconds = 11.0
    yp_arcseconds = 22.0
    ts.polar_motion_table = [0.0], [xp_arcseconds], [yp_arcseconds]

    top = iers2010.latlon(wms(42, 21, 24.1), wms(-71, 3, 24.8), 43.0)
    t = ts.utc(2005, 11, 12, 22, 2)

    # "expected" comes from:
    # from novas.compat import ter2cel
    # print(ter2cel(t.whole, t.ut1_fraction, t.delta_t, xp_arcseconds,
    #               yp_arcseconds, top.itrs_xyz.km, method=1))

    expected = (3129.530248036487, -3535.1665884086683, 4273.94957733827)
    assert max(abs(top.at(t).position.km - expected)) < 3e-11

def test_polar_motion_when_computing_altaz_coordinates(ts):
    latitude = 37.3414
    longitude = -121.6429
    elevation = 1283.0
    ra_hours = 5.59
    dec_degrees = -5.45

    xp_arcseconds = 11.0
    yp_arcseconds = 22.0
    ts.polar_motion_table = [0.0], [xp_arcseconds], [yp_arcseconds]

    t = ts.utc(2020, 11, 12, 22, 16)
    top = wgs84.latlon(latitude, longitude, elevation)

    pos = Apparent.from_radec(ra_hours, dec_degrees, epoch=t)
    pos.t = t
    pos.center = top

    alt, az, distance = pos.altaz()

    # To generate the test altitude and azimuth below:
    # from novas.compat import equ2hor, make_on_surface
    # location = make_on_surface(latitude, longitude, elevation, 0, 0)
    # (novas_zd, novas_az), (rar, decr) = equ2hor(
    #     t.ut1, t.delta_t, xp_arcseconds, yp_arcseconds, location,
    #     ra_hours, dec_degrees, 0,
    # )
    # novas_alt = 90.0 - novas_zd
    # print(novas_alt, novas_az)

    novas_alt = -58.091983295564205
    novas_az = 1.8872567543791035

    assert abs(alt.degrees - novas_alt) < 1.9e-9
    assert abs(az.degrees - novas_az) < 1.3e-7

def test_subpoint_with_wrong_center(ts, angle):
    t = ts.utc(2020, 12, 31)
    p = Barycentric([0,0,0], t=t)
    with assert_raises(ValueError, 'you can only calculate a geographic'
                       ' position from a position which is geocentric'
                       ' .center=399., but this position has a center of 0'):
        wgs84.subpoint(p)

def test_iers2010_subpoint(ts, angle):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    # An elevation of 0 is more difficult for the routine's accuracy
    # than a very large elevation.
    top = iers2010.latlon(angle, angle, elevation_m=0.0)
    p = top.at(t)
    b = iers2010.subpoint(p)

    error_degrees = abs(b.latitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

    error_degrees = abs(b.longitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

def test_wgs84_subpoint(ts, angle):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    # An elevation of 0 is more difficult for the routine's accuracy
    # than a very large elevation.
    top = wgs84.latlon(angle, angle, elevation_m=0.0)
    p = top.at(t)
    b = wgs84.subpoint(p)

    error_degrees = abs(b.latitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

    error_degrees = abs(b.longitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

def test_wgs84_subpoint_at_pole(ts):
    # The `height` previously suffered from very low precision at the pole.
    t = ts.utc(2023, 4, 7, 12, 44)
    p = wgs84.latlon(90, 0, elevation_m=10.0).at(t)
    micrometer = 1e-6

    h = wgs84.height_of(p)
    assert abs(h.m - 10.0) < micrometer

    g = wgs84.geographic_position_of(p)
    assert abs(g.elevation.m - 10.0) < micrometer

def test_wgs84_round_trip_with_polar_motion(ts, angle):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    ts.polar_motion_table = [0.0], [0.003483], [0.358609]

    top = wgs84.latlon(angle, angle, elevation_m=0.0)
    p = top.at(t)
    b = wgs84.subpoint(p)

    error_degrees = abs(b.latitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

    error_degrees = abs(b.longitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

def test_latlon_and_subpoint_methods(t, angle):
    g = wgs84.latlon(angle, 2 * angle, elevation_m=1234.0)
    pos = g.at(t)
    x = all if t.shape else lambda x: x

    def check_lat(lat): assert x(abs(g.latitude.mas() - lat.mas()) < 0.1)
    def check_lon(lon): assert x(abs(g.longitude.mas() - lon.mas()) < 0.1)
    def check_height(h): assert x(abs(g.elevation.m - h.m) < 1e-7)
    def check_itrs(xyz, expected_distance):
        r1 = g.itrs_xyz.m
        r2 = xyz
        if len(r2.shape) > len(r1.shape):
            r1.shape += (1,)
        actual_distance = length_of(r1 - r2)
        assert x(abs(actual_distance - expected_distance) < 1e-7)

    lat, lon = wgs84.latlon_of(pos)
    check_lat(lat)
    check_lon(lon)

    height = wgs84.height_of(pos)
    check_height(height)

    g2 = wgs84.geographic_position_of(pos)
    check_lat(g2.latitude)
    check_lon(g2.longitude)
    check_height(g2.elevation)
    check_itrs(g2.itrs_xyz.m, 0.0)

    g2 = wgs84.subpoint(pos)  # old deprecated method name
    check_lat(g2.latitude)
    check_lon(g2.longitude)
    check_height(g2.elevation)
    check_itrs(g2.itrs_xyz.m, 0.0)

    g2 = wgs84.subpoint_of(pos)
    check_lat(g2.latitude)
    check_lon(g2.longitude)
    assert g2.elevation.m == 0.0
    check_itrs(g2.itrs_xyz.m, 1234.0)

def test_deprecated_position_subpoint_method(ts, angle):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    top = iers2010.latlon(angle, angle, elevation_m=0.0)
    b = top.at(t).subpoint()

    error_degrees = abs(b.latitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

    error_degrees = abs(b.longitude.degrees - angle)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

def test_intersection_from_pole(ts):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    p = wgs84.latlon(90.0, 0.0, 1234.0).at(t)
    direction = -p.xyz.au / length_of(p.xyz.au)
    Vector = namedtuple("Vector", "center, target, t")
    vector = Vector(p, direction, t)
    earth_point = wgs84.intersection_of(vector)

    error_degrees = abs(earth_point.latitude.degrees - 90.0)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1
    assert earth_point.elevation.m < 0.1

def test_intersection_from_equator(ts):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    p = wgs84.latlon(0.0, 0.0, 1234.0).at(t)
    direction = -p.xyz.au / length_of(p.xyz.au)
    Vector = namedtuple("Vector", "center, target, t")
    vector = Vector(p, direction, t)
    earth_point = wgs84.intersection_of(vector)

    error_degrees = abs(earth_point.latitude.degrees - 0.0)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1

    error_degrees = abs(earth_point.longitude.degrees - 0.0)
    error_mas = 60.0 * 60.0 * 1000.0 * error_degrees
    assert error_mas < 0.1
    assert earth_point.elevation.m < 0.1

def test_limb_intersection_points(ts):
    t = ts.utc(2018, 1, 19, 14, 37, 55)
    d = 100.0
    a = wgs84.radius.au
    c = a * (1.0 - 1.0 / wgs84.inverse_flattening)
    pos = ICRF(position_au=[d, 0.0, 0.0], t=t, center=399)

    # Vectors pointing to the polar and equatorial limbs of the Earth
    direction_bottom_tangent = [-d, 0.0, -c] / sqrt(d**2 + c**2)
    direction_top_tangent = [-d, 0.0, c] / sqrt(d**2 + c**2)
    direction_left_tangent = [-d, -a, 0.0] / sqrt(d**2 + c**2)
    direction_right_tangent = [-d, a, 0.0] / sqrt(d**2 + c**2)
    Vector = namedtuple("Vector", "center, target, t")
    bottom_tangent = Vector(pos, direction_bottom_tangent, t)
    top_tangent = Vector(pos, direction_top_tangent, t)
    left_tangent = Vector(pos, direction_left_tangent, t)
    right_tangent = Vector(pos, direction_right_tangent, t)
    # Attitude vector pointing straight down
    zenith = Vector(pos, [-1.0, 0.0, 0.0], t)

    intersection_bottom = wgs84.intersection_of(bottom_tangent)
    intersection_top = wgs84.intersection_of(top_tangent)
    intersection_left = wgs84.intersection_of(left_tangent)
    intersection_right = wgs84.intersection_of(right_tangent)
    intersection_zenith = wgs84.intersection_of(zenith)

    # Viewed from sufficient distance, points of intersection should be nearly
    # tangent to the north and south poles, and the zenith longitude +/- 90.0
    zenith_lon = intersection_zenith.longitude.degrees
    left_limb_lon = zenith_lon - 90.0
    right_limb_lon = zenith_lon + 90.0

    error_degrees = abs(intersection_bottom.latitude.degrees + 90.0)
    assert error_degrees < 0.1
    assert intersection_bottom.elevation.m < 0.1

    error_degrees = abs(intersection_top.latitude.degrees - 90.0)
    assert error_degrees < 0.1
    assert intersection_top.elevation.m < 0.1

    error_degrees = abs(intersection_left.latitude.degrees - 0.0)
    assert error_degrees < 0.1

    error_degrees = abs(intersection_left.longitude.degrees - left_limb_lon)
    assert error_degrees < 0.1
    assert intersection_left.elevation.m < 0.1

    error_degrees = abs(intersection_right.latitude.degrees - 0.0)
    assert error_degrees < 0.1

    error_degrees = abs(intersection_right.longitude.degrees - right_limb_lon)
    assert error_degrees < 0.1
    assert intersection_right.elevation.m < 0.1

