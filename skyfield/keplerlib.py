from __future__ import division

import sys
import math
from numpy import (
    abs, amax, amin, arange, arccos, arctan, array, atleast_1d,
    clip, copy, copyto, cos, cosh, exp, full_like, log, ndarray, newaxis,
    pi, power, repeat, reshape, sign, sin, sinh, squeeze, sqrt, sum,
    tan, tanh, zeros_like,
)
from skyfield.constants import AU_KM, DAY_S, DEG2RAD
from skyfield.functions import dots, length_of, mxv
from skyfield.descriptorlib import reify
from skyfield.elementslib import OsculatingElements, normpi
from skyfield.units import Distance, Velocity
from skyfield.vectorlib import VectorFunction
from skyfield.sgp4lib import _cross

_CONVERT_GM = DAY_S * DAY_S / AU_KM / AU_KM / AU_KM

class _KeplerOrbit(VectorFunction):
    def __init__(self,
                 position,
                 velocity,
                 epoch,
                 mu_au3_d2,
                 center=None,
                 target=None,
        ):
        """ Calculates the position of an object using 2 body propagation

        Parameters
        ----------
        position : Distance
            Position vector at epoch with shape (3,)
        velocity : Velocity
            Velocity vector at epoch with shape (3,)
        epoch : Time
            Time corresponding to `position` and `velocity`
        mu_au_d : float
            Value of mu (G * M) in au^3/d^2
        center : int
            NAIF ID of the primary body, 399 for geocentric orbits, 10 for
            heliocentric orbits
        target : int
            NAIF ID of the secondary body
        """
        self.position_at_epoch = position
        self.velocity_at_epoch = velocity
        self.epoch = epoch
        self.mu_au3_d2 = mu_au3_d2
        self.center = center
        self.target = target

        self._rotation = None  # TODO: make argument?

    @classmethod
    def _from_periapsis(
            cls,
            semilatus_rectum_au,
            eccentricity,
            inclination_degrees,
            longitude_of_ascending_node_degrees,
            argument_of_perihelion_degrees,
            t_periapsis,
            gm_km3_s2,
            center=None,
            target=None,
        ):
        """Build a `KeplerOrbit` given its parameters and date of periapsis."""
        gm_au3_d2 = gm_km3_s2 * _CONVERT_GM
        pos, vel = ele_to_vec(
            semilatus_rectum_au,
            eccentricity,
            DEG2RAD * inclination_degrees,
            DEG2RAD * longitude_of_ascending_node_degrees,
            DEG2RAD * argument_of_perihelion_degrees,
            0.0,
            gm_au3_d2,
        )
        return cls(
            Distance(pos),
            Velocity(vel),
            t_periapsis,
            gm_au3_d2,
            center,
            target,
        )

    @classmethod
    def _from_true_anomaly(cls, p, e, i, Om, w, v,
                          epoch,
                          mu_km_s=None,
                          mu_au3_d2=None,
                          center=None,
                          target=None,
        ):
        """ Creates a `KeplerOrbit` object from elements using true anomaly

        Parameters
        ----------
        p : Distance
            Semi-Latus Rectum
        e : float
             Eccentricity
        i : Angle
            Inclination
        Om : Angle
            Longitude of Ascending Node
        w : Angle
            Argument of periapsis
        v : Angle
            True anomaly
        epoch : Time
            Time corresponding to `position` and `velocity`
        mu_km_s : float
            Value of mu (G * M) in km^3/s^2
        mu_au3_d2 : float
            Value of mu (G * M) in au^3/d^2
        center : int
            NAIF ID of the primary body, 399 for geocentric orbits, 10 for
            heliocentric orbits
        target : int
            NAIF ID of the secondary body
        """
        if (mu_km_s and mu_au3_d2) or (not mu_km_s and not mu_au3_d2):
            raise ValueError('Either mu_km_s or mu_au3_d2 should be used, but not both')

        if mu_au3_d2:
            mu_km_s = mu_au3_d2 * AU_KM**3 / DAY_S**2

        position, velocity = ele_to_vec(p.km,
                                        e,
                                        i.radians,
                                        Om.radians,
                                        w.radians,
                                        v.radians,
                                        mu_km_s,
        )
        return cls(Distance(km=position),
                   Velocity(km_per_s=velocity),
                   epoch,
                   mu_km_s,
                   center=center,
                   target=target,
        )


    @classmethod
    def _from_mean_anomaly(
            cls,
            semilatus_rectum_au,
            eccentricity,
            inclination_degrees,
            longitude_of_ascending_node_degrees,
            argument_of_perihelion_degrees,
            mean_anomaly_degrees,
            epoch,
            gm_km3_s2,
            center=None,
            target=None,
        ):
        """ Creates a `KeplerOrbit` object from elements using mean anomaly

        Parameters
        ----------
        p : Distance
            Semi-Latus Rectum
        e : float
             Eccentricity
        i : Angle
            Inclination
        Om : Angle
            Longitude of Ascending Node
        w : Angle
            Argument of periapsis
        M : Angle
            Mean anomaly
        epoch : Time
            Time corresponding to `position` and `velocity`
        mu_km_s : float
            Value of mu (G * M) in km^3/s^2
        mu_au3_d2 : float
            Value of mu (G * M) in au^3/d^2
        center : int
            NAIF ID of the primary body, 399 for geocentric orbits, 10 for
            heliocentric orbits
        target : int
            NAIF ID of the secondary body
        """
        M = DEG2RAD * mean_anomaly_degrees
        gm_au3_d2 = gm_km3_s2 * _CONVERT_GM
        if eccentricity < 1.0:
            E = eccentric_anomaly(eccentricity, M)
            v = true_anomaly_closed(eccentricity, E)
        elif eccentricity > 1.0:
            E = eccentric_anomaly(eccentricity, M)
            v = true_anomaly_hyperbolic(eccentricity, E)
        else:
            v = true_anomaly_parabolic(semilatus_rectum_au, gm_au3_d2, M)

        pos, vel = ele_to_vec(
            semilatus_rectum_au,
            eccentricity,
            DEG2RAD * inclination_degrees,
            DEG2RAD * longitude_of_ascending_node_degrees,
            DEG2RAD * argument_of_perihelion_degrees,
            v,
            gm_au3_d2,
        )
        return cls(
            Distance(pos),
            Velocity(vel),
            epoch,
            gm_au3_d2,
            center,
            target,
        )

    def _at(self, time):
        """Propagate the KeplerOrbit to the given Time object

        The Time object can contain one time, or an array of times
        """
        pos, vel = propagate(
            self.position_at_epoch.au,
            self.velocity_at_epoch.au_per_d,
            self.epoch.tt,
            time.tt,
            self.mu_au3_d2,
        )
        if self._rotation is not None:
            pos = mxv(self._rotation, pos)
            vel = mxv(self._rotation, vel)
        return pos, vel, None, None


    @reify
    def elements_at_epoch(self):
        return OsculatingElements(self.position_at_epoch,
                                  self.velocity_at_epoch,
                                  self.epoch,
                                  mu_km_s = self.mu_au3_d2 / _CONVERT_GM,
        )


    def __str__(self):
        ele = self.elements_at_epoch
        if self.target_name:
            return 'KeplerOrbit {0} {1} -> {2} {3}'.format(self.center,
                                                           self.center_name,
                                                           self.target,
                                                           self.target_name,
            )
        else:
            ele = self.elements_at_epoch
            string = 'KeplerOrbit {0} {1} -> q={2:.2}au e={3:.3f} i={4:.1f} Om={5:.1f} w={6:.1f}'
            return string.format(self.center,
                                 self.center_name,
                                 ele.periapsis_distance.au,
                                 ele.eccentricity,
                                 ele.inclination.degrees,
                                 ele.longitude_of_ascending_node.degrees,
                                 ele.argument_of_periapsis.degrees,
            )


    def __repr__(self):
        return '<{0}>'.format(str(self))

_ten_iterations = tuple([None] * 10)

def eccentric_anomaly(e, M):
    """Iterate to solve Kepler's equation to find the eccentric anomaly.

    See arXiv:2108.03215.

    """
    M = normpi(M)
    sign_M = sign(M)
    M *= sign_M
    ebar = 0.25 * pi/e - 1.0
    E = 0.5 * pi * ebar * (sign(ebar) * sqrt(1 + M/(e*ebar*ebar)) - 1.0)

    for _ in _ten_iterations:
        f1 = 1.0 - e*cos(E)
        f2 = e*sin(E)
        f = E - f2 - M
        dE = f*f1 / (f1*f1 - 0.5*f*f2)
        E -= dE
        if abs(dE) < 1e-14:
            return E * sign_M

    raise ValueError('eccentric anomaly failed to converge')

def true_anomaly_hyperbolic(e, E):
    """Calculates true anomaly from eccentricity and eccentric anomaly.

    Valid for hyperbolic orbits. Equations from the relevant Wikipedia entries.

    """
    return 2.0 * arctan(sqrt((e + 1.0) / (e - 1.0)) * tanh(E/2))


def true_anomaly_closed(e, E):
    """Calculates true anomaly from eccentricity and eccentric anomaly.

    Valid for closed orbits. Equations from the relevant Wikipedia entries.

    """
    return 2.0 * arctan(sqrt((1.0 + e) / (1.0 - e)) * tan(E/2))


def true_anomaly_parabolic(p, gm, M):
    """Calculates true anomaly from semi-latus rectum, gm, and mean anomaly.

    Valid for parabolic orbits. Equations from
    https://en.wikipedia.org/wiki/Parabolic_trajectory.

    """
    delta_t = sqrt(2 * p**3 / gm) * M # from http://www.bogan.ca/orbits/kepler/orbteqtn.html
    periapsis_distance = p / 2
    A = 3 / 2 * sqrt(gm / (2 * periapsis_distance**3)) * delta_t
    B = (A + (A*A + 1))**(1/3)
    return 2 * arctan(B - 1/B)


def ele_to_vec(p, e, i, Om, w, v, mu):
    """Calculates state vectors from orbital elements. Also checks for invalid
    sets of elements.

    Based on equations from this document:

    https://web.archive.org/web/*/http://ccar.colorado.edu/asen5070/handouts/kep2cart_2002.doc
    """
    # Checks that true anomaly is less than arccos(-1/e) for hyperbolic orbits
    if isinstance(e, ndarray) and isinstance(v, ndarray):
        inds = (e>1)
        if (v[inds]>arccos(-1/e[inds])).any():
            raise ValueError('If eccentricity is >1, abs(true anomaly) cannot be more than arccos(-1/e)')
    elif isinstance(e, ndarray) and not isinstance(v, ndarray):
        inds = (e>1)
        if (v>arccos(-1/e[inds])).any():
            raise ValueError('If eccentricity is >1, abs(true anomaly) cannot be more than arccos(-1/e)')
    elif isinstance(v, ndarray) and not isinstance(e, ndarray):
        if e>1 and (v>arccos(-1/e)).any():
            raise ValueError('If eccentricity is >1, abs(true anomaly) cannot be more than arccos(-1/e)')
    else:
        if e>1 and v>arccos(-1/e):
            raise ValueError('If eccentricity is >1, abs(true anomaly) cannot be more than arccos(-1/e)')

    # Checks that inclination is in the range [0, pi]
    if isinstance(i, ndarray):
        if not ((i>=0) * (i <= pi)).all():
            raise ValueError('Inclination outside the range [0, pi] radians')
    else:
        if not 0 <= i <= pi:
            raise ValueError('Inclination outside the range [0, pi] radians')

    r = p/(1 + e*cos(v))
    h = sqrt(p*mu)
    u = v+w

    X = r*(cos(Om)*cos(u) - sin(Om)*sin(u)*cos(i))
    Y = r*(sin(Om)*cos(u) + cos(Om)*sin(u)*cos(i))
    Z = r*(sin(i)*sin(u))

    X_dot = X*h*e/(r*p)*sin(v) - h/r*(cos(Om)*sin(u) + sin(Om)*cos(u)*cos(i))
    Y_dot = Y*h*e/(r*p)*sin(v) - h/r*(sin(Om)*sin(u) - cos(Om)*cos(u)*cos(i))
    Z_dot = Z*h*e/(r*p)*sin(v) + h/r*sin(i)*cos(u)

    # z and z_dot are independent of Om, so if Om is an array and the other
    # elements are scalars, z and z_dot need to be repeated
    if Z.size!=X.size:
        Z = repeat(Z, X.size)
        Z_dot = repeat(Z_dot, X.size)

    return array([X, Y, Z]), array([X_dot, Y_dot, Z_dot])


dpmax = sys.float_info.max


def find_trunc():
    denom = 2
    factr = 2
    trunc = 1
    x = 1 / denom
    while 1+x > 1:
        denom = denom * (2+factr) * (1+factr)
        factr = factr + 2
        trunc = trunc + 1
        x = 1 / denom
    return trunc


trunc = find_trunc()
odd_factorials = array([math.factorial(i) for i in range(3, trunc*2, 2)])
even_factorials = array([math.factorial(i) for i in range(2, trunc*2, 2)])
exponents = arange(0, trunc-1)
stumpff_bound = -(log(2) + log(dpmax))**2

def stumpff(x):
    """Calculates Stumpff functions

    Based on the function toolkit/src/spicelib/stmp03.f from the SPICE toolkit,
    which can be downloaded from naif.jpl.nasa.gov/naif/toolkit_FORTRAN.html
    """
    if x.min() < stumpff_bound:
        raise ValueError('Argument below lower bound')

    z = sqrt(abs(x))

    c0 = zeros_like(x)
    c1 = zeros_like(x)
    c2 = zeros_like(x)
    c3 = zeros_like(x)

    low = x < -1
    c0[low] = cosh(z[low])
    c1[low] = sinh(z[low])/z[low]

    high = x > 1
    c0[high] = cos(z[high])
    c1[high] = sin(z[high])/z[high]

    mid = ~(low|high)
    if sum(mid):
        numerators = repeat(x[mid][:, newaxis], trunc-1, axis=1)
        numerators[:, 1::2] *= -1
        c3[mid] = sum(power(numerators, exponents)/odd_factorials, axis=1)
        c2[mid] = sum(power(numerators, exponents)/even_factorials, axis=1)

        c1[mid] = 1 - x[mid]*c3[mid]
        c0[mid] = 1 - x[mid]*c2[mid]

    not_mid = ~mid
    c2[not_mid] = (1 - c0[not_mid])/x[not_mid]
    c3[not_mid] = (1 - c1[not_mid])/x[not_mid]

    return c0, c1, c2, c3

def propagate(position, velocity, t0, t1, gm):
    """Propagates a position and velocity vector with an array of times.

    Based on the function toolkit/src/spicelib/prop2b.f from the SPICE toolkit,
    which can be downloaded from naif.jpl.nasa.gov/naif/toolkit_FORTRAN.html

    Parameters
    ----------
    position : ndarray
       Position vector with shape (3,)
    velocity : ndarray
        Velocity vector with shape (3,)
    t0 : float
        Time corresponding to `position` and `velocity`
    t1 : float or ndarray
        Time or times to propagate to
    gm : float
        Gravitational parameter in units that match the other arguments
    """
    if getattr(t1, 'shape', 0) and t1.shape[0] == 1:
        expected_shape= (3, t1.shape[0])
    else:
        expected_shape= None

    gm = atleast_1d(gm)
    if (gm <= 0).any():
        raise ValueError("'gm' should be positive")
    if (length_of(velocity)).any() == 0:
        raise ValueError('Velocity vector has zero magnitude')
    if (length_of(position)).any() == 0:
        raise ValueError('Position vector has zero magnitude')

    if position.ndim == 1:
        position = position[:, newaxis]
    if velocity.ndim == 1:
        velocity = velocity[:, newaxis]

    r0 = length_of(position)
    rv = dots(position, velocity)

    hvec = _cross(position, velocity)
    h2 = dots(hvec, hvec)

    if (h2 == 0).any():
        raise ValueError('Motion is not conical')

    eqvec = _cross(velocity, hvec)/gm + -position/r0
    e = length_of(eqvec)
    q = h2 / (gm * (1+e))

    f = 1 - e
    b = sqrt(q/gm)

    br0 = b * r0
    b2rv = b * b * rv
    bq = b * q
    qovr0 = q / r0


    maxc = amax(array([abs(br0),
               abs(b2rv),
               abs(bq),
               abs(qovr0/bq)]), axis=0)

    hyperbolic = (f<0)
    bound = zeros_like(f)

    fixed = log(dpmax/2) - log(maxc[hyperbolic])
    rootf = sqrt(-f[hyperbolic])
    logf = log(-f[hyperbolic])
    bound[hyperbolic] = amin(array([fixed/rootf, (fixed + 1.5*logf)/rootf]), axis=0)

    logbound = (log(1.5) + log(dpmax) - log(maxc[~hyperbolic])) / 3
    bound[~hyperbolic] = exp(logbound)

    # each of these arrays has 1 entry per orbit, so its shape is (#orbits, 1)
    f = f[:, newaxis]
    bq = bq[:, newaxis]
    b2rv = b2rv[:, newaxis]
    br0 = br0[:, newaxis]
    qovr0 = qovr0[:, newaxis]
    bound = bound[:, newaxis]

    def kepler(x):
        _, c1, c2, c3 = stumpff(f*x*x)
        return x*(br0*c1 + x*(b2rv*c2 + x*bq*c3))

    def kepler_1d(x, orb_inds):
        _, c1, c2, c3 = stumpff(x*x*repeat(f, orb_inds))
        return x*(c1*repeat(br0, orb_inds) + x*(c2*repeat(b2rv, orb_inds) + x*(c3*repeat(bq, orb_inds))))


    t1 = atleast_1d(t1)
    t0 = atleast_1d(t0)
    if len(t0) == 1:
        t0 = repeat(t0, position.shape[1])

    # shape of 2 dimensional arrays from here on out should be (#orbits, len(t1))
    dt = t1 - t0[:, newaxis]

    x = dt/bq
    copyto(x, -bound, where=(x<-bound))
    copyto(x, bound, where=(x>bound))

    kfun = kepler(x)

    past = dt < 0
    future = dt > 0

    upper = zeros_like(dt, dtype='float64')
    lower = zeros_like(dt, dtype='float64')
    oldx = zeros_like(dt, dtype='float64')

    copyto(lower, x, where=past)
    copyto(upper, x, where=future)

    while (kfun[past] > dt[past]).any():
        copyto(upper, lower, where=past)
        lower[past] *= 2
        copyto(oldx, x, where=past)
        orb_ind = sum(past, axis=1)
        x[past] = clip(lower[past], repeat(-bound, orb_ind), repeat(bound, orb_ind))
        if (x[past] == oldx[past]).any():
            raise ValueError('The input delta time (dt) has a value of {0}.'
                             'This is beyond the range of DT for which we '
                             'can reliably propagate states. The limits for '
                             'this GM and initial state are from {1}'
                             'to {2}.'.format(dt, kepler(-bound), kepler(bound)))
        kfun[past] = kepler_1d(x[past], orb_ind)

    while (kfun[future] < dt[future]).any():
        copyto(lower, upper, where=future)
        upper[future] *= 2
        copyto(oldx, x, where=future)
        orb_ind = sum(future, axis=1)
        x[future] = clip(upper[future], repeat(-bound, orb_ind), repeat(bound, orb_ind))
        if (x[future] == oldx[future]).any():
            raise ValueError('The input delta time (dt) has a value of {0}.'
                             'This is beyond the range of DT for which we '
                             'can reliably propagate states. The limits for '
                             'this GM and initial state are from {1} '
                             'to {2}.'.format(dt, kepler(-bound), kepler(bound)))
        kfun[future] = kepler_1d(x[future], orb_ind)

    x = copy(upper)
    copyto(x, (upper+lower)/2, where=(lower<=upper))

    lcount = zeros_like(dt)
    mostc = full_like(dt, 1000)
    not_done = (lower < x) & (x < upper)

    while not_done.any():
        orb_inds = sum(not_done, axis=1)
        kfun[not_done] = kepler_1d(x[not_done], orb_inds)

        high = (kfun > dt) & not_done
        low = (kfun < dt) & not_done
        same = (~high & ~low) & not_done

        copyto(upper, x, where=(high|same))
        copyto(lower, x, where=(low|same))

        condition = not_done & (mostc > 64) & (upper != 0) & (lower != 0)
        mostc[condition] = 64
        lcount[condition] = 0

        copyto(x, upper, where=(not_done & (lower>upper)))
        copyto(x, (upper+lower)/2, where=(not_done & (lower<=upper)))

        lcount += 1
        not_done = (lower < x) & (x < upper) & (lcount < mostc)

    c0, c1, c2, c3 = stumpff(f*x*x)
    br = br0*c0 + x*(b2rv*c1 + x*bq*c2)

    pc = 1 - qovr0 * x * x * c2
    vc = dt - bq * x**3 * c3
    pcdot = -qovr0 / br * x * c1
    vcdot = 1 - bq / br * x * x * c2

    position_prop = pc[newaxis, :, :]*position[:, :, newaxis] + vc[newaxis, :, :]*velocity[:, :, newaxis]
    velocity_prop = pcdot[newaxis, :, :]*position[:, :, newaxis] + vcdot[newaxis, :, :]*velocity[:, :, newaxis]

    if expected_shape:
        return reshape(squeeze(position_prop), expected_shape ), reshape(squeeze(velocity_prop), expected_shape )

    return squeeze(position_prop), squeeze(velocity_prop)
