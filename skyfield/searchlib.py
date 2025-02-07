"""Routines to search for maxima and zero crossings."""

from __future__ import print_function, division

from numpy import (add, append, argsort, bool_, concatenate, diff, flatnonzero,
                   int8, issubdtype, linspace, multiply, reshape, sign)
from .constants import DAY_S
EPSILON = 0.001 / DAY_S

_trace = None  # User can replace with a routine to save search iterations.

def find_discrete(start_time, end_time, f, epsilon=EPSILON, num=12):
    """Find the times at which a discrete function of time changes value.

    This routine is used to find instantaneous events like sunrise,
    transits, and the seasons.  See :doc:`searches` for how to use it
    yourself.

    """
    ts = start_time.ts
    jd0 = start_time.tt
    jd1 = end_time.tt
    if jd0 >= jd1:
        raise ValueError('your start_time {0} is later than your end_time {1}'
                         .format(start_time, end_time))

    step_days = getattr(f, 'step_days', None)
    if step_days is None:
        # Legacy "rough_period" attribute.
        rough_period = getattr(f, 'rough_period', None)
        if rough_period is None:
            raise AttributeError('the function you have passed'
                                 ' is missing a "step_days" attribute')
        periods = (jd1 - jd0) / rough_period
        if periods < 1.0:
            periods = 1.0
        sample_count = int(periods * num)
    else:
        # Insist on at least 2 samples even if the dates are less than
        # step_days apart, so the range at least has endpoints.
        sample_count = int((jd1 - jd0) / step_days) + 2

    jd = linspace(jd0, jd1, sample_count)
    return _find_discrete(ts, jd, f, epsilon, num)

# TODO: pass in `y` so it can be precomputed?

def _find_discrete(ts, jd, f, epsilon, num):
    """Algorithm core, for callers that already have a `jd` vector."""
    end_mask = linspace(0.0, 1.0, num)
    start_mask = end_mask[::-1]
    o = multiply.outer

    while True:
        t = ts.tt_jd(jd)
        y = f(t)

        indices = flatnonzero(diff(y))
        if not len(indices):
            # Nothing found, so immediately return empty arrays.
            ends = jd.take(indices)
            y = y.take(indices)
            break

        starts = jd.take(indices)
        ends = jd.take(indices + 1)

        # Since we start with equal intervals, they all should fall
        # below epsilon at around the same time; so for efficiency we
        # only test the first pair.
        if ends[0] - starts[0] <= epsilon:
            y = y.take(indices + 1)
            # Keep only the last of several zero crossings that might
            # possibly be separated by less than epsilon.
            mask = concatenate(((diff(ends) > epsilon), (True,)))
            ends = ends[mask]
            y = y[mask]
            break

        jd = o(starts, start_mask).flatten() + o(ends, end_mask).flatten()

    return ts.tt_jd(ends), _fix_numpy_deprecation(y)

def find_minima(start_time, end_time, f, epsilon=1.0 / DAY_S, num=12):
    """Find the local minima in the values returned by a function of time.

    This routine is used to find events like minimum elongation.  See
    :doc:`searches` for how to use it yourself.

    """
    def g(t): return -f(t)
    g.rough_period = getattr(f, 'rough_period', None)
    g.step_days = getattr(f, 'step_days', None)
    t, y = find_maxima(start_time, end_time, g, epsilon, num)
    return t, _fix_numpy_deprecation(-y)

def find_maxima(start_time, end_time, f, epsilon=1.0 / DAY_S, num=12):
    """Find the local maxima in the values returned by a function of time.

    This routine is used to find events like highest altitude and
    maximum elongation.  See :doc:`searches` for how to use it yourself.

    """
    #    @@       @@_@@       @@_@@_@@_@@
    #   /  \     /     \     /           \
    # @@    @@ @@       @@ @@             @@
    # +1 -1    +1  0 -1    +1  0  0  0 -1    sd = sign(diff(y))
    # -2       -1 -1       -1  0  0 -1       diff(sign(diff(y))

    ts = start_time.ts
    jd0 = start_time.tt
    jd1 = end_time.tt

    if jd0 >= jd1:
        raise ValueError('start_time {0} is not earlier than end_time {1}'
                         .format(start_time, end_time))

    # We find maxima by investigating every point that is higher than
    # both points next to it.  This presents a problem: if the initial
    # heights are, for example, [1.7, 1.1, 0.3, ...], there might be a
    # maximum 1.8 hidden between the first two heights, but it would not
    # meet the criteria for further investigation because we can't see
    # whether the curve is on its way up or down to the left of 1.7.  So
    # we put an extra point out beyond each end of our range, then
    # filter our final result to remove maxima that fall outside the
    # range.
    step_days = getattr(f, 'step_days', None)
    if step_days is None:
        bump = f.rough_period / num
        bumps = int((jd1 - jd0) / bump) + 3
        jd = linspace(jd0 - bump, jd1 + bump, bumps)
    else:
        # Insist on at least 3 samples, even for very close dates; and
        # add 2 more to stand outside the range.
        steps = int((jd1 - jd0) / step_days) + 3
        real_step = (jd1 - jd0) / steps
        jd = linspace(jd0 - real_step, jd1 + real_step, steps + 2)

    end_alpha = linspace(0.0, 1.0, num)
    start_alpha = end_alpha[::-1]
    o = multiply.outer

    while True:
        t = ts.tt_jd(jd)
        y = f(t)

        # Since we start with equal intervals, they all should fall
        # below epsilon at around the same time; so for efficiency we
        # only test the first pair.
        if t[1] - t[0] <= epsilon:
            jd, y = _identify_maxima(jd, y)

            # Filter out maxima that fell slightly outside our bounds.
            keepers = (jd >= jd0) & (jd <= jd1)
            jd = jd[keepers]
            y = y[keepers]

            # Keep only the first of several maxima that are separated
            # by less than epsilon.
            if len(jd):
                mask = concatenate(((True,), diff(jd) > epsilon))
                jd = jd[mask]
                y = y[mask]

            break

        left, right = _choose_brackets(y)

        if _trace is not None:
            _trace((t, y, left, right))

        if not len(left):
            # No maxima found.
            jd = y = y[0:0]
            break

        starts = jd.take(left)
        ends = jd.take(right)

        jd = o(starts, start_alpha).flatten() + o(ends, end_alpha).flatten()
        jd = _remove_adjacent_duplicates(jd)

    return ts.tt_jd(jd), _fix_numpy_deprecation(y)

def _choose_brackets(y):
    """Return the indices between which we should search for maxima of `y`."""
    dsd = diff(sign(diff(y)))
    indices = flatnonzero(dsd < 0)
    left = reshape(add.outer(indices, [0, 1]), -1)
    left = _remove_adjacent_duplicates(left)
    right = left + 1
    return left, right

def _identify_maxima(x, y):
    """Return the maxima we can see in the series y as simple points."""
    dsd = diff(sign(diff(y)))

    # Choose every point that is higher than the two adjacent points.
    indices = flatnonzero(dsd == -2) + 1
    peak_x = x.take(indices)
    peak_y = y.take(indices)

    # Also choose the midpoint between the edges of a plateau, if both
    # edges are in view.  First we eliminate runs of zeroes, then look
    # for adjacent -1 values, then map those back to the main array.
    indices = flatnonzero(dsd)
    dsd2 = dsd.take(indices)
    minus_ones = dsd2 == -1
    plateau_indices = flatnonzero(minus_ones[:-1] & minus_ones[1:])
    plateau_left_indices = indices.take(plateau_indices)
    plateau_right_indices = indices.take(plateau_indices + 1) + 2
    plateau_x = x.take(plateau_left_indices) + x.take(plateau_right_indices)
    plateau_x /= 2.0
    plateau_y = y.take(plateau_left_indices + 1)

    x = concatenate((peak_x, plateau_x))
    y = concatenate((peak_y, plateau_y))
    indices = argsort(x)
    return x[indices], y[indices]

def _remove_adjacent_duplicates(a):
    if not len(a):
        return a
    mask = diff(a) != 0
    mask = append(mask, [True])
    return a[mask]

def _fix_numpy_deprecation(y):
    # Alas, in the future NumPy will apparently disallow using Booleans
    # as indices, whereas we often encourage the use of `y` as an index.
    if issubdtype(y.dtype, bool_):
        y.dtype = int8
    return y
