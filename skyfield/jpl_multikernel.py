# coding: utf-8
# jpl_multikernel.py
"""
A kernel capable of loading multiple SpiceKernels and combinig their segments.
When a target's position is queried, MultiKerne/MultiSegment picks the appropriate
SPICESegments from all segments available in all loaded SPK files.

Per definition that includes support of ephemerides that have more than 1 segment per planet
"""
from collections import namedtuple, defaultdict
from numpy import array

from jplephem.names import target_name_pairs

from skyfield.api import load
from skyfield.timelib import calendar_tuple
from skyfield.vectorlib import (
    VectorSum, VectorFunction,
    _jpl_code_name_dict, _jpl_name
)
from skyfield.jpllib import (
    _center,
    _jpl_name_code_dict, _format_code_and_name
)

BaseChainTuple = namedtuple('BaseChainTuple', 'start end center chain')

class MultiKernel(object):

    def __init__(self):
        self.kernels = []
        self.kernel_segments = []
        self.codes = set()
        # collect attached MultiSegmenta for notifications
        self._multisegments = []

    def __repr__(self):
        return '<{0}>'.format(type(self).__name__)

    def __str__(self):
        kernels = self.kernels
        lines = ['SPICE multi segment kernel with {0} SPK kernel(s)'
                 .format(len(kernels))]
        for k in kernels:
            lines.append(k.__repr__())
        return '\n'.join(lines)

    def __getitem__(self, target):
        """Return a new MultiSegment object for computing the location of `target`."""
        target = self.decode(target)
        return MultiSegment(self, target)

    def __contains__(self, name_or_code):
        if isinstance(name_or_code, int):
            code = name_or_code
        else:
            code = _jpl_name_code_dict.get(name_or_code.upper())
        return code in self.codes

    def _refresh_kernels(self):
        segments = []
        for k in self.kernels:
            segments.extend(k.segments)
        self.kernel_segments = segments

        codes = set()
        for kernel in self.kernels:
            codes = codes.union(set(
                         s.center for s in kernel.segments).union(
                         s.target for s in kernel.segments)
                     )
        self.codes = codes

    def names(self):
        """Return all target names that are valid with this kernel.

        >>> pprint(multikernel.names())
        {0: ['SOLAR_SYSTEM_BARYCENTER', 'SSB', 'SOLAR SYSTEM BARYCENTER'],
         1: ['MERCURY_BARYCENTER', 'MERCURY BARYCENTER'],
         2: ['VENUS_BARYCENTER', 'VENUS BARYCENTER'],
         3: ['EARTH_BARYCENTER',
             'EMB',
         ...

        The result is a dictionary with target code keys and name lists
        as values.  The last name in each list is the one that Skyfield
        uses when printing information about a body.

        """
        d = defaultdict(list)
        for code, name in target_name_pairs:
            if code in self.codes:
                d[code].append(name)
        return dict(d)


    def decode(self, name):
        """Translate a target name into its integer code and check
        whether its a valid SPICE target.
        Unlike SpiceKernel's ``decode()`` no check is done whethr targt is
        present in (any of) the currently loaded kernels.

        >>> multikernel.decode('Venus')
        299

        Raises ``ValueError`` if you supply a name (string or integer) that's
        not in the jplephem.names list
        """
        if isinstance(name, int):
            code = name
            if _jpl_code_name_dict.get(code) is None:
                raise ValueError('unknown SPICE target {0!r}'.format(name))
        else:
            name = name.upper()
            code = _jpl_name_code_dict.get(name)
            if code is None:
                raise ValueError('unknown SPICE target {0!r}'.format(name))
        return code

    def load_kernel(self, path):
        # TODO: expand the path to an absolute one?
        for k in self.kernels:
            if path == k.path:
                return k
        kernel = load(path)
        self.kernels.append(kernel)
        self._refresh_kernels()
        return kernel;

    def _spicekernel_removed(self, kernel):
        """remove basechains that depended on the closed kernel
        """
        def uses(value, k):
            # value is a BaseChainTuple
            if isinstance(value.chain, VectorSum):
                for func in value.chain.vector_functions:
                    if func.ephemeris == k:
                        return True
            else:
                if value.chain.ephemeris == k:
                    return True
            return False

        for ms in self._multisegments:
            if ms.target not in self.codes:
                ms.reset()
                continue

            remove_keys = [key for key, value in ms.basechains.items()
                            if uses(value, kernel) ]
            for key in remove_keys:
                ms.basechains.pop(key, 0)

    def close_kernel(self, kernel):
        if kernel not in self.kernels:
            return

        self.kernels.remove(kernel)
        self._refresh_kernels()
        self._spicekernel_removed(kernel)

        kernel.close()

    def close(self):
        for k in self.kernels:
            k.close()
        self.kernels.clear()
        self.codes.clear()
        for ms in self._multisegments:
            ms.reset()
        self._multisegments.clear()

    def _pick_basechaintuple(self, target, tdb):
        """Return a BaseChainTuple -- start, end, center, vector function -- for computing
        the location of `target`at tdb.
        """
        # check for existing target
        target = self.decode(target)
        if target not in self.codes:
            if len(self.codes) == 0:
                raise KeyError('multikernel contains no targets at all. Call `load_kernel()`')
            targets = ', '.join(_format_code_and_name(c) for c in self.codes )
            raise KeyError('no target {0!r} in multikernel -'
                           ' the targets it supports are: {1}'
                           .format(target, targets))

        # collect segments matching the time
        # last loaded segment wins
        segments = self.kernel_segments
        segment_dict = dict()
        for segment in segments:
            if segment.spk_segment.start_jd <= tdb and tdb < segment.spk_segment.end_jd:
                segment_dict[segment.target] = segment

        # no need to call ''_center()'' when there's no target
        if target not in segment_dict:
            format_datetime = '{0}-{1:02}-{2:02} {3:02}:{4:02}:{5:02.0f}'.format
            raise ValueError('multikernel has no segments for target "{0}" at TBD {1}'
                                .format(
                                _jpl_name(target) ,
                                format_datetime(*calendar_tuple(tdb))
                            ))

        # try to find a chain of VectorFunctions ending at target
        chain = tuple(_center(target, segment_dict))

        # obtain smallest time range covered by the segments
        starters = [seg.spk_segment.start_jd for seg in chain]
        enders = [seg.spk_segment.end_jd for seg in chain]
        start = max(starters)
        end   = min(enders)

        if len(chain) == 1:
            result = chain[0]
        else:
            chain = chain[::-1]  # revert the chain
            center = chain[0].center
            target = chain[-1].target
            result = VectorSum(center, target, chain)

        return BaseChainTuple(start, end, result.center, result)

class MultiSegment(VectorFunction):
    """
    A meta-segment capable of picking and combining the appropriate SPICESegments
    available in a MultiKernel.

    Properties of interest:

    * segment.target - any valid SPICE target

      You may create MultiSegments in advance for any valid SPICE target
      no matter whether it's already present in the segment.kernel.
      You won't know whether the target's position can be calculated (at a given time)
      until you call `at()` or `observe()` for the first time.

    * segment.basechains - dictionary of collected VectorFunctions
      key is (start, end), values are BaseChainTuples

      Each successful call to ``at()`` or ``observe()`` corresponds to a
      particular BaseChainTuple.

      If there isn't a basechain for the requested time yet it will be determined
      from the SPICESegments present in the segment.kernel *at that moment*.
      Otherwise it's re-used.

      To reset the basechains, call segment.reset()

    * segment._fallback_target - BARYCENTER of the target (if it has one)

      Builtin automatic switch to the BARYCENTER in case the target id not present
      was invented in order not to break Astrometric.apparent() -> add_deflection()

      It's also convenient for outer planets in DExxx SPKs.
      Starting with the DE43x MARS was removed.
      Dunno whether DExxx ever contained the other outer planets .
      Setting ._fallback_target to None will suppress the switch.
      But why should you?

    MultiSegment also has a .center. But unlike the fixed .center of a
    SPICESegment it follows the center of the last used basechain.
    Which is not guaranteed to be 0 all the time ;)

    """

    def __init__(self, multikernel, target):
        self.kernel = multikernel
        self.kernel._multisegments.append(self)

        # Translate the target name into its integer code.
        self.target = multikernel.decode(target)
        # BARYCENTER to the rescue
        myname = _jpl_code_name_dict.get(self.target)
        self._fallback_target = _jpl_name_code_dict.get(myname + ' BARYCENTER')

        self.basechains = dict()

        # ``ephmeris`` is required for Astrometric.apparent()
        # The 'ephemeris' doesn't have to be a SpiceKernel.
        # Calling ``ephemeris[target]`` must return an object (or VectorFunction)
        # with a proper ``._at(t)`` function.for
        # 'sun', 'jupiter', 'saturn', 'moon', 'venus', 'uranus', 'neptune' and 'earth'
        # See skyfield.relativity.
        self.ephemeris = self.kernel

        # trick 17 needed for _observe_from_bcrs()
        self._barychain_required = False
        self.center = None

    def reset(self):
        self.basechains.clear()

    # self.center may change but self.center_name() is reified
    def _set_center(self, newcenter):
        if newcenter != self.center:
            self.center_name = _jpl_name(newcenter)
        self.center = newcenter

    def _get_chain_for(self, tdb):
        for start, end in self.basechains:
            if start <= tdb and tdb < end:
                return self.basechains[(start, end)]

        try:
            basechain = self.kernel._pick_basechaintuple(self.target, tdb)
        except Exception as e:
            if self._fallback_target and self._fallback_target in self.kernel:
                # let's try the BARYCENTER
                basechain = self.kernel._pick_basechaintuple(self._fallback_target, tdb)
            else:
                raise e
        self.basechains[(basechain.start, basechain.end)] = basechain
        return basechain

    def _generate_from_chain_dict(self, t, chain_dict):
        # collect positions and rates
        positions = []
        rates = []
        for time in t:
            tdb = time.tdb
            basechain = chain_dict[tdb]
            p, v, _, _ = basechain.chain._at(time)
            positions.append(p)
            rates.append(v)

        # convert to nparray of  x- ,y- ,z-rows
        yield array([row for row in zip(*positions) ])
        yield array([row for row in zip(*rates) ])

    def _observe_from_bcrs(self, observer):
        """
        super checks for self.center being SSB
        Since the current center is spotted in ``_at()`` it's possible that we don't
        have a center yet, or it's not the center we get for the observer's Time

        workaround:
        - set center to 0
        - verify SSB in _at()
        """
        self._barychain_required = True
        self._set_center(0)
        p, v, t, light_time = super(MultiSegment, self)._observe_from_bcrs(observer)
        self._barychain_required = False
        return p, v, t, light_time

    def _at(self, t):
        """Compute the target's position at time ``t``

        Raises KeyError when the target isn' found in the MultiKernel.
        Raises ValueError when the MultiKernel doesn't contain a chain for the
        target at the requsted time,

        If ``t`` is an array of times, the basechains must have the same center.
        Otherwise a ValueError is raised.

        Requirement for Time arrays is of course, that every ``VectorFunction._at()``
        resp. ``Segment.compute_and_differentiate()`` involved supports Time arrays.
        """
        tdb = t.tdb
        scalar = not getattr(t.whole, 'shape', 0) and not getattr(t.tdb_fraction, 'shape', 0)
        if scalar:
            tdb = array((tdb,))

        # Obtain the basechain(s).
        # Take care for Time array aiming at different chains
        required_chains = dict()
        chain_dict = dict()
        centers = set()
        for x in tdb:
            basechain = self._get_chain_for(x)
            chain_dict[x] = basechain
            required_chains[(basechain.start, basechain.end)] = basechain
            # should we break and raise as soon as a second center is detected?
            centers.add(basechain.center)

        # same center is required!
        if len(centers) != 1:
            format_datetime = '{0}-{1:02}-{2:02} {3:02}:{4:02}:{5:02.0f}'.format
            lines = []
            for (start, end), basechain in required_chains.items():
                lines.append('{0} - {1}: center = {2}'.format(
                        format_datetime(*calendar_tuple(start)),
                        format_datetime(*calendar_tuple(end)),
                        _jpl_name(basechain.center)
                ))
            raise ValueError('Multiple chains with different centers not possible.\n'
                 '{0} chains with {1} centers are required for the requested Time array:\n'
                 '{2}'
                .format(len(required_chains), len(centers),
                       '\n'.join(lines))
            )

        # subsequent processing requires a center
        for center in centers:
            self._set_center(center)
            break

        if self._barychain_required and self.center != 0:
            raise ValueError('you can only observe() a body whose vector'
                            ' center is the Solar System Barycenter,'
                            ' but this vector has the center {0}'
                            .format(self.center_name))

        if len(required_chains) == 1:
            # one chain for all times-> no problem
            basechain = chain_dict [tdb[0] ]
            return basechain.chain._at(t)
        else:
            # we have to assemble the positions and rates one by one, on our own
            positions, rates = self._generate_from_chain_dict(t, chain_dict)
            return positions, rates, None, None

