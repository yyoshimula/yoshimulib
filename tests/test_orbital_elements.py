"""OE -> state regressions, including circular and equatorial phase.

Run from the library root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yoshimulib.orbit.orbital_elements import oe2rv, MU_EARTH_KM


def reference_state(row, flag, mu=MU_EARTH_KM):
    """Independent scalar Kepler solve and explicit PQW -> ECI matrix."""
    a, e, inc, raan, argp, anomaly = row
    if flag == 0:
        mean = anomaly % (2*np.pi)
        E = brentq(lambda x: x-e*np.sin(x)-mean, 0., 2*np.pi, xtol=1e-14)
        f = np.arctan2(np.sqrt(1-e*e)*np.sin(E), np.cos(E)-e)
    else:
        f = anomaly
    p = a*(1-e*e)
    r = p/(1+e*np.cos(f))*np.array([np.cos(f), np.sin(f), 0.])
    v = np.sqrt(mu/p)*np.array([-np.sin(f), e+np.cos(f), 0.])
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array([
        [co*cw-so*sw*ci, -co*sw-so*cw*ci, so*si],
        [so*cw+co*sw*ci, -so*sw+co*cw*ci, -co*si],
        [sw*si, cw*si, ci],
    ])
    return rot@r, rot@v


class OrbitalElementsTests(unittest.TestCase):
    def assert_state_matches(self, row, flag):
        r, v = oe2rv(np.asarray(row), flag, MU_EARTH_KM)
        rr, vv = reference_state(row, flag)
        self.assertEqual(r.shape, (1, 3))
        self.assertEqual(v.shape, (1, 3))
        np.testing.assert_allclose(r[0], rr, atol=1e-7, rtol=0)
        np.testing.assert_allclose(v[0], vv, atol=1e-10, rtol=0)

    def test_circular_orbit_retains_argument_of_latitude(self):
        for flag in [0, 1]:
            self.assert_state_matches([7000, 0, .7, 0, np.pi/2, 0], flag)

    def test_equatorial_orbit_retains_periapsis_longitude(self):
        for flag in [0, 1]:
            self.assert_state_matches([10000, .1, 0, np.pi/2, .2, .3], flag)

    def test_retrograde_equatorial_orbit_retains_phase(self):
        for e in [0., .1]:
            for flag in [0, 1]:
                self.assert_state_matches([10000, e, np.pi, .8, 1.1, .3], flag)

    def test_states_near_singularities_match_continuous_reference(self):
        for e in [0., 1e-12, 1e-10, 2e-10, .1]:
            for inc in [0., 1e-12, 1e-10, 2e-10, .7, np.pi-1e-12, np.pi]:
                for flag in [0, 1]:
                    with self.subTest(e=e, inc=inc, flag=flag):
                        self.assert_state_matches([10000, e, inc, .8, 1.1, -.4], flag)

    def test_batch_both_anomaly_conventions_and_input_unchanged(self):
        rng = np.random.default_rng(907)
        rows = np.column_stack([rng.uniform(10000, 50000, 50), rng.uniform(.001, .8, 50),
                                rng.uniform(.01, 3.13, 50), rng.uniform(-10, 10, (50, 3))])
        rows = np.vstack([rows, [7000, 0, .7, .8, 1.1, .3],
                          [10000, .1, 0, .8, 1.1, .3], [10000, 0, np.pi, .8, 1.1, .3]])
        original = rows.copy()
        for flag in [0, 1]:
            r, v = oe2rv(rows, flag, MU_EARTH_KM)
            self.assertEqual(r.shape, (53, 3))
            for i, row in enumerate(rows):
                rr, vv = reference_state(row, flag)
                np.testing.assert_allclose(r[i], rr, atol=1e-7, rtol=0)
                np.testing.assert_allclose(v[i], vv, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(rows, original)

    def test_equivalent_singular_elements_give_same_state(self):
        for row, shifted in [
            ([10000, 0, .7, .8, 1.1, .3], [10000, 0, .7, .8, 1.5, -.1]),
            ([10000, .1, 0, .8, 1.1, .3], [10000, .1, 0, 1.2, .7, .3]),
            ([10000, .1, np.pi, .8, 1.1, .3], [10000, .1, np.pi, 1.2, 1.5, .3]),
        ]:
            for flag in [0, 1]:
                a = oe2rv(np.array(row), flag, MU_EARTH_KM)
                b = oe2rv(np.array(shifted), flag, MU_EARTH_KM)
                np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)

    def test_metre_and_kilometre_units(self):
        rows = np.array([[10000., 0, 0, .8, 1.1, .3], [10000., .1, np.pi, .8, 1.1, .3]])
        metres = rows.copy(); metres[:, 0] *= 1000
        for flag in [0, 1]:
            km_state = oe2rv(rows, flag, MU_EARTH_KM)
            m_state = oe2rv(metres, flag, MU_EARTH_KM*1e9)
            np.testing.assert_allclose(np.array(m_state)/1000, km_state, atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
