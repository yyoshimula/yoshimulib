"""Regression tests for the MATLAB yoshimuLibrary fixes ported on 2026-10-05.

Covers the MATLAB fixes c0ad127 (rv2oe), c2721b7 (leap seconds, dAT), 04fa227
(srpCT Gauss weight), 8661517 (hms2deg, calcAreaObj, geodeticIGRF, ...),
eb47a4c (srpAS, J2 sign), 8d1c71f (jr1971 Tc), the 2026-10-01 porting fixes
(sclerp, roe2DeputyOE, srpApproxCT2), the 2026-10-05 fixes made in both libraries
(jr1971 between 90 and 100 km, RAAN difference in oe2roe / roe2DeputyOE, diffuse
term of srpAS / srpASuni per facet) and the
functions ported at the same time (jr1971, loadSpaceWeather, lookupSolarGeoIndex,
disturbances, ukfInitPara, day2s, doy2gc). Reference numbers were generated
with the MATLAB library.

Run from the library root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yoshimulib.conversion import au2km, km2au, gc2jd, hms2deg, s2day, day2s
from yoshimulib.time_utils import doy2gc, leap_s, dat
from yoshimulib.orbit import oe2rv, rv2oe, orbit_const
from yoshimulib.relative_orbit import oe2roe, roe2deputy_oe
from yoshimulib.object import SatelliteModel, calc_area_obj, calc_local_frame
from yoshimulib.srp import srp_as, srp_as_uni, srp_ct, srp_ct_uni
from yoshimulib.hifi_srp import srp_approx_ct, srp_approx_ct2
from yoshimulib.dual_quaternions import sclerp
from yoshimulib.ukf_ckf import ukf_init_para, set_ukf_para
from yoshimulib.environment import (jr1971, load_space_weather, lookup_solar_geo_index,
                                    disturbances)
import yoshimulib.environment.magnetic as magnetic


class ConversionTimeTests(unittest.TestCase):
    def test_hms2deg_is_dms_to_deg(self):
        # the hour (degree) term must not be multiplied by 15
        self.assertEqual(hms2deg(23, 0, 0), 23.0)
        self.assertAlmostEqual(hms2deg(23, 26, 21.448), 23.4392911111111, places=12)
        self.assertAlmostEqual(hms2deg(0, 0, 46.8150), 46.8150 / 3600, places=15)

    def test_day2s_inverts_s2day(self):
        np.testing.assert_array_equal(day2s(np.array([1, 0.5, 2.25])), [86400, 43200, 194400])
        self.assertAlmostEqual(s2day(day2s(0.3)), 0.3, places=15)

    def test_doy2gc(self):
        month, day, hour, minute, second = doy2gc(
            np.array([2024, 2024, 2023, 2000]), np.array([1.0, 60.5, 365.75, 366.25]))
        np.testing.assert_array_equal(month, [1, 2, 12, 12])
        np.testing.assert_array_equal(day, [1, 29, 31, 31])
        np.testing.assert_array_equal(hour, [0, 12, 18, 6])
        np.testing.assert_array_equal(minute, [0, 0, 0, 0])
        np.testing.assert_allclose(second, 0, atol=1e-4)
        self.assertEqual(doy2gc(2024, 32.0)[:2], (2, 1))

    def test_leap_second_takes_effect_the_day_after_insertion(self):
        leap_jd = leap_s()
        # inserted at the end of 2016-12-31 -> TAI-UTC = 37 from 2017-01-01 00:00 UTC
        self.assertEqual(dat(gc2jd(2016, 12, 31, 23, 0, 0), leap_jd), 36.0)
        self.assertEqual(dat(gc2jd(2017, 1, 1, 0, 0, 0), leap_jd), 37.0)
        self.assertEqual(dat(gc2jd(2015, 6, 30, 12, 0, 0), leap_jd), 35.0)
        self.assertEqual(dat(gc2jd(2015, 7, 1, 0, 0, 0), leap_jd), 36.0)
        self.assertEqual(dat(gc2jd(1971, 12, 31, 0, 0, 0), leap_jd), 10.0)

    def test_dat_is_vectorized(self):
        leap_jd = leap_s()
        jd = np.array([gc2jd(1971, 12, 31), gc2jd(1972, 6, 30, 23, 59, 59), gc2jd(1972, 7, 1),
                       gc2jd(2016, 12, 31, 23), gc2jd(2017, 1, 1), gc2jd(2026, 1, 1)])
        expected = [10, 10, 11, 36, 37, 37]
        np.testing.assert_array_equal(dat(jd, leap_jd), expected)
        self.assertEqual(dat(jd.reshape(3, 2), leap_jd).shape, (3, 2))
        self.assertIsInstance(dat(float(jd[0]), leap_jd), float)
        for j, e in zip(jd, expected):
            self.assertEqual(dat(j, leap_jd), e)

    def test_au_conversions_accept_constants_like_matlab(self):
        const = orbit_const()
        self.assertEqual(au2km(1.0, const), const.AU)
        self.assertEqual(km2au(const.AU, const), 1.0)
        self.assertEqual(au2km(2.0, const.AU), 2 * const.AU)
        self.assertEqual(km2au(149597870.7), 1.0)


class Rv2oeTests(unittest.TestCase):
    def setUp(self):
        self.mu = orbit_const().GE

    def test_circular_equatorial_hand_built(self):
        # convention: w = 0, raan = 0, nu = true longitude (angle of r from +x)
        a = 7000.0
        vc = np.sqrt(self.mu / a)
        oe = rv2oe([a, 0, 0], [0, vc, 0], self.mu)
        np.testing.assert_allclose(oe[0], [a, 0, 0, 0, 0, 0], atol=1e-9)
        th = 2.0
        oe = rv2oe(a * np.array([np.cos(th), np.sin(th), 0]),
                   vc * np.array([-np.sin(th), np.cos(th), 0]), self.mu)[0]
        self.assertAlmostEqual(oe[0] / a, 1.0, places=12)
        np.testing.assert_allclose(oe[1:5], 0, atol=1e-12)
        self.assertAlmostEqual(oe[5], th, places=9)

    def test_circular_polar_hand_built(self):
        # r along +z, node line along +x: raan = 0, w = 0, nu = argument of latitude = pi/2
        a = 7000.0
        vc = np.sqrt(self.mu / a)
        oe = rv2oe([0, 0, a], [-vc, 0, 0], self.mu)[0]
        self.assertAlmostEqual(oe[0] / a, 1.0, places=12)
        np.testing.assert_allclose(oe[1:], [0, np.pi / 2, 0, 0, np.pi / 2], atol=1e-9)

    def test_vectorized_mixed_orbit_types_round_trip(self):
        oe_set = np.array([[8000, 0.1, 0.5, 1.0, 2.0, 3.0],                # elliptical inclined
                           [7000, 0, 0, 0, 0, 2.0],                        # circular equatorial
                           [7000, 0, np.deg2rad(51.6), 1.0, 0, 2.5],       # circular inclined
                           [8000, 0.1, 0, 0, 1.2, 0.7],                    # elliptical equatorial
                           [26600, 0.74, np.deg2rad(63.4), 4.0, 4.7, 5.5],
                           [7200, 0, np.deg2rad(98), 5.5, 0, 6.0]])
        r, v = oe2rv(oe_set, 1, self.mu)
        oe = rv2oe(r, v, self.mu)
        np.testing.assert_allclose(oe[:, 0], oe_set[:, 0], rtol=1e-9)
        np.testing.assert_allclose(oe[:, 1], oe_set[:, 1], atol=1e-12)
        np.testing.assert_allclose(oe[:, 2:], oe_set[:, 2:], atol=1e-9)

    def test_state_is_recovered_for_every_orbit_type(self):
        # rv2oe -> oe2rv must give the state back, including retrograde equatorial orbits
        rows = np.array([[8000, 0.1, 0.5, 1.0, 2.0, 3.0],
                         [7000, 0, 0, 0.3, 0.4, 2.0],
                         [7000, 0, 0.9, 1.0, 0.7, 2.5],
                         [8000, 0.1, 0, 0.6, 1.2, 0.7],
                         [7000, 0, np.pi, 0.2, 0.5, 1.3],
                         [9000, 0.2, np.pi, 0.9, 0.4, 2.2]])
        r, v = oe2rv(rows, 1, self.mu)
        oe = rv2oe(r, v, self.mu)
        self.assertTrue(np.all(np.isfinite(oe)))
        r2, v2 = oe2rv(oe, 1, self.mu)
        np.testing.assert_allclose(r2, r, atol=1e-6, rtol=0)
        np.testing.assert_allclose(v2, v, atol=1e-9, rtol=0)
        # circular: w = 0; equatorial: raan = 0
        np.testing.assert_array_equal(oe[[1, 2, 4], 4], 0)
        np.testing.assert_array_equal(oe[[1, 3, 4, 5], 3], 0)

    def test_matlab_reference(self):
        # Vallado example 2-5 and a Molniya-type orbit (values from MATLAB rv2oe)
        r = [[1813.362163017412, 5884.188717788758, -4940.075250447526],
             [6524.834, 6862.875, 6448.296]]
        v = [[-6.827344756456069, -5.052719592229377, -3.7228683904362363],
             [4.901327, 5.533756, -1.976341]]
        expected = [[26600.000000000164, 0.7400000000000014, 1.106538745764405, 4.0,
                     4.699999999999999, 5.500000000000001],
                    [36127.33776397483, 0.8328533990836886, 1.5336055626394494,
                     3.9775750028016947, 0.9317428111437859, 1.6115524999414732]]
        np.testing.assert_allclose(rv2oe(r, v, self.mu), expected, rtol=1e-9)


class RelativeOrbitTests(unittest.TestCase):
    chief = np.array([[7000, 1e-3, np.deg2rad(50), 0.7, 0.4, 1.1],
                      [7100, 0.01, 1.2, 3.0, 2.0, 5.0],
                      [6900, 0.02, 0.3, 5.0, 4.0, 2.0]])
    deputy = chief + np.array([[0.1, 1e-4, 1e-4, 2e-4, -3e-3, 2e-3],
                               [-0.2, -2e-4, 2e-4, -1e-4, 1e-3, -1e-3],
                               [0.05, 1e-4, -1e-4, 1e-4, 2e-3, 1e-3]])

    def test_multi_row_round_trip(self):
        for flag in [1, 0]:
            roe = oe2roe(self.chief, self.deputy, flag)
            dep = roe2deputy_oe(roe, self.chief, flag)
            self.assertEqual(dep.shape, self.chief.shape)
            for k in range(3):
                np.testing.assert_allclose(
                    dep[k], roe2deputy_oe(roe[k], self.chief[k], flag)[0], atol=1e-12)
            np.testing.assert_allclose(dep[:, 0], self.deputy[:, 0], atol=1e-8)
            np.testing.assert_allclose(dep[:, 1:], self.deputy[:, 1:], atol=1e-10)

    def test_matlab_reference(self):
        roe = oe2roe(self.chief, self.deputy, 0)
        np.testing.assert_allclose(
            roe[0], [1.4285714285766258e-05, -0.0008714424780627006, 9.338661875378615e-05,
                     3.590040989055667e-05, 9.999999999998899e-05, 0.00015320888862377873],
            rtol=1e-9, atol=1e-15)

    def test_equatorial_chief_row_does_not_affect_other_rows(self):
        roe = oe2roe(self.chief, self.deputy, 0)
        chief2 = self.chief.copy()
        chief2[1, 2] = 0
        dep = roe2deputy_oe(roe, chief2, 0)
        self.assertTrue(np.all(np.isfinite(dep)))
        self.assertEqual(dep[1, 3], chief2[1, 3])  # raan of the equatorial chief
        np.testing.assert_allclose(
            dep[[0, 2]], roe2deputy_oe(roe[[0, 2]], self.chief[[0, 2]], 0), atol=1e-12)

    def test_raan_across_zero(self):
        # chief RAAN just below 2*pi, deputy RAAN just above 0 (and vice versa)
        shift = np.array([0, 0, 0, 1.0, 0, 0])  # the same pair away from 0/2*pi
        for flag in [1, 0]:
            for s in [1, -1]:
                chief = np.array([7000, 1e-3, 0.9, 2 * np.pi - s * 1e-4, 0.4, 1.1])
                deputy = chief + np.array([0.1, 1e-4, 1e-4, s * 3e-4, -3e-3, 2e-3])
                deputy[3] = np.mod(deputy[3], 2 * np.pi)
                roe = oe2roe(chief, deputy, flag)
                np.testing.assert_allclose(roe, oe2roe(chief - shift, deputy - shift, flag),
                                           atol=1e-12)
                self.assertAlmostEqual(roe[0, 5], s * 3e-4 * np.sin(0.9), places=12)
                d = roe2deputy_oe(roe, chief, flag)[0] - deputy
                d[3:] = np.mod(d[3:] + np.pi, 2 * np.pi) - np.pi
                self.assertLess(abs(d[0]), 1e-8)
                self.assertLess(np.max(np.abs(d[1:])), 1e-10)

    def test_angles_are_wrapped_and_inputs_unchanged(self):
        chief = self.chief.copy()
        deputy = self.deputy.copy()
        chief[0, 3:6] += [2 * np.pi, -2 * np.pi, 4 * np.pi]
        chief0, deputy0 = chief.copy(), deputy.copy()
        roe = oe2roe(chief, deputy, 0)
        np.testing.assert_array_equal(chief, chief0)
        np.testing.assert_array_equal(deputy, deputy0)
        np.testing.assert_allclose(roe, oe2roe(self.chief, self.deputy, 0), atol=1e-12)
        dep = roe2deputy_oe(roe, chief, 0)
        self.assertTrue(np.all((dep[:, 3:] >= 0) & (dep[:, 3:] < 2 * np.pi)))


class GeometryTests(unittest.TestCase):
    def test_quad_area_is_the_sum_of_two_triangles(self):
        sat = SatelliteModel()
        sat.vertices = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
                                 [3, 0, 0], [2.5, 2, 0], [0, 0, 1], [1, 0, 1.5]], dtype=float)
        # rectangle, trapezoid, triangle, general quad, non-planar quad (0-based, -1: none)
        sat.faces = np.array([[0, 1, 2, 3], [0, 1, 5, 3], [1, 4, 2, -1], [0, 4, 5, 3], [6, 7, 2, 3]])
        area, pos = calc_area_obj(sat)
        np.testing.assert_allclose(area, [2.0, 3.25, 0.5, 4.25, 2.559857486112055], rtol=1e-14)
        np.testing.assert_allclose(pos[0], [1.0, 0.5, 0.0])
        np.testing.assert_allclose(pos[2], [7 / 3, 1 / 3, 0.0])

    def test_three_column_faces(self):
        sat = SatelliteModel()
        sat.vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
        sat.faces = np.array([[0, 1, 2]])
        np.testing.assert_allclose(calc_area_obj(sat)[0], [1.0])


def _single_facet(nu, nv, m_ct):
    sat = SatelliteModel()
    sat.normal = np.array([[0.0, 0.0, 1.0]])
    sat.area = np.array([1.0])
    sat.pos = np.zeros((1, 3))
    sat.Cd = np.array([0.3])
    sat.F0 = np.array([0.5])
    sat.nu = np.array([float(nu)])
    sat.nv = np.array([float(nv)])
    sat.mCT = np.array([m_ct])
    sat.uu = np.array([[1.0, 0.0, 0.0]])
    sat.uv = np.array([[0.0, 1.0, 0.0]])
    sat.qlb = np.array([[0.0, 0.0, 0.0, 1.0]])
    return sat


class SrpSamplingTests(unittest.TestCase):
    """Importance sampling must agree with the uniform-sampling references."""

    def setUp(self):
        self.const = orbit_const()
        self.d = self.const.AU * 1e3  # m
        th, ph = np.deg2rad(40), np.deg2rad(30)
        self.sun = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    def assert_close(self, a, b, tol):
        self.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(b), tol)

    def test_srp_as_anisotropic(self):
        # nu != nv: the old quadrant shift (+pi/2) and the extra 1/(n.h) gave errors > 10 %
        for nu, nv in [(8, 40), (40, 8)]:
            np.random.seed(1)
            _, _, imp = srp_as(_single_facet(nu, nv, 0.2), self.sun, self.d, self.const, n_mc=400_000)
            np.random.seed(2)
            _, _, uni = srp_as_uni(_single_facet(nu, nv, 0.2), self.sun, self.d, self.const,
                                   n_mc=1_000_000)
            self.assert_close(imp, uni, 0.02)

    def test_srp_ct_beckmann_and_gauss(self):
        for ndf in ['Beckmann', 'Gauss']:
            np.random.seed(3)
            _, _, imp = srp_ct(_single_facet(8, 40, 0.3), self.sun, self.d, self.const,
                               ndf=ndf, n_mc=400_000)
            np.random.seed(4)
            _, _, uni = srp_ct_uni(_single_facet(8, 40, 0.3), self.sun, self.d, self.const,
                                   ndf=ndf, n_mc=1_000_000)
            self.assert_close(imp, uni, 0.02)


def _three_facets(idx=slice(None)):
    """Three triangular facets with different normals and optical properties"""
    normal = np.array([[0, 0, 1.0], [0.6, 0.0, 0.8], [0.0, -0.8, 0.6]])
    u = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])  # in-plane edge directions
    vertices = []
    for i in range(3):
        p0 = np.array([i + 1.0, 0, 0])
        vertices += [p0, p0 + u[i], p0 + np.cross(normal[i], u[i])]
    sat = SatelliteModel()
    sat.vertices = np.array(vertices)
    sat.faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])[idx]
    sat.normal = normal[idx]
    sat.area = np.array([1.0, 0.5, 2.0])[idx]
    sat.pos = np.array([[0, 0, 0], [1, 0, 0.3], [0, 1, -0.2]])[idx]
    sat.Cd = np.array([0.3, 0.5, 0.1])[idx]
    sat.F0 = np.array([0.5, 0.2, 0.7])[idx]
    sat.nu = np.array([8.0, 40.0, 20.0])[idx]
    sat.nv = np.array([40.0, 8.0, 20.0])[idx]
    sat.uu, sat.uv, sat.qlb = calc_local_frame(sat)
    return sat


class AshikhminShirleyFacetTests(unittest.TestCase):
    """Facets that are not normal to the body z-axis"""

    def setUp(self):
        self.const = orbit_const()
        self.d = au2km(1.0, self.const) * 1e3  # m
        self.sun = np.array([0.3, -0.2, 0.93]) / np.linalg.norm([0.3, -0.2, 0.93])

    def test_diffuse_term_per_facet_along_its_normal(self):
        # int (1 - (1 - n.v/2)^5) (n.v) v dw = 1573/2688 * pi * n over the hemisphere
        sat = _three_facets()
        NS = sat.normal @ self.sun
        cd1 = 28 / 23 * sat.Cd / np.pi * (1 - sat.F0) * (1 - (1 - NS / 2) ** 5)
        expected = np.sum((-self.const.S0 / self.const.c * sat.area * NS * cd1
                           * 1573 / 2688 * np.pi)[:, np.newaxis] * sat.normal, axis=0)
        _, cd_as, _ = srp_as(_three_facets(), self.sun, self.d, self.const, n_mc=100)
        _, cd_uni, _ = srp_as_uni(_three_facets(), self.sun, self.d, self.const, n_mc=100)
        np.testing.assert_allclose(cd_as, expected, rtol=1e-12)
        np.testing.assert_allclose(cd_uni, expected, rtol=1e-12)
        # MATLAB srpAS / srpASuni
        np.testing.assert_allclose(
            cd_as, [-3.4248934e-07, 9.9449160e-08, -9.6268770e-07], rtol=1e-6)
        # the same as the sum of the facets taken one by one
        total = np.zeros(3)
        for i in range(3):
            total += srp_as(_three_facets([i]), self.sun, self.d, self.const, n_mc=100)[1]
        np.testing.assert_allclose(cd_as, total, rtol=1e-12)

    def test_importance_sampling_matches_uniform_reference(self):
        np.random.seed(1)
        sat_as, _, _ = srp_as(_three_facets(), self.sun, self.d, self.const, n_mc=400_000)
        np.random.seed(2)
        sat_uni, _, _ = srp_as_uni(_three_facets(), self.sun, self.d, self.const, n_mc=1_000_000)
        err = (np.linalg.norm(sat_as.force - sat_uni.force, axis=1)
               / np.linalg.norm(sat_uni.force, axis=1))
        self.assertLess(np.max(err), 0.02)  # Monte-Carlo noise is about 0.1 %


class _Facets:
    pass


class HifiSrpTests(unittest.TestCase):
    def test_srp_approx_ct2_equals_sum_of_facets_and_matlab(self):
        const = orbit_const()
        d = au2km(1.0, const) * 1e3  # m
        theta_n = np.deg2rad([0, 10, 30, 50, 75])
        sat = _Facets()
        sat.area = np.array([1, 0.5, 2, 1, 0.3])
        sat.F0 = np.array([0.5, 0.4, 0.6, 0.5, 0.3])
        sat.mCT = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        srp2 = srp_approx_ct2(sat, theta_n, [0, 0, 1], d, const)
        self.assertTrue(np.all(np.isfinite(srp2)))
        total = np.zeros(3)
        for i in range(len(theta_n)):
            s = _Facets()
            s.area, s.F0, s.mCT = sat.area[i], sat.F0[i], sat.mCT[i]
            total = total + srp_approx_ct(s, theta_n[i], [0, 0, 1], d, const)
        np.testing.assert_allclose(srp2, total, rtol=1e-12, atol=1e-18)
        # MATLAB srpApproxCT2 (the integration grid must have exactly n + 1 bounds)
        np.testing.assert_allclose(
            srp2, [-2.6362956225508657e-19, -1.095772846974599e-06, -7.032159537268817e-07],
            rtol=1e-10, atol=1e-17)


class DualQuaternionTests(unittest.TestCase):
    def test_sclerp_row_inputs(self):
        dq1 = np.array([0.10259783520851541, -0.20519567041703082, 0.3077935056255462,
                        0.9233805168766387, 0.20519567041703085, -1.051627810887283,
                        0.23084512921915967, -0.33344296442767507])
        dq2 = np.array([0.4338609156373123, 0.10846522890932808, -0.21693045781865616,
                        0.8677218312746247, 1.138884903547945, 0.9761870601839527,
                        0.9219544457292888, -0.4609772228646444])
        t = np.arange(5) / 4
        for a, b in [(dq1, dq2), (dq1.reshape(1, 8), dq2.reshape(1, 8))]:
            dqt = sclerp(t, 4, a, b)
            self.assertEqual(dqt.shape, (5, 8))
            np.testing.assert_allclose(dqt[0], dq1, atol=1e-12)
            np.testing.assert_allclose(dqt[-1], dq2, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(dqt[:, :4], axis=1), 1, atol=1e-12)
            # MATLAB sclerp at t = 0.5
            np.testing.assert_allclose(
                dqt[2], [0.2861999403615959, -0.05160554571438339, 0.04847529995997457,
                         0.955550421004737, 0.7651091280565929, -0.048910509085145604,
                         0.623154356898466, -0.26341449825303076], atol=1e-12)


class UkfTests(unittest.TestCase):
    def test_ukf_init_para_defaults(self):
        ukf = ukf_init_para(3)
        self.assertEqual(ukf['n'], 3)
        self.assertEqual((ukf['alp'], ukf['bet'], ukf['kappa']), (1e-4, 2, 0))
        self.assertAlmostEqual(ukf['lam'], 1e-8 * 3 - 3, places=15)
        self.assertEqual(ukf['wm'].shape, (7,))
        self.assertAlmostEqual(np.sum(ukf['wm']), 1.0, places=6)
        np.testing.assert_array_equal(ukf['wc'][1:], ukf['wm'][1:])

    def test_given_parameters_are_kept(self):
        given = {'alp': 0.5, 'kappa': 1, 'wc': [[1.0], [2.0]]}
        ukf = ukf_init_para(6, given)
        self.assertEqual((ukf['n'], ukf['alp'], ukf['bet'], ukf['kappa']), (6, 0.5, 2, 1))
        self.assertEqual(ukf['lam'], 0.25 * 7 - 6)
        np.testing.assert_array_equal(ukf['wc'], [1.0, 2.0])
        self.assertEqual(set(given), {'alp', 'kappa', 'wc'})  # the input dict is not modified

    def test_set_ukf_para_is_compatible(self):
        old = set_ukf_para(3, alp=0.5)
        new = ukf_init_para(3, {'alp': 0.5})
        self.assertEqual(set(old), set(new))
        np.testing.assert_array_equal(old['wm'], [-3.0] + [2 / 3] * 6)
        np.testing.assert_array_equal(old['wc'], new['wc'])


class GeodeticIgrfTests(unittest.TestCase):
    def test_decimal_year_is_the_fraction_of_the_actual_year(self):
        years = []
        original = magnetic.igrf12
        magnetic.igrf12 = lambda year, alt, lat, lon, *a, **k: years.append(year) or np.zeros(3)
        try:
            for jd in [gc2jd(2015, 1, 1), gc2jd(2016, 7, 2, 12), gc2jd(2019, 12, 31, 23, 59, 59),
                       gc2jd(2020, 2, 29, 6)]:
                magnetic.geodetic_igrf(jd, 0.1, 0.2, 400.0)
        finally:
            magnetic.igrf12 = original
        # MATLAB geodeticIGRF (2016 and 2020 are leap years)
        np.testing.assert_allclose(
            years, [2015.0, 2016.5013661202186, 2019.9999999682905, 2020.1618852459017], rtol=1e-14)


class Jr1971Tests(unittest.TestCase):
    keys = ['total_density', 'temperature', 'exospheric_temperature', 'N2_number_density',
            'O2_number_density', 'O_number_density', 'Ar_number_density', 'He_number_density',
            'H_number_density']
    jd0 = 2457754.5  # 2017-01-01 00:00:00

    # [jd - jd0, lat (deg), lon (deg), h (km), F10, F10a, Kp] -> MATLAB jr1971
    cases = [
        ([0.0, 20.0, 60.0, 110.0, 150.0, 100.0, 4.0],
         [1.1354561266927785e-07, 236.28280316396655, 825.7255165059072, 1.8303739530411786e+18,
          3.054283652668935e+17, 4.294646896817409e+17, 1.146324551669485e+16,
          57199339213450.79, 0.0]),
        ([0.0, 20.0, 60.0, 300.0, 150.0, 100.0, 4.0],
         [1.581157429718661e-11, 868.8430149272061, 882.2714980062386, 42999861210366.43,
          1947666276889.03, 514040168753049.0, 5422895287.763345, 7583817616106.261, 0.0]),
        ([0.0, 20.0, 60.0, 800.0, 150.0, 100.0, 4.0],
         [6.336877239291683e-15, 882.2703914859659, 882.2714980062386, 4912308.9671473745,
          23142.681610386295, 53929867247.89096, 0.705697524609739, 727817942969.0455,
          39632153569.02293]),
        ([100.3, -45.0, 200.0, 400.0, 70.0, 75.0, 1.0],
         [8.105558354842828e-13, 701.2398457099465, 701.8109925079514, 198246960268.5913,
          3912542670.894773, 29009798234175.855, 2077202.691772964, 4572566510448.909, 0.0]),
        ([200.7, 80.0, -30.0, 350.0, 220.0, 180.0, 7.3],
         [2.0668828583272344e-11, 1448.1583763995852, 1485.506449593526, 94720466371179.08,
          5294023730209.095, 600797104804730.5, 22422539334.9884, 2680240139026.885, 0.0]),
    ]

    def test_matlab_reference(self):
        for c, expected in self.cases:
            with self.subTest(case=c):
                out = jr1971(self.jd0 + c[0], np.deg2rad(c[1]), np.deg2rad(c[2]), c[3] * 1e3,
                             c[4], c[5], c[6])
                np.testing.assert_allclose([out[k] for k in self.keys], expected, rtol=1e-10)

    def test_tc_uses_the_81_day_average(self):
        # Tc = 379 + 3.24 F10a + 1.3 (F10 - F10a): a change of the daily F10 alone enters
        # the exospheric temperature with the factor 1.3 (not 4.54), independent of F10a
        args = (self.jd0, np.deg2rad(20), np.deg2rad(60), 300e3)
        d_tinf = 0.03  # geomagnetic term for Kp = 0 above 200 km
        for f10a in [100, 160]:
            t0 = jr1971(*args, 100, f10a, 0)['exospheric_temperature']
            t1 = jr1971(*args, 150, f10a, 0)['exospheric_temperature']
            base = jr1971(*args, f10a, f10a, 0)['exospheric_temperature']
            diurnal = (base - d_tinf) / (379 + 3.24 * f10a)
            self.assertAlmostEqual((t1 - t0) / diurnal, 1.3 * 50, places=8)

    def test_barometric_equation_between_90_and_100_km(self):
        # the exponent needs the factor f: scale height of about 5.6 km and a
        # continuous junction with the 100-125 km branch
        for lat, lon, f10, f10a, kp in [(20, 60, 150, 100, 4), (-45, 200, 70, 75, 1),
                                        (80, -30, 220, 180, 7.3)]:
            rho = lambda h_km: jr1971(self.jd0, np.deg2rad(lat), np.deg2rad(lon), h_km * 1e3,
                                      f10, f10a, kp)['total_density']
            scale_height = -10 / np.log(rho(100) / rho(90))  # km
            self.assertTrue(5 < scale_height < 6.5)
            self.assertAlmostEqual(rho(100) / rho(100 + 1e-6), 1.0, places=3)

    def test_hydrostatic_integration_from_90_km(self):
        import math
        from scipy.integrate import quad
        import yoshimulib.environment.jacchia_roberts as jr

        jd, phi, lam = self.jd0 + 77.3, np.deg2rad(20), np.deg2rad(60)
        f10, f10a, kp = 150, 100, 4
        t_inf = jr1971(jd, phi, lam, 95e3, f10, f10a, kp)['exospheric_temperature']
        t_x = 371.6678 + 0.0518806 * t_inf - 294.3505 * math.exp(-0.00216222 * t_inf)
        temperature = lambda z: jr._jr1971_temperature(z, t_x, t_inf)
        mass = lambda z: jr._evalpoly_asc(z, jr._AA)
        integrand = lambda z: mass(z) * jr._G0 * (jr._RA / (jr._RA + z)) ** 2 / (jr._RSTAR * temperature(z))

        def correction(h):  # geomagnetic, semi-annual and latitudinal corrections
            Phi = (jd - 2436204.5) / 365.2422
            tau = Phi + 0.09544 * ((0.5 * (1 + math.sin(2 * math.pi * Phi + 6.035))) ** 1.65 - 0.5)
            g = 0.012 * kp + 1.2e-5 * math.exp(kp)
            sa = ((5.876e-7 * h ** 2.331 + 0.06328) * math.exp(-0.002868 * h)
                  * (0.02835 + (0.3817 + 0.17829 * math.sin(2 * math.pi * tau + 4.137))
                     * math.sin(4 * math.pi * tau + 4.259)))
            lt = (0.014 * (h - 90) * math.exp(-0.0013 * (h - 90) ** 2)
                  * math.sin(2 * math.pi * Phi + 1.72) * math.sin(phi) * abs(math.sin(phi)))
            return 10 ** (g + lt + sa)

        for z in [90.5, 92, 95, 98, 100]:
            integral = quad(integrand, 90, z, epsabs=1e-13, epsrel=1e-13)[0]
            expected = (jr._RHO1 * 1000 * correction(z) * mass(z) * jr._T1
                        / (jr._M1 * temperature(z)) * math.exp(-integral))
            # the closed form uses rounded coefficients (alpha, beta): 1e-5 level
            self.assertAlmostEqual(jr1971(jd, phi, lam, z * 1e3, f10, f10a, kp)['total_density']
                                   / expected, 1.0, places=4)

    def test_below_90_km_is_rejected(self):
        with self.assertRaises(ValueError):
            jr1971(self.jd0, 0.3, 1.0, 89.9e3, 150, 100, 4)


class SpaceWeatherTests(unittest.TestCase):
    header = ('DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,'
              'AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,'
              'F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81')
    rows = [
        '2024-01-01,2596,1,7,10,13,17,20,23,27,30,147,3,4,5,6,7,9,12,15,8,0.3,1,54,135.9,131.4,OBS,160.1,152.8,155.9,150.6',
        '2024-01-02,2596,2,33,37,40,43,47,50,53,57,360,18,22,27,32,39,48,56,67,39,1.2,5,72,142.1,137.4,OBS,160.5,152.6,156.2,150.3',
        '2024-01-03,2596,3,3,0,3,7,10,13,17,20,73,2,0,2,3,4,5,6,7,4,0.1,0,104,140.2,135.6,OBS,161.0,152.5,156.6,150.1',
        '2024-01-04,2596,4,,,,,,,,,,,,,,,,,,,,,90,125.7,121.6,PRD,,152.3,,149.9',
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / 'SW-All.csv')
        Path(self.path).write_text('\n'.join([self.header] + self.rows) + '\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_space_weather(self):
        sw = load_space_weather(self.path)
        self.assertEqual(sw['nDays'], 4)
        self.assertEqual(sw['jdNoonStart'], gc2jd(2024, 1, 1, 12, 0, 0))
        self.assertEqual(sw['jdNoonEnd'], gc2jd(2024, 1, 4, 12, 0, 0))
        np.testing.assert_allclose(sw['Kp'][0], [0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0])
        np.testing.assert_array_equal(sw['Ap'][1], [18, 22, 27, 32, 39, 48, 56, 67])
        np.testing.assert_array_equal(sw['ApAvg'][:3], [8, 39, 4])
        np.testing.assert_array_equal(sw['F107obs'], [135.9, 142.1, 140.2, 125.7])
        self.assertTrue(np.isnan(sw['F107obsCenter81'][3]) and np.all(np.isnan(sw['Kp'][3])))
        with self.assertRaises(FileNotFoundError):
            load_space_weather(self.path + '.missing')

    def test_lookup_uses_one_day_lag_for_f10_and_no_lag_for_kp(self):
        sw = load_space_weather(self.path)
        # 2024-01-02 04:00 UTC: F10/F10a of 01-01, Kp of 01-02 block 03-06 h
        self.assertEqual(lookup_solar_geo_index(gc2jd(2024, 1, 2, 4, 0, 0), sw), (135.9, 160.1, 3.7))
        # 2024-01-03 23:59 UTC: F10/F10a of 01-02, Kp of 01-03 block 21-24 h
        self.assertEqual(lookup_solar_geo_index(gc2jd(2024, 1, 3, 23, 59, 0), sw), (142.1, 160.5, 2.0))
        # the path is accepted as well (read once and cached)
        self.assertEqual(lookup_solar_geo_index(gc2jd(2024, 1, 2, 0, 0, 0), self.path),
                         (135.9, 160.1, 3.3))
        self.assertEqual(lookup_solar_geo_index(gc2jd(2024, 1, 2, 12, 0, 0), Path(self.path)),
                         (135.9, 160.1, 4.7))

    def test_lookup_errors(self):
        sw = load_space_weather(self.path)
        with self.assertRaises(ValueError):  # the lag date 2023-12-31 is not in the file
            lookup_solar_geo_index(gc2jd(2024, 1, 1, 12, 0, 0), sw)
        with self.assertRaises(ValueError):  # after the last row
            lookup_solar_geo_index(gc2jd(2024, 1, 5, 12, 0, 0), sw)
        sw['F107obsCenter81'][2] = np.nan
        with self.assertRaises(ValueError):  # F10a missing at the lag date
            lookup_solar_geo_index(gc2jd(2024, 1, 4, 12, 0, 0), sw)
        with self.assertRaises(TypeError):
            lookup_solar_geo_index(gc2jd(2024, 1, 2, 12, 0, 0), 3.0)

    def test_non_consecutive_rows_warn(self):
        Path(self.path).write_text('\n'.join([self.header, self.rows[0], self.rows[2]]) + '\n')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            load_space_weather(self.path)
        self.assertEqual(len(caught), 1)


class DisturbancesTests(unittest.TestCase):
    def test_accelerations(self):
        const = orbit_const()
        out = disturbances(plot=False)
        self.assertEqual(out['r'].shape, (4301,))
        for key in ['a_earth', 'a_j2', 'a_sun', 'a_moon']:
            self.assertEqual(out[key].shape, (4301, 3))
            self.assertTrue(np.all(np.isfinite(out[key])))
        r = out['r']
        np.testing.assert_allclose(out['a_earth'][:, 0], -const.GE / r ** 2, rtol=1e-14)
        # J2 on the equator: inward, 3/2 J2 mu RE^2 / r^4
        np.testing.assert_allclose(out['a_j2'][:, 0], -1.5 * const.J2 * const.GE * const.RE ** 2 / r ** 4,
                                   rtol=1e-14)
        np.testing.assert_array_equal(out['a_j2'][:, 1:], 0)
        # MATLAB disturbances.m at h = 0 and 43000 km (third-body accelerations, km/s^2)
        np.testing.assert_allclose(out['a_sun'][[0, -1], 0], [5.056930817627285e-10, 3.916653558152174e-09],
                                   rtol=1e-8)
        np.testing.assert_allclose(out['a_moon'][[0, -1], 0], [1.1291011683892943e-09, 1.050145658833566e-08],
                                   rtol=1e-8)

    def test_custom_altitudes(self):
        const = orbit_const()
        out = disturbances(const, h=[400.0, 35786.0], plot=False)
        np.testing.assert_array_equal(out['r'], [const.RE + 400.0, const.RE + 35786.0])


if __name__ == '__main__':
    unittest.main()
