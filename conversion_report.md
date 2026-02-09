# MATLAB -> Python Conversion Coverage Report

MATLAB functions found: 214

MATLAB scripts (no function signature found): 28

Matched (same package): 208

Matched (different package): 6

Unmatched: 0

## Unmatched MATLAB Functions

- None

## MATLAB Functions Matched In Other Packages
- associatedLegendre (math/associatedLegendre.m) -> associated_legendre in math_utils [exact]
- double_factorial (math/associatedLegendre.m) -> _double_factorial in math_utils [normalized]
- legendreRecursive (math/associatedLegendre.m) -> _legendre_recursive in math_utils [normalized]
- normRow (math/normRow.m) -> norm_row in math_utils [exact]
- skew (math/skew.m) -> skew in math_utils [exact]
- wrapPi (math/wrapPi.m) -> wrap_pi in math_utils [exact]

## Normalized Matches (Same Package)
- axiQsol (geometricIntegration/axiQsol.m) -> axi_q_sol
- ecef2LatLonH (orbit/ecef2LatLonH.m) -> ecef2lat_lon_h
- roe2DeputyOE (relativeOrbit/roe2DeputyOE.m) -> roe2deputy_oe
- srpASuni (srp/srpASuni.m) -> srp_as_uni
- srpCTinterp (srp/srpCTinterp.m) -> srp_ct_interp
- srpCTuni (srp/srpCTuni.m) -> srp_ct_uni
- setUKFpara (ukfCkf/setUKFpara.m) -> set_ukf_para
- adjustFont (utility/fig4Paper.m) -> _adjust_font
- decideFigSize (utility/fig4Paper.m) -> _decide_fig_size
- detectLayout (utility/fig4Paper.m) -> _detect_layout
- optimizeFig (utility/fig4Paper.m) -> _optimize_fig

## Python Top-Level Functions Without MATLAB Name Match
- generate_euler_angle_kinematics (attitude)
- _day_of_year (environment)
- exponential_atmosphere (environment)
- nrlmsise00_density (environment)
- _find_file (orbit)
- _get (orbit)
- _get_egm (orbit)
- _read_txt_matrix (orbit)
- get_eop (orbit)
- mee_derivatives (orbit)
- read_tle_single (orbit)
- _calc_f_double_r (orbit_determination)
- furnsh (spice)
- kclear (spice)
- ktotal (spice)
- list_loaded_kernels (spice)
- load_spice_kernel (spice)
- make_coeff_table (srp)

## Python Warning Notes (TODO/FIXME/approx/simplified)
- relative_orbit/roe.py:216 roe2los_approx
- relative_orbit/roe.py:332 # Calculate approximated line of sight (LOS) from relative orbital elements (ROE)
- geometric_integration/gi.py:300 Assumes axisymmetric body (MOI(1,1) = MOI(2,2))
- orbit/precession.py:232 # Nutation in longitude and obliquity (IAU 1980 model - simplified)
- orbit/tle.py:144 # Approximate conversion
- orbit/tle.py:147 # Default: UT1 to TT approximation
- orbit/tle.py:162 # Eccentricity (assumed decimal point)
- attitude/euler.py:520 This is a simplified version. The original MATLAB code uses symbolic
- attitude/euler.py:536 # Python: This is a simplified implementation for the ZYX case
- hifi_srp/approx_ct.py:2 High-fidelity SRP approximation using Cook-Torrance model
- hifi_srp/approx_ct.py:35 Analytic Approximation of High-Fidelity Solar Radiation Pressure.
- hifi_srp/approx_ct.py:43 srp_approx_ct
- hifi_srp/approx_ct.py:97 Analytic Approximation of High-Fidelity Solar Radiation Pressure.
- hifi_srp/approx_ct.py:105 srp_approx_ct2
- hifi_srp/approx_ct.py:190 # Calculate 1st-order approximation coefficients
- hifi_srp/approx_ct.py:258 # Analytic solution for 1st order approximation
- hifi_srp/approx_ct.py:373 def srp_approx_ct(sat: SatelliteModel, theta_n: float, sun_b: np.ndarray,
- hifi_srp/approx_ct.py:376 # Approximating specular term of SRP with Cook-Torrance model
- hifi_srp/approx_ct.py:398 approximated SRP with Cook-Torrance model expressed with Sun-fixed frame
- hifi_srp/approx_ct.py:402 Cook-Torrance model analytical SRP approximation
- hifi_srp/approx_ct.py:406 Analytic Approximation of High-Fidelity Solar Radiation Pressure.
- hifi_srp/approx_ct.py:414 srp_approx_ct2
- hifi_srp/approx_ct.py:492 def srp_approx_ct2(sat: SatelliteModel, theta_n: np.ndarray, sun_b: np.ndarray,
- hifi_srp/approx_ct.py:495 # Approximating specular term of SRP with Cook-Torrance model (multiple facets)
- hifi_srp/approx_ct.py:517 approximated SRP with Cook-Torrance model expressed with Sun-fixed frame, 1x3
- hifi_srp/approx_ct.py:525 Analytic Approximation of High-Fidelity Solar Radiation Pressure.
- hifi_srp/approx_ct.py:533 srp_approx_ct
- hifi_srp/approx_ct.py:566 sunlit_flag = 1  # Assuming all facets are sunlit
- hifi_srp/__init__.py:8 from .approx_ct import (
- hifi_srp/__init__.py:14 srp_approx_ct,
- hifi_srp/__init__.py:15 srp_approx_ct2,
- hifi_srp/__init__.py:21 'srp_approx_ct', 'srp_approx_ct2',
- sun_moon/ephemeris.py:551 # ELP2000 coefficients (simplified)
- srp/srp_models.py:800 # Separation (Approximate)
- srp/srp_models.py:838 Analytic Approximation of High-Fidelity Solar Radiation Pressure.
- srp/srp_models.py:995 # Exponential integral approximation
- utility/plotting.py:547 # This is a simplified version - getting actual monitor info
