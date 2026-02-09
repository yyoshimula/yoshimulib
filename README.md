# yoshimulib

宇宙機の力学・姿勢制御・軌道力学・宇宙環境モデリングのためのライブラリです。
MATLABライブラリyoshimuLibraryのPython版です。

## インストール

```bash
pip install -r requirements.txt
```

SPICE機能を使用する場合:

```bash
pip install spiceypy
```

## クイックスタート

```python
import yoshimulib as yl

# クォータニオン演算
q = yl.attitude.zyx2q(0.1, 0.2, 0.3)          # オイラー角 → クォータニオン
dcm = yl.attitude.q2dcm(q)                      # クォータニオン → DCM
q_err = yl.attitude.q_err(q, [1, 0, 0, 0])      # クォータニオン誤差

# 軌道力学
r, v = yl.orbit.oe2rv(7000, 0.01, 0.5, 0, 0, 0) # 軌道要素 → 位置・速度
oe = yl.orbit.rv2oe(r, v)                         # 位置・速度 → 軌道要素
E = yl.orbit.kepler_eq(M=1.0, e=0.01)             # ケプラー方程式

# 時刻変換
jd = yl.time_utils.cal2jd(2024, 1, 1, 0, 0, 0)   # 暦 → ユリウス日
mjd = yl.time_utils.jd2mjd(jd)                     # JD → 修正ユリウス日
```

## モジュール一覧

### 低レベルユーティリティ

| モジュール   | 説明                                           |
| ------------ | ---------------------------------------------- |
| `math_utils` | ルジャンドル多項式、歪対称行列、角度ラッピング |
| `conversion` | 単位変換 (AU/km, ラジアン/秒角)                |
| `time_utils` | 時刻系変換 (JD, MJD, UTC, TT, うるう秒)        |

### 中レベル物理モデル

| モジュール    | 説明                                                         |
| ------------- | ------------------------------------------------------------ |
| `orbit`       | 軌道力学、ケプラー方程式、EGM2008、座標変換 (GCRF/ITRF/TEME) |
| `environment` | IGRF-12磁場モデル、Jaccia-Bowman 2008大気モデル、NRLMSISE-00 |
| `sun_moon`    | 太陽・月の暦 (ELP2000-82)                                    |

### 高レベルアプリケーション

| モジュール    | 説明                                                            |
| ------------- | --------------------------------------------------------------- |
| `attitude`    | DCM、クォータニオン、オイラー角、SLERP、TRIAD、キネマティクス   |
| `srp`         | 太陽輻射圧 (Simple, Ashikhmin-Shirley, Cook-Torrance)           |
| `hifi_srp`    | 高精度解析的SRP近似                                             |
| `lightcurves` | BRDFモデルによる合成ライトカーブ生成                            |
| `ukf_ckf`     | 無香料カルマンフィルタ / キュバチャカルマンフィルタ / 平方根CKF |
| `object`      | 3D宇宙機モデルI/O (OBJ/MTL)、自己遮蔽、レイ交差判定             |
| `utility`     | 可視化・プロットユーティリティ                                  |

### 専門モジュール

| モジュール              | 説明                                        |
| ----------------------- | ------------------------------------------- |
| `dual_quaternions`      | 双対クォータニオン演算、ScLERP補間          |
| `geometric_integration` | SO(3)構造を保存するリー群積分器             |
| `relative_orbit`        | 相対軌道要素、HCW方程式                     |
| `orbit_determination`   | 初期軌道決定 (Gibbs法, Gauss法, Double-R法) |
| `gpr`                   | ガウス過程回帰                              |
| `spherical_gaussian`    | BRDFモデリング用球面ガウス関数              |
| `spice`                 | SPICEカーネル読み込み (SpiceyPy経由)        |

## クォータニオン規約

`scalar`パラメータにより2つの規約をサポート:

- `scalar=0`: **q** = [q0, q1, q2, q3]^T = [cos(theta/2), e*sin(theta/2)]^T
- `scalar=4`: **q** = [q1, q2, q3, q4]^T = [e*sin(theta/2), cos(theta/2)]^T

## 座標系

- **GCRF** - 地心天体基準座標系
- **ITRF/ECEF** - 地球固定座標系
- **J2000** - 慣性座標系
- **TEME** - 真赤道平均春分点座標系
- **RTN** - 動径・接線・法線座標系 (相対軌道)

---

# yoshimulib (English)

Aerospace engineering library for spacecraft dynamics, attitude control, orbital mechanics, and space environment modeling.

Python conversion of the yMATLAB library. 214 MATLAB functions fully converted with 100% coverage.

## Installation

```bash
pip install -r requirements.txt
```

For SPICE functionality:

```bash
pip install spiceypy
```

## Quick Start

```python
import yoshimulib as yl

# Quaternion operations
q = yl.attitude.zyx2q(0.1, 0.2, 0.3)          # Euler angles to quaternion
dcm = yl.attitude.q2dcm(q)                      # Quaternion to DCM
q_err = yl.attitude.q_err(q, [1, 0, 0, 0])      # Quaternion error

# Orbital mechanics
r, v = yl.orbit.oe2rv(7000, 0.01, 0.5, 0, 0, 0) # Orbital elements to r, v
oe = yl.orbit.rv2oe(r, v)                         # r, v to orbital elements
E = yl.orbit.kepler_eq(M=1.0, e=0.01)             # Kepler's equation

# Time conversions
jd = yl.time_utils.cal2jd(2024, 1, 1, 0, 0, 0)   # Calendar to Julian Date
mjd = yl.time_utils.jd2mjd(jd)                     # JD to Modified JD
```

## Modules

### Low-Level Utilities

| Module       | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `math_utils` | Legendre polynomials, skew-symmetric matrices, angle wrapping |
| `conversion` | Unit conversions (AU/km, radians/arcseconds)                  |
| `time_utils` | Time system conversions (JD, MJD, UTC, TT, leap seconds)      |

### Mid-Level Physics

| Module        | Description                                                                           |
| ------------- | ------------------------------------------------------------------------------------- |
| `orbit`       | Orbital mechanics, Kepler's equation, EGM2008, coordinate transforms (GCRF/ITRF/TEME) |
| `environment` | IGRF-12 magnetic field, Jaccia-Bowman 2008 atmosphere, NRLMSISE-00                    |
| `sun_moon`    | Solar and lunar ephemerides (ELP2000-82)                                              |

### High-Level Applications

| Module        | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| `attitude`    | DCM, quaternions, Euler angles, SLERP, TRIAD, kinematics            |
| `srp`         | Solar Radiation Pressure (Simple, Ashikhmin-Shirley, Cook-Torrance) |
| `hifi_srp`    | High-fidelity analytic SRP approximation                            |
| `lightcurves` | Synthetic light curve generation using BRDF models                  |
| `ukf_ckf`     | Unscented / Cubature / Square-Root Cubature Kalman Filters          |
| `object`      | 3D spacecraft model I/O (OBJ/MTL), self-shadowing, ray intersection |
| `utility`     | Visualization and plotting utilities                                |

### Specialized

| Module                  | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `dual_quaternions`      | Dual quaternion operations, ScLERP interpolation     |
| `geometric_integration` | Lie group integrators preserving SO(3) structure     |
| `relative_orbit`        | Relative orbital elements, HCW equations             |
| `orbit_determination`   | Initial orbit determination (Gibbs, Gauss, Double-R) |
| `gpr`                   | Gaussian Process Regression                          |
| `spherical_gaussian`    | Spherical Gaussian functions for BRDF modeling       |
| `spice`                 | SPICE kernel loading (via SpiceyPy)                  |

## Quaternion Convention

Two conventions are supported via the `scalar` parameter:

- `scalar=0`: **q** = [q0, q1, q2, q3]^T = [cos(theta/2), e*sin(theta/2)]^T
- `scalar=4`: **q** = [q1, q2, q3, q4]^T = [e*sin(theta/2), cos(theta/2)]^T

## Coordinate Systems

- **GCRF** - Geocentric Celestial Reference Frame
- **ITRF/ECEF** - Earth-Centered Earth-Fixed
- **J2000** - Inertial frame
- **TEME** - True Equator Mean Equinox
- **RTN** - Radial-Tangential-Normal (relative orbits)

## Author

Yasuhiro Yoshimura (y.yoshimura.a64@m.kyushu-u.ac.jp)

## License

MIT License. See [LICENSE](LICENSE) for details.
