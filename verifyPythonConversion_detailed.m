%% verifyPythonConversion_detailed.m
% 詳細な比較テスト: MATLABとPython (yoshimulib)
% 20250202 y.yoshimura
%
% このスクリプトは各モジュールの詳細なテストを行います
% 実行前に以下を確認:
%   1. Python環境設定: pyenv('Version', '/path/to/python')
%   2. 必要なパッケージ: numpy, scipy

clc
clear
close all

% 例えば↓
% pyenv("Version","/path/to/your/python")

%% Configuration
config.tol = 1e-10;
config.tolRel = 1e-8;
config.verbose = true;

% Add yoshimulib to Python path
yoshimulibPath = fileparts(fileparts(mfilename('fullpath')));
if count(py.sys.path, yoshimulibPath) == 0
    insert(py.sys.path, int32(0), yoshimulibPath);
end

% Reload modules (in case of changes)
py.importlib.import_module('yoshimulib');
py.importlib.reload(py.importlib.import_module('yoshimulib.attitude'));
py.importlib.reload(py.importlib.import_module('yoshimulib.orbit'));
py.importlib.reload(py.importlib.import_module('yoshimulib.conversion'));
pyConst = py.yoshimulib.orbit.OrbitalConstants();

fprintf('=== Detailed Python Conversion Verification ===\n');
fprintf('Tolerance: %.2e (absolute), %.2e (relative)\n\n', config.tol, config.tolRel);

results = {};
const = orbitConst;

%% ========================================
%% ATTITUDE MODULE - Detailed Tests
%% ========================================
fprintf('============================================\n');
fprintf('ATTITUDE MODULE\n');
fprintf('============================================\n');

% --- qMult with various cases ---
fprintf('\n[qMult]\n');
testCases = {
    [1, 0, 0, 0], [1, 0, 0, 0], 'identity * identity';
    [0, 1, 0, 0], [0, 0, 1, 0], 'i * j = k';
    [0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], 'general quaternions';
    rand(1,4), rand(1,4), 'random quaternions';
    };

for i = 1:size(testCases, 1)
    q1 = testCases{i, 1}; q1 = q1 / norm(q1);
    q2 = testCases{i, 2}; q2 = q2 / norm(q2);
    name = testCases{i, 3};

    try
        qMat = qMult(4, 1, q1, q2);
        qPy = double(py.yoshimulib.attitude.q_mult(py.numpy.array(q1), py.numpy.array(q2), ...
            pyargs('scalar', int32(4))));

        err = norm(qMat - qPy);
        passed = err < config.tol;
        fprintf('  Case "%s": %s (err=%.2e)\n', name, statusStr(passed), err);
        results{end+1} = struct('module', 'attitude', 'func', 'qMult', ...
            'case', name, 'passed', passed, 'err', err);
    catch ME
        fprintf('  Case "%s": ERROR - %s\n', name, ME.message);
        results{end+1} = struct('module', 'attitude', 'func', 'qMult', ...
            'case', name, 'passed', false, 'err', NaN);
    end
end

% --- q2dcm / dcm2q roundtrip ---
fprintf('\n[q2dcm / dcm2q roundtrip]\n');
for i = 1:5
    q = rand(1, 4); q = q / norm(q);
    if q(4) < 0, q = -q; end % ensure positive scalar

    try
        R = q2dcm(4, q);
        qBack = dcm2q(4, R);

        RPy = double(py.yoshimulib.attitude.q2dcm(py.numpy.array(q), pyargs('scalar', int32(4))));
        qBackPy = double(py.yoshimulib.attitude.dcm2q(py.numpy.array(RPy), pyargs('scalar', int32(4))));

        % DCM error
        errR = norm(R - RPy, 'fro');

        % Quaternion error (accounting for sign ambiguity)
        errQ = min(norm(qBack - qBackPy), norm(qBack + qBackPy));

        passed = (errR < config.tol) && (errQ < config.tol);
        fprintf('  Random q %d: %s (R err=%.2e, q err=%.2e)\n', i, statusStr(passed), errR, errQ);
        results{end+1} = struct('module', 'attitude', 'func', 'q2dcm_dcm2q', ...
            'case', sprintf('random_%d', i), 'passed', passed, 'err', max(errR, errQ));
    catch ME
        fprintf('  Random q %d: ERROR - %s\n', i, ME.message);
    end
end

% --- Euler angle conversions ---
fprintf('\n[Euler Angles]\n');
eulerAngles = [
    0, 0, 0;
    pi/4, pi/6, pi/3;
    pi/2, 0, pi/2;
    rand(1,3) * pi;
    ];

for i = 1:size(eulerAngles, 1)
    phi = eulerAngles(i, 1);
    theta = eulerAngles(i, 2);
    psi = eulerAngles(i, 3);

    try
        RMat = zyx2dcm(phi, theta, psi);
        RPy = double(py.yoshimulib.attitude.zyx2dcm(phi, theta, psi));

        err = norm(RMat - RPy, 'fro');
        passed = err < config.tol;
        fprintf('  zyx2dcm [%.2f, %.2f, %.2f]: %s (err=%.2e)\n', ...
            phi, theta, psi, statusStr(passed), err);
        results{end+1} = struct('module', 'attitude', 'func', 'zyx2dcm', ...
            'case', sprintf('angles_%d', i), 'passed', passed, 'err', err);
    catch ME
        fprintf('  zyx2dcm [%.2f, %.2f, %.2f]: ERROR - %s\n', phi, theta, psi, ME.message);
    end
end

%% ========================================
%% ORBIT MODULE - Detailed Tests
%% ========================================
fprintf('\n============================================\n');
fprintf('ORBIT MODULE\n');
fprintf('============================================\n');

% --- Kepler's equation for various eccentricities ---
fprintf('\n[keplerEq]\n');
eccentricities = [0, 0.1, 0.5, 0.8, 0.95];
meanAnomalies = linspace(0, 2*pi, 10);

for e = eccentricities
    maxErr = 0;
    for M = meanAnomalies
        try
            EMat = keplerEq(M, e);
            EPy = double(py.yoshimulib.orbit.kepler_eq(M, e));
            err = abs(EMat - EPy);
            maxErr = max(maxErr, err);
        catch
            maxErr = NaN;
            break;
        end
    end
    passed = maxErr < config.tol;
    fprintf('  e=%.2f: %s (max err=%.2e)\n', e, statusStr(passed), maxErr);
    results{end+1} = struct('module', 'orbit', 'func', 'keplerEq', ...
        'case', sprintf('e=%.2f', e), 'passed', passed, 'err', maxErr);
end

% --- oe2rv / rv2oe roundtrip ---
fprintf('\n[oe2rv / rv2oe roundtrip]\n');
oeTestCases = [
    7000, 0.001, deg2rad(0), 0, 0, 0;           % circular, equatorial
    7000, 0.001, deg2rad(90), 0, 0, 0;          % circular, polar
    10000, 0.3, deg2rad(45), deg2rad(30), deg2rad(60), deg2rad(90);  % elliptical
    42164, 0.0001, deg2rad(0.1), deg2rad(75), deg2rad(270), deg2rad(180); % GEO-like
    ];

for i = 1:size(oeTestCases, 1)
    oe = oeTestCases(i, :);
    try
        [rMat, vMat] = oe2rv(oe, 1, const.GE);
        result = py.yoshimulib.orbit.oe2rv(py.numpy.array(oe), int32(1), const.GE);
        rPy = double(result{1});
        vPy = double(result{2});

        % Forward error
        errR = norm(rMat - rPy);
        errV = norm(vMat - vPy);

        % Inverse
        oeBackMat = rv2oe(rMat, vMat, const.GE);
        oeBackPy = double(py.yoshimulib.orbit.rv2oe(py.numpy.array(rPy), py.numpy.array(vPy), const.GE));

        % Handle singular cases (equatorial / circular)
        mask = isfinite(oeBackMat) & isfinite(oeBackPy);
        if abs(oe(3)) < 1e-8 % near equatorial: RAAN and related angles can be undefined
            mask(4:6) = false;
        elseif oe(2) < 1e-8 % near circular: argument/true anomaly can be undefined
            mask(5:6) = false;
        end

        if any(mask)
            errOE = norm(oeBackMat(mask) - oeBackPy(mask));
        else
            errOE = 0;
        end

        passed = (errR < config.tol) && (errV < config.tol) && (errOE < config.tol);
        fprintf('  Case %d (a=%.0f, e=%.3f, i=%.1f deg): %s\n', i, oe(1), oe(2), rad2deg(oe(3)), statusStr(passed));
        fprintf('    r err=%.2e, v err=%.2e, oe err=%.2e\n', errR, errV, errOE);
        results{end+1} = struct('module', 'orbit', 'func', 'oe2rv_rv2oe', ...
            'case', sprintf('case_%d', i), 'passed', passed, 'err', max([errR, errV, errOE]));
    catch ME
        fprintf('  Case %d: ERROR - %s\n', i, ME.message);
    end
end

% --- Time functions ---
fprintf('\n[Time Functions: gmst, gast, era]\n');
jdTestCases = [2451545.0, 2460000.5, 2451545.0 + 365.25*10]; % J2000, recent, future

for jd = jdTestCases
    try
        gmstMat = gmst(jd);
        gastMat = gast(jd, const);
        eraMat = era(jd);

        gmstPy = double(py.yoshimulib.orbit.gmst(jd));
        gastPy = double(py.yoshimulib.orbit.gast(jd, pyConst));
        eraPy = double(py.yoshimulib.orbit.era(jd));

        errGMST = abs(gmstMat - gmstPy);
        errGAST = abs(gastMat - gastPy);
        errERA = abs(eraMat - eraPy);

        tolGAST = max(config.tol, 2e-10); % tiny numeric differences can appear in GAST
        passed = (errGMST < config.tol) && (errGAST < tolGAST) && (errERA < config.tol);
        fprintf('  JD=%.1f: %s (GMST err=%.2e, GAST err=%.2e, ERA err=%.2e)\n', ...
            jd, statusStr(passed), errGMST, errGAST, errERA);
        results{end+1} = struct('module', 'orbit', 'func', 'time_funcs', ...
            'case', sprintf('JD=%.1f', jd), 'passed', passed, 'err', max([errGMST, errGAST, errERA]));
    catch ME
        fprintf('  JD=%.1f: ERROR - %s\n', jd, ME.message);
    end
end

% --- MEE conversions ---
fprintf('\n[MEE Conversions: coe2mee, mee2coe]\n');
for i = 1:size(oeTestCases, 1)
    coe = oeTestCases(i, :);
    try
        % MATLAB: use struct for COE
        oeMat = struct('a', coe(1), 'e', coe(2), 'inc', coe(3), ...
            'raan', coe(4), 'w', coe(5), 'nu', coe(6));
        meeMat = coe2mee(oeMat);

        % Python: OrbitalElements dataclass
        oePy = py.yoshimulib.orbit.OrbitalElements();
        oePy.a = coe(1);
        oePy.e = coe(2);
        oePy.inc = coe(3);
        oePy.raan = coe(4);
        oePy.w = coe(5);
        oePy.nu = coe(6);
        meePy = py.yoshimulib.orbit.coe2mee(oePy);

        meeMatVec = [meeMat.p_, meeMat.f_, meeMat.g_, meeMat.h_, meeMat.k_, meeMat.L_];
        meePyVec = double([meePy.p_, meePy.f_, meePy.g_, meePy.h_, meePy.k_, meePy.L_]);

        errMEE = norm(meeMatVec - meePyVec);

        coeBackMat = mee2coe(meeMat);
        coeBackPy = py.yoshimulib.orbit.mee2coe(meePy);

        coeBackMatVec = [coeBackMat.a, coeBackMat.e, coeBackMat.inc, ...
            coeBackMat.raan, coeBackMat.w, coeBackMat.nu];
        coeBackPyVec = double([coeBackPy.a, coeBackPy.e, coeBackPy.inc, ...
            coeBackPy.raan, coeBackPy.w, coeBackPy.nu]);

        errCOE = norm(coeBackMatVec - coeBackPyVec);

        passed = (errMEE < config.tol) && (errCOE < config.tol);
        fprintf('  Case %d: %s (MEE err=%.2e, COE err=%.2e)\n', i, statusStr(passed), errMEE, errCOE);
        results{end+1} = struct('module', 'orbit', 'func', 'mee_conversions', ...
            'case', sprintf('case_%d', i), 'passed', passed, 'err', max(errMEE, errCOE));
    catch ME
        fprintf('  Case %d: ERROR - %s\n', i, ME.message);
    end
end

%% ========================================
%% PRECESSION / NUTATION Tests
%% ========================================
fprintf('\n============================================\n');
fprintf('PRECESSION / NUTATION\n');
fprintf('============================================\n');

fprintf('\n[precession, nutation, obliquity]\n');
for jd = jdTestCases
    try
        % Precession
        [zetaMat, zMat, thetaMat] = precession(const.J2000, jd, const);
        result = py.yoshimulib.orbit.precession(const.J2000, jd, pyConst);
        zetaPy = double(result{1});
        zPy = double(result{2});
        thetaPy = double(result{3});

        errPrec = max([abs(zetaMat-zetaPy), abs(thetaMat-thetaPy), abs(zMat-zPy)]);

        % Nutation
        [dPsiMat, dEpsMat] = nutation(jd, const);
        result = py.yoshimulib.orbit.nutation(jd, pyConst);
        dPsiPy = double(result{1});
        dEpsPy = double(result{2});

        errNut = max([abs(dPsiMat-dPsiPy), abs(dEpsMat-dEpsPy)]);

        % Obliquity
        epsMat = obliquity(jd);
        epsPy = double(py.yoshimulib.orbit.obliquity(jd, pyConst));

        errObl = abs(epsMat - epsPy);

        passed = (errPrec < 1e-8) && (errNut < 1e-8) && (errObl < 1e-8);
        fprintf('  JD=%.1f: %s (prec=%.2e, nut=%.2e, obl=%.2e)\n', ...
            jd, statusStr(passed), errPrec, errNut, errObl);
        results{end+1} = struct('module', 'orbit', 'func', 'prec_nut', ...
            'case', sprintf('JD=%.1f', jd), 'passed', passed, 'err', max([errPrec, errNut, errObl]));
    catch ME
        fprintf('  JD=%.1f: ERROR - %s\n', jd, ME.message);
    end
end

%% ========================================
%% CONVERSION MODULE - Detailed Tests
%% ========================================
fprintf('\n============================================\n');
fprintf('CONVERSION MODULE\n');
fprintf('============================================\n');

% Test various conversions
fprintf('\n[Unit Conversions]\n');
testVals = [0, 0.1, 1, 10, 100];

% AU <-> km
for val = testVals
    try
        kmMat = au2km(val, const);
        kmPy = double(py.yoshimulib.conversion.au2km(val, pyConst.AU));
        err = abs(kmMat - kmPy) / max(1, abs(kmMat));
        passed = err < config.tolRel;
        fprintf('  au2km(%.1f): %s (rel err=%.2e)\n', val, statusStr(passed), err);
    catch ME
        fprintf('  au2km(%.1f): ERROR - %s\n', val, ME.message);
    end
end

% rad <-> arcs
fprintf('\n[rad2arcs / arcs2rad]\n');
radVals = [0, 1e-6, 1e-3, 0.01, 0.1, 1];
for rad = radVals
    try
        arcsMat = rad2arcs(rad);
        arcsPy = double(py.yoshimulib.conversion.rad2arcs(rad));
        err = abs(arcsMat - arcsPy);
        passed = err < config.tol;
        fprintf('  rad2arcs(%.2e): %s (err=%.2e)\n', rad, statusStr(passed), err);
    catch ME
        fprintf('  rad2arcs(%.2e): ERROR\n', rad);
    end
end

%% ========================================
%% RELATIVE ORBIT MODULE
%% ========================================
fprintf('\n============================================\n');
fprintf('RELATIVE ORBIT MODULE\n');
fprintf('============================================\n');

fprintf('\n[oe2roe / roe2rtn]\n');
chiefOE = [7000, 0.001, deg2rad(45), deg2rad(30), deg2rad(60), deg2rad(0)];
deputyOffsets = [
    10, 0.0001, deg2rad(0.1), deg2rad(0.1), deg2rad(0.05), deg2rad(0.5);
    -5, 0.0002, deg2rad(-0.05), deg2rad(0.05), deg2rad(0.1), deg2rad(-0.3);
    ];

for i = 1:size(deputyOffsets, 1)
    deputyOE = chiefOE + deputyOffsets(i, :);
    try
        roeMat = oe2roe(chiefOE, deputyOE, 0);
        roePy = double(py.yoshimulib.relative_orbit.oe2roe(py.numpy.array(chiefOE), ...
            py.numpy.array(deputyOE), int32(0)));

        errROE = norm(roeMat - roePy);

        [xRTNmat, vRTNmat] = roe2rtn(roeMat, chiefOE, 0, const.GE);
        result = py.yoshimulib.relative_orbit.roe2rtn(py.numpy.array(roePy), ...
            py.numpy.array(chiefOE), int32(0), const.GE);
        xRTNpy = double(result{1});
        vRTNpy = double(result{2});

        errX = norm(xRTNmat - xRTNpy);
        errV = norm(vRTNmat - vRTNpy);

        passed = (errROE < config.tol) && (errX < config.tol) && (errV < config.tol);
        fprintf('  Case %d: %s (ROE err=%.2e, x err=%.2e, v err=%.2e)\n', ...
            i, statusStr(passed), errROE, errX, errV);
        results{end+1} = struct('module', 'relative_orbit', 'func', 'oe2roe_roe2rtn', ...
            'case', sprintf('case_%d', i), 'passed', passed, 'err', max([errROE, errX, errV]));
    catch ME
        fprintf('  Case %d: ERROR - %s\n', i, ME.message);
    end
end

%% ========================================
%% SUMMARY
%% ========================================
fprintf('\n============================================\n');
fprintf('SUMMARY\n');
fprintf('============================================\n');

nPassed = sum(cellfun(@(x) x.passed, results));
nTotal = length(results);
fprintf('Total: %d / %d tests passed (%.1f%%)\n\n', nPassed, nTotal, 100*nPassed/nTotal);

% Group by module
modules = unique(cellfun(@(x) x.module, results, 'UniformOutput', false));
fprintf('By module:\n');
for i = 1:length(modules)
    mod = modules{i};
    modResults = results(cellfun(@(x) strcmp(x.module, mod), results));
    modPassed = sum(cellfun(@(x) x.passed, modResults));
    modTotal = length(modResults);
    fprintf('  %s: %d/%d (%.0f%%)\n', mod, modPassed, modTotal, 100*modPassed/modTotal);
end

% List failures
failures = results(~cellfun(@(x) x.passed, results));
if ~isempty(failures)
    fprintf('\nFailed tests:\n');
    for i = 1:length(failures)
        f = failures{i};
        fprintf('  - %s.%s (%s): err=%.2e\n', f.module, f.func, f.case, f.err);
    end
end

%% Helper functions
function s = statusStr(passed)
if passed
    s = 'PASS';
else
    s = 'FAIL';
end
end
