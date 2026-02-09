function run_verifyPythonConversion_detailed()
% Run verifyPythonConversion_detailed.m with logging for batch execution.
% This wrapper does not change pyenv; adjust verifyPythonConversion_detailed.m if needed.

rootDir = fileparts(mfilename('fullpath'));
logDir = fullfile(rootDir, 'verification_logs');
if ~exist(logDir, 'dir')
    mkdir(logDir);
end

stamp = datestr(now, 'yyyymmdd_HHMMSS');
logFile = fullfile(logDir, ['verifyPythonConversion_detailed_' stamp '.log']);
matFile = fullfile(logDir, ['verifyPythonConversion_detailed_' stamp '.mat']);

diary(logFile);
diary on;

fprintf('=== Batch Runner: verifyPythonConversion_detailed ===\n');
fprintf('Timestamp: %s\n', stamp);
fprintf('Root: %s\n', rootDir);

try
    evalin('base', 'verifyPythonConversion_detailed'); % run in base workspace
catch ME
    fprintf('\nERROR:\n');
    disp(getReport(ME, 'extended'));
end

if evalin('base', 'exist(''results'', ''var'')')
    try
        results = evalin('base', 'results');
        save(matFile, 'results');
        fprintf('Saved results to: %s\n', matFile);
    catch ME
        fprintf('Failed to save results MAT: %s\n', ME.message);
    end
else
    fprintf('No variable named "results" found.\n');
end

diary off;

fprintf('Log: %s\n', logFile);
end
