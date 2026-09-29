%% Batch convert GSH coefficient files -> DREAM.3D uniform ODF angle files
%
% Best operating mode (validated via pole-figure correlation analysis):
%   sigma     = 1.0   (sigma < 1 breaks MatchCrystallography; > 1 over-smooths)
%   nSamples  = 100000 (raw discrete samples from ODF; no binning)
%   weights   = uniform (every sample gets weight 1; density of samples
%               in Euler space encodes texture via discreteSample)
%   MaxIter   = 500000 (set in 04_generate_rves.py, not here)
%
% NOTE: This writes raw continuous Euler angles from discreteSample, NOT
% binned grid centers. This matches the rigorously validated test approach
% in Dream3D_generation/test/validation_optimized/.
%
clear; clc; close all;
set(0, 'DefaultFigureVisible', 'off');

%% -------------------- User settings --------------------
parentDir    = 'data/odf_harmonics_sampled';
numThreads   = 8;

% Sampling from reconstructed MTEX ODF
nSamples     = 100000;     % 100k raw discrete samples (no binning)
rngSeed      = 1;

% DREAM.3D export settings
sigmaVal     = 1.0;        % validated optimum; sigma < 1 collapses texture
outSubDir    = 'dream3d_angles_uniform';

%% Crystal / specimen symmetry
CS = crystalSymmetry('6/mmm', [3.2093 3.2093 5.2103], ...
    'X||a*', 'Y||b', 'Z||c', 'mineral', 'Mg', 'color', 'light blue');
SS = specimenSymmetry('1');
%% ------------------------------------------------------

%% 1. Parallel pool
p = gcp('nocreate');
if isempty(p)
    fprintf('Starting parallel pool with %d workers...\n', numThreads);
    parpool('local', numThreads);
elseif p.NumWorkers ~= numThreads
    fprintf('Restarting parallel pool to use %d workers...\n', numThreads);
    delete(p);
    parpool('local', numThreads);
end

%% 2. Find files
files = dir(fullfile(parentDir, '**', '*.txt'));

validIdx = false(numel(files), 1);
for i = 1:numel(files)
    inName = files(i).name;
    inDir  = files(i).folder;

    % Skip generated outputs
    if contains(inName, '_ODF.txt') || contains(inName, '_dream3d_odf_angles.txt')
        continue;
    end

    [~, baseName, ~] = fileparts(inName);
    outDir  = fullfile(inDir, outSubDir);
    outFile = fullfile(outDir, [baseName '_dream3d_odf_angles.txt']);

    if isfile(outFile)
        continue;
    end

    validIdx(i) = true;
end

filesToProcess = files(validIdx);
numFiles = numel(filesToProcess);
fprintf('Found %d unprocessed files. Starting weighted DREAM.3D export...\n', numFiles);

%% 3. Progress tracking
D = parallel.pool.DataQueue;
clear updateProgress;
afterEach(D, @(~) updateProgress(numFiles));

%% 4. Parallel batch
parfor i = 1:numFiles

    setMTEXpref('EulerAngleConvention', 'Bunge');
    setMTEXpref('xAxisDirection', 'east');
    setMTEXpref('zAxisDirection', 'outOfPlane');
    setMTEXpref('quiet', true);
    setMTEXpref('useNFFT', false);

    CS_local = crystalSymmetry('6/mmm', [3.2093 3.2093 5.2103], ...
        'X||a*', 'Y||b', 'Z||c', 'mineral', 'Mg', 'color', 'light blue');
    SS_local = specimenSymmetry('1');

    inDir  = filesToProcess(i).folder;
    inName = filesToProcess(i).name;
    inFile = fullfile(inDir, inName);

    [~, baseName, ~] = fileparts(inName);
    outDir  = fullfile(inDir, outSubDir);
    outFile = fullfile(outDir, [baseName '_dream3d_odf_angles.txt']);

    try
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end

        rng(rngSeed + i, 'twister');

        convert_gsh_to_dream3d_raw( ...
            inFile, outFile, CS_local, SS_local, ...
            nSamples, sigmaVal);

        send(D, true);

    catch ME
        warning('❌ Error processing %s:\n%s', inFile, ME.message);
        close all;
    end
end

set(0, 'DefaultFigureVisible', 'on');
fprintf('\n🎉 Parallel uniform DREAM.3D export completed!\n');

%% ========================================================================
function convert_gsh_to_dream3d_raw( ...
    inFile, outFile, CS, SS, ...
    nSamples, sigmaVal)

    %% A. Read harmonic coefficients
    data = readmatrix(inFile);
    data(any(isnan(data), 2), :) = [];

    if isempty(data) || size(data, 2) < 2
        error('Input file is empty or missing real/imag columns: %s', inFile);
    end

    fhat = complex(data(:,1), data(:,2));
    fhat = fhat(:);

    %% B. Reconstruct ODF
    try
        odf_recons = SO3FunHarmonic(fhat, CS, SS);
    catch
        base_odf   = SO3FunHarmonic(fhat);
        odf_recons = SO3FunHarmonic(base_odf.fhat, CS, SS);
    end

    %% C. Randomly sample orientations from the continuous ODF
    ori = discreteSample(odf_recons, nSamples);

    %% D. Convert to Bunge Euler angles in degrees
    [phi1, Phi, phi2] = Euler(ori, 'Bunge');

    phi1 = phi1 ./ degree;
    Phi  = Phi  ./ degree;
    phi2 = phi2 ./ degree;

    phi1 = mod(phi1, 360.0);
    Phi  = mod(Phi, 180.0);
    phi2 = mod(phi2, 360.0);

    %% E. Write raw samples with uniform weight and sigma
    weights = ones(nSamples, 1);
    sigmas  = sigmaVal * ones(nSamples, 1);

    outData = [phi1(:), Phi(:), phi2(:), weights, sigmas];

    %% F. Write DREAM.3D StatsGenerator angle file
    fid = fopen(outFile, 'w');
    if fid < 0
        error('Cannot open output file: %s', outFile);
    end

    fprintf(fid, '# DREAM.3D StatsGenerator Angles Input File\n');
    fprintf(fid, '# Generated from GSH coefficients via MTEX ODF reconstruction\n');
    fprintf(fid, '# Export mode: raw discreteSample (no binning)\n');
    fprintf(fid, '# Source: %s\n', inFile);
    fprintf(fid, '# Euler0 Euler1 Euler2 Weight Sigma\n');
    fprintf(fid, 'Angle Count:%d\n', nSamples);
    fprintf(fid, '%.6f %.6f %.6f %.6f %.6f\n', outData.');

    fclose(fid);

    fprintf('✅ Wrote: %s\n', outFile);
    fprintf('   Samples     : %d\n', nSamples);
    fprintf('   Sigma       : %.3f\n', sigmaVal);
end

%% ========================================================================
function updateProgress(totalFiles)
    persistent completedCount;

    if isempty(completedCount)
        completedCount = 0;
    end

    completedCount = completedCount + 1;

    if mod(completedCount, 500) == 0 || completedCount == totalFiles
        fprintf('✅ Progress: %d / %d completed\n', completedCount, totalFiles);
    end
end