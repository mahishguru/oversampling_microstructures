%% 1. Setup Parent Directory
clear; clc; close all;

% Set default figure visibility off so MATLAB doesn't flash figures
set(0, 'DefaultFigureVisible', 'off');

% UPDATED PARENT DIRECTORY
parentDir = 'data/odf_harmonics_sampled';

%% 2. Setup Parallel Pool (Control Threads)
numThreads = 8; % <--- CHANGE THIS TO YOUR DESIRED NUMBER OF THREADS/CORES

% Check current parallel pool
p = gcp('nocreate');
if isempty(p)
    fprintf('Starting parallel pool with %d workers...\n', numThreads);
    parpool('local', numThreads);
elseif p.NumWorkers ~= numThreads
    fprintf('Restarting parallel pool to use %d workers...\n', numThreads);
    delete(p);
    parpool('local', numThreads);
end

%% 3. Find and Filter Files (Skip Already Processed)
files = dir(fullfile(parentDir, '**', '*.txt'));

validIdx = false(length(files), 1);
for i = 1:length(files)
    inName = files(i).name;
    inDir = files(i).folder;
    
    % Skip if the file itself is a generated output
    if contains(inName, '_ODF.txt') || contains(inName, '_dream3d_odf_angles.txt')
        continue;
    end
    
    % Define Expected Output Paths 
    [~, baseName, ~] = fileparts(inName);
    outODF = fullfile(inDir, [baseName, '_ODF.txt']);
    
    % Skip if the ODF output already exists
    if isfile(outODF)
        continue; 
    end
    
    % Flag for processing
    validIdx(i) = true;
end

filesToProcess = files(validIdx);
numFiles = length(filesToProcess);
fprintf('Found %d unprocessed files to run. Starting PARALLEL batch processing...\n', numFiles);

%% 3.5 Set up Parallel Data Queue for True Sequential Progress Tracking
D = parallel.pool.DataQueue;

% Clear the persistent memory before starting, in case you run the script multiple times
clear updateProgress;

% Attach the callback function to the queue
afterEach(D, @(~) updateProgress(numFiles));

%% 4. Parallel Batch Process Each File
parfor i = 1:numFiles
    
    % =========================================================================
    % WORKER INITIALIZATION
    % =========================================================================
    setMTEXpref('EulerAngleConvention', 'Bunge');
    setMTEXpref('xAxisDirection', 'east');
    setMTEXpref('zAxisDirection', 'outOfPlane');
    setMTEXpref('quiet', true); 
    
    % CRITICAL: Disable NFFT cache to prevent matrix dimension crashes in parfor
    setMTEXpref('useNFFT', false);
    
    CS_local = crystalSymmetry('6/mmm', [3.2093, 3.2093, 5.2103], ...
        'X||a*','Y||b','Z||c', 'mineral', 'Mg');
    SS_local = specimenSymmetry('1');
    
    h_local = {
      Miller(1,0,-1,0, CS_local), ...
      Miller(0,0,0,2, CS_local), ...
      Miller(1,1,-2,0, CS_local)
    };
    
    inDir = filesToProcess(i).folder;
    inName = filesToProcess(i).name;
    
    inFile = fullfile(inDir, inName);
    [~, baseName, ~] = fileparts(inName);
    
    outODF = fullfile(inDir, [baseName, '_ODF.txt']);
    outJPG = fullfile(inDir, [baseName, '_pf.jpg']);
    
    try
        %% A. Read Harmonic Coefficients
        data = readmatrix(inFile); 
        
        % Clean out any pure NaNs (caused by text headers or blank lines)
        data(any(isnan(data), 2), :) = [];
        
        if isempty(data) || size(data, 2) < 2
            warning('❌ Skipping %s: Empty or missing columns.', inName);
            continue;
        end
        
        % Form strict complex column vector
        fhat = complex(data(:,1), data(:,2)); 
        fhat = fhat(:); 
        
        %% B. Reconstruct ODF using the local thread symmetries
        try
            % Attempt standard creation
            odf_recons = SO3FunHarmonic(fhat, CS_local, SS_local);
        catch
            % If length mathematically mismatches the exact symmetry requirements, rebuild safely
            base_odf = SO3FunHarmonic(fhat); 
            base_fhat = base_odf.fhat; 
            odf_recons = SO3FunHarmonic(base_fhat, CS_local, SS_local);
        end
        
        % %% C. Visualize and Save JPG (Fully Invisible Mode)
        % % Use MTEX's figure creator so it handles the 1x3 subplot grid automatically
        % mtexFig = mtexFigure('Visible', 'off'); 
        % 
        % plotPDF(odf_recons, h_local, 'minmax', 'reduced', ...
        %     1:0.5:12, 'antipodal', 'silent'); 
        % 
        % currentMTEXFig = gcm;
        % mtexColorMap(currentMTEXFig, parula);
        % CLim(currentMTEXFig, 'equal'); 
        % mtexColorbar(currentMTEXFig);
        % 
        % drawNow(currentMTEXFig); 
        % 
        % % Export the active invisible figure (gcf works locally per parfor worker)
        % exportgraphics(gcf, outJPG, 'Resolution', 300, ...
        %     'Padding', 'figure', 'BackgroundColor', 'w');
        % 
        % % Close figure using safe worker-local command
        % close(gcf); 
        
        %% D. Export ODF to Bunge Euler Angle Format
        export(odf_recons, outODF, 'Bunge');
        
        % Send signal to the main thread that one file has finished
        send(D, true);
        
    catch ME
        warning('❌ Error processing %s:\n%s', inName, ME.message);
        
        % Safely clean up worker memory if a crash occurs
        close all;
    end
end

% Restore default figure visibility
set(0, 'DefaultFigureVisible', 'on');
fprintf('\n🎉 Parallel batch processing completed successfully!\n');


%% 5. Progress Tracking Function
% This function executes entirely in the main MATLAB thread, strictly sequentially
function updateProgress(totalFiles)
    persistent completedCount;
    
    if isempty(completedCount)
        completedCount = 0;
    end
    
    completedCount = completedCount + 1;
    
    % Print progress every 500 files
    if mod(completedCount, 500) == 0
        fprintf('✅ Progress: Processed %d out of %d files...\n', completedCount, totalFiles);
    end
end