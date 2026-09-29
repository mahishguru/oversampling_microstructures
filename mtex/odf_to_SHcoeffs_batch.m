%% Batch MTEX script: compute SH coefficients for all odf.txt files

% 1. Setup symmetries
CS = crystalSymmetry('6/mmm',[3.2093 3.2093 5.2103],...
    'X||a*','Y||b','Z||c','mineral','Mg');
SS = specimenSymmetry('1');
bandwidth = 18;

% 2. Define input and output directories
baseDir   = 'data/measured';
outDir    = 'data/odf_harmonics';

% Create output directory if it doesn't exist
if ~exist(outDir,'dir')
    mkdir(outDir);
end

% 3. Find all odf.txt files recursively
files = dir(fullfile(baseDir,'**','odf.txt'));

for i = 1:numel(files)
    % Full path to input ODF
    infile = fullfile(files(i).folder, files(i).name);

    % Extract alloy and condition from relative path
    relPath = strrep(files(i).folder, baseDir, '');
    parts   = strsplit(relPath, filesep);
    % parts: {'', 'Alloy_extruded', 'Temp_Strain', 'XRD'}
    alloy      = parts{2};
    temp_str   = parts{3};

    % Construct output filename
    outName  = sprintf('%s_%s_odf_harmonics.txt', alloy, temp_str);
    rawFile  = fullfile(outDir, outName);

    % 4. Load the ODF (assume columns: phi1 phi Phi phi2 weight)
    odf = loadODF_generic(infile, 'cs', CS, 'ss', SS, ...
        'ColumnNames', {'phi1','Phi','phi2','weight'}, ...
        'Bunge', 'delimiter', '\t', 'header', 1);

    % 5. Compute harmonic expansion
    odf_harm = SO3FunHarmonic(odf, 'bandwidth', bandwidth);
    coeffs    = odf_harm.fhat;

    % 6. Write raw real/imag coefficients
    fid = fopen(rawFile,'w');
    if fid==-1
        error('Cannot open output file: %s', rawFile);
    end
    for k = 1:numel(coeffs)
        fprintf(fid, '%.6f %.6f\n', real(coeffs(k)), imag(coeffs(k)));
    end
    fclose(fid);

    fprintf('✅ Written %s (%d coefficients)\n', outName, numel(coeffs));
end

fprintf('All done: SH coefficients saved to %s\n', outDir);
