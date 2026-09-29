%% Publication pole figures: AZ31 validation set + oversampled ODF samples
% Figure 1 (pf_publication_AZ31_validation): 3x3 panel
%   row 1 : RVE voxels    (AZ31_validation_voxel_orientations.txt, deg)
%   row 2 : DREAM.3D RVE  (AZ31_Phenopower_new.dream3d CellData EulerAngles, rad)
%   row 3 : Input ODF     (AZ31_odf_dream3d_odf_angles.txt, StatsGenerator input)
% Figure 2 (pf_publication_AZ31_oversampled): 2x3 panel
%   rows  : oversampled GSH-harmonic ODFs, samples 3323 and 4968
% columns: {10-10}, {0002}, {11-20}. ONE shared MRD colorbar, fixed [0 4].
set(0, 'DefaultFigureVisible', 'off');
base = fileparts(mfilename('fullpath'));
% inputs: <repo>/data/validation_AZ31/ (RVE, its input angle file, voxel
% orientations from validation/extract_rve_euler.py, and a folder
% harmonics_oversampled/ with oversampled GSH files); figures are written there too
dataDir = fullfile(dataDir, '..', 'data', 'validation_AZ31');
% GSH harmonics staged from the database backup
% (data/odf_harmonics_sampled/AZ31_extruded)
harmDir = fullfile(dataDir, 'harmonics_oversampled');

CS = crystalSymmetry('6/mmm', [3.2093 3.2093 5.2103], ...
    'X||a*', 'Y||b', 'Z||c', 'mineral', 'Mg');
SS = specimenSymmetry('1');
setMTEXpref('xAxisDirection', 'east');
setMTEXpref('zAxisDirection', 'OutOfPlane');
setMTEXpref('EulerAngleConvention', 'Bunge');
setMTEXpref('FontSize', 22);
setMTEXpref('pfAnnotations', @(varargin) []);
h = [Miller(1,0,-1,0, CS), Miller(0,0,0,2, CS), Miller(1,1,-2,0, CS)];

halfwidth_deg = 10;
cRange = [0 4];

%% ---- Figure 1: validation set --------------------------------------------
% row 1: voxel orientations txt (phi1 Phi phi2, degrees, no header)
m = readmatrix(fullfile(dataDir, 'AZ31_validation_voxel_orientations.txt'));
o = orientation.byEuler(m(:,1)*degree, m(:,2)*degree, m(:,3)*degree, CS, SS);
odfs{1} = calcDensity(o, 'halfwidth', halfwidth_deg*degree);

% row 2: .dream3d RVE voxel Euler angles (radians)
e = h5read(fullfile(dataDir, 'AZ31_Phenopower_new.dream3d'), ...
    '/DataContainers/SyntheticVolumeDataContainer/CellData/EulerAngles');
e = reshape(e, 3, []).';
o = orientation.byEuler(e(:,1), e(:,2), e(:,3), CS, SS);
odfs{2} = calcDensity(o, 'halfwidth', halfwidth_deg*degree);

% row 3: DREAM.3D StatsGenerator angle input (deg + weight, 5 header lines)
m = readmatrix(fullfile(dataDir, 'AZ31_odf_dream3d_odf_angles.txt'), ...
    'NumHeaderLines', 5, 'FileType', 'text');
o = orientation.byEuler(m(:,1)*degree, m(:,2)*degree, m(:,3)*degree, CS, SS);
odfs{3} = calcDensity(o, 'weights', m(:,4), 'halfwidth', halfwidth_deg*degree);

rowlab = {'RVE voxels', 'DREAM.3D RVE', 'Input ODF'};
make_panel(odfs, rowlab, h, cRange, fullfile(dataDir, 'pf_publication_AZ31_validation'));

%% ---- Figure 2: oversampled harmonic ODF samples --------------------------
samples = [3323 4968];
odfs2 = cell(1, numel(samples));
for i = 1:numel(samples)
    d = readmatrix(fullfile(harmDir, sprintf('AZ31_extruded_%d.txt', samples(i))));
    odfs2{i} = SO3FunHarmonic(complex(d(:,1), d(:,2)), CS, SS);
end
rowlab2 = {'Sample 3323', 'Sample 4968'};
make_panel(odfs2, rowlab2, h, cRange, fullfile(dataDir, 'pf_publication_AZ31_oversampled'));

set(0, 'DefaultFigureVisible', 'on');

%% ---------------------------------------------------------------------------
function make_panel(odfs, rowlab, h, cRange, out)
    n = numel(odfs);
    fprintf('\n== %s ==\n%-15s | %-8s | %-10s\n', out, 'row', 'ODF L1', 'texIdx');
    fprintf('%s\n', repmat('-', 1, 42));
    for i = 1:n
        fprintf('%-15s | %8.4f | %10.3f\n', rowlab{i}, ...
            calcError(odfs{i}, odfs{1}, 'L1'), norm(odfs{i})^2);
    end
    fprintf('shared colour range: [0 %.1f] MRD (fixed)\n', cRange(2));

    newMtexFigure('nrows', n, 'ncols', 3, 'visible', 'off', 'outerPlotSpacing', 45);
    set(gcf, 'Units', 'pixels', 'Position', [100 100 1500 100+400*n], 'Color', 'w');
    for i = 1:n
        plotPDF(odfs{i}, h, 'contourf', 'antipodal', 'silent', ...
            'resolution', 2.5*degree);
        if i < n; nextAxis; end
    end
    mtexColorMap jet;
    setColorRange(cRange);
    mtexColorbar('title', 'MRD');
    drawNow(gcm, 'figSize', 'large');

    % Miller labels only above the top row, with extra spacing
    mtexFig = gcm; allAx = mtexFig.children(:);
    for k = 1:numel(allAx)
        if k <= 3
            allAx(k).Title.Units = 'normalized';
            allAx(k).Title.Position(2) = allAx(k).Title.Position(2) + 0.08;
        else
            allAx(k).Title.String = '';
        end
    end

    lab_ax = axes('Position', [0 0 1 1], 'Visible', 'off', 'HitTest', 'off');
    for i = 1:n
        yc = 1 - (i - 0.5)/n;
        text(lab_ax, 0.012, yc, rowlab{i}, 'Rotation', 90, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontWeight', 'bold', 'FontSize', 20, 'Color', 'k');
    end

    print(gcf, [out '.png'], '-dpng', '-r300');
    exportgraphics(gcf, [out '.pdf'], 'ContentType', 'image', 'Resolution', 300);
    close(gcf);
    fprintf('saved %s.png / .pdf\n', out);
end
