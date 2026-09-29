clear all;
clc;
close all;

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = crystalSymmetry('6/mmm', [3.2093 3.2093 5.2103], 'X||a*', 'Y||b', 'Z||c', 'mineral', 'Mg', 'color', 'light blue');

% specimen symmetry
SS = specimenSymmetry('1');

% plotting convention
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','OutOfPlane');

%% Specify Miller Indice

h = { ...
  Miller(1, 0,-1, 0,CS),...
  Miller(0, 0, 0, 2,CS),...
  Miller(1, 0,-1, 1,CS),...
  Miller(1, 0,-1, 2,CS),...
  Miller(1, 1,-2, 0,CS),...
  Miller(1, 0,-1, 3,CS),...
  };

%% Rotate for Rods

x = vector3d(1, 0, 0);

y = vector3d(0, 1, 0);

z = vector3d(0, 0, 1);


q_x = axis2quat(x,90*degree);

q_y = axis2quat(y, 90*degree);

q_z = axis2quat(z, 90*degree);

%% Main

% Specify the main directory
mainDir = 'data/measured';  % <alloy>/<T_v>/XRD/*.xrdml; all conditions below are processed

% Get a list of subdirectories containing 'XRD'
subDirs = dir(fullfile(mainDir, '**', 'XRD'));

% Assuming subDirs is your structure array
folderValues = {subDirs.folder};  % Extract folder values into a cell array
[~, uniqueIndices] = unique(folderValues, 'stable');  % Get indices of unique values
subDirs = subDirs(uniqueIndices);  % Update subDirs with unique values
subDirs = {subDirs.folder};

% Loop through each subdirectory
for i = 1:numel(subDirs)
    subDirPath = subDirs(i);

    pname = char(subDirPath);

    % which files to be imported
    fname = {...
      [pname '/PF_100.xrdml'],...
      [pname '/PF_002.xrdml'],...
      [pname '/PF_101.xrdml'],...
      [pname '/PF_102.xrdml'],...
      [pname '/PF_110.xrdml'],...
      [pname '/PF_103.xrdml'],...
      };

    % background
    fname_bg = {...
      [pname '/UG_100.xrdml'],...
      [pname '/UG_002.xrdml'],...
      [pname '/UG_101.xrdml'],...
      [pname '/UG_102.xrdml'],...
      [pname '/UG_110.xrdml'],...
      [pname '/UG_103.xrdml'],...
      };


    % Create Pole Figure and correct data
    pf = loadPoleFigure(fname,h,CS,SS,'interface','xrdml');
    pf_bg = loadPoleFigure(fname_bg,h,CS,SS,'interface','xrdml');
    pf_cor = correct(pf,'bg',pf_bg);

    % Calculate ODF
    odf_cor = calcODF(pf_cor, 'resolution', 5*degree, 'ghost_correction');

    % Rotate for Rods (use your existing rotation code)
    odf_rot_z = rotate(odf_cor,q_z);

    % Export ODF to odf.txt in the same XRD folder
    odfPath = fullfile(pname, 'odf.txt');
    export(odf_rot_z, odfPath, 'Bunge');
end
