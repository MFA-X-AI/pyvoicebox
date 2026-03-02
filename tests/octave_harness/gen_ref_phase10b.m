% Generate reference data for Phase 10b functions (plotting/display, file readers)
% Usage: octave --no-gui --silent gen_ref_phase10b.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 10b reference data...\n');

%% v_lambda2rgb - 1931 observer
lambdas = [380 420 460 500 540 580 620 660 700 740]';
xyz_1931 = v_lambda2rgb(lambdas, 'x');
rgb_1931_signed = v_lambda2rgb(lambdas, 's');
rgb_1931_clipped = v_lambda2rgb(lambdas, 'r');

% Single wavelength tests
xyz_single = v_lambda2rgb(550, 'x');
rgb_single_s = v_lambda2rgb(550, 's');
rgb_single_r = v_lambda2rgb(550, 'r');

% 1964 observer
xyz_1964 = v_lambda2rgb(lambdas, 'X');
rgb_1964_signed = v_lambda2rgb(lambdas, 'S');
rgb_1964_clipped = v_lambda2rgb(lambdas, 'R');

save('-mat', fullfile(outdir, 'ref_lambda2rgb.mat'), ...
    'lambdas', 'xyz_1931', 'rgb_1931_signed', 'rgb_1931_clipped', ...
    'xyz_single', 'rgb_single_s', 'rgb_single_r', ...
    'xyz_1964', 'rgb_1964_signed', 'rgb_1964_clipped');
fprintf('  v_lambda2rgb done\n');

%% v_colormap - test with custom named maps
% v_thermliny
rgb_therm = v_colormap('v_thermliny');
[rgb_therm64, y_therm64, l_therm64] = v_colormap('v_thermliny');

% v_bipliny
rgb_bip = v_colormap('v_bipliny');
[rgb_bip64, y_bip64, l_bip64] = v_colormap('v_bipliny');

% v_bipveey
rgb_bipv = v_colormap('v_bipveey');

% v_circrby
rgb_circ = v_colormap('v_circrby');

% Test basic operations: numeric map with y-linearization
test_map = [0 0 0; 0.2 0 0.5; 0.5 0.5 0; 0.8 0.8 0.2; 1 1 1];
[rgb_ylin, y_ylin, l_ylin] = v_colormap(test_map, 'y', 10);

% Test flip
[rgb_flip, y_flip, l_flip] = v_colormap(test_map, 'f');

% Test interpolation (no luminance linearization)
[rgb_interp, y_interp, l_interp] = v_colormap(test_map, '', 20);

save('-mat', fullfile(outdir, 'ref_colormap.mat'), ...
    'rgb_therm64', 'y_therm64', 'l_therm64', ...
    'rgb_bip64', 'y_bip64', 'l_bip64', ...
    'rgb_bipv', 'rgb_circ', ...
    'test_map', 'rgb_ylin', 'y_ylin', 'l_ylin', ...
    'rgb_flip', 'y_flip', 'l_flip', ...
    'rgb_interp', 'y_interp', 'l_interp');
fprintf('  v_colormap done\n');

fprintf('Phase 10b reference data generation complete.\n');
