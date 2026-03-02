% Generate reference data for Phase 1 functions
% Usage: octave --no-gui --silent gen_ref_phase1.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 1 reference data...\n');

%% v_logsum
x1 = [1 2 3 4 5];
y1 = v_logsum(x1);

x2 = [1 2 3; 4 5 6];
y2_d1 = v_logsum(x2, 1);
y2_d2 = v_logsum(x2, 2);

x3 = [-1000 -1001 -1002];
y3 = v_logsum(x3);

x4 = [Inf 1 2];
y4 = v_logsum(x4);

x5 = [1 2 3; 4 5 6];
k5 = [0.5; 2.0];
y5 = v_logsum(x5, 1, k5);

save('-mat', fullfile(outdir, 'ref_logsum.mat'), ...
     'x1', 'y1', 'x2', 'y2_d1', 'y2_d2', 'x3', 'y3', 'x4', 'y4', 'x5', 'k5', 'y5');
fprintf('  v_logsum done\n');

%% v_gammalns
xg1 = [0.5, 1, 1.5, 2, 5, 10];
yg1 = v_gammalns(xg1);

xg2 = [-0.5, -1.5, -2.5];
[yg2, sg2] = v_gammalns(xg2);

xg3 = [-1, -2, -3, 0];
yg3 = v_gammalns(xg3);

save('-mat', fullfile(outdir, 'ref_gammalns.mat'), ...
     'xg1', 'yg1', 'xg2', 'yg2', 'sg2', 'xg3', 'yg3');
fprintf('  v_gammalns done\n');

%% v_entropy
pe1 = [0.25 0.25 0.25 0.25];
he1 = v_entropy(pe1);

pe2 = [0.5 0.5];
he2 = v_entropy(pe2);

pe3 = [1 0 0 0];
he3 = v_entropy(pe3);

pe4 = [0.1 0.2 0.3 0.4];
he4 = v_entropy(pe4);

save('-mat', fullfile(outdir, 'ref_entropy.mat'), ...
     'pe1', 'he1', 'pe2', 'he2', 'pe3', 'he3', 'pe4', 'he4');
fprintf('  v_entropy done\n');

%% v_bitsprec
xb = [2.5 1.5 1.1 1.0 0.9 0.5 0.1 -0.1 -0.5 -0.9 -1.5];
yb_fne = v_bitsprec(xb, 0, 'fne');
yb_fno = v_bitsprec(xb, 0, 'fno');
yb_fp = v_bitsprec(xb, 0, 'fp-');
yb_fm = v_bitsprec(xb, 0, 'fm-');
yb_fz = v_bitsprec(xb, 0, 'fz-');
yb_s3 = v_bitsprec(3.14159, 3, 'sne');
yb_s10 = v_bitsprec(3.14159, 10, 'sne');

save('-mat', fullfile(outdir, 'ref_bitsprec.mat'), ...
     'xb', 'yb_fne', 'yb_fno', 'yb_fp', 'yb_fm', 'yb_fz', 'yb_s3', 'yb_s10');
fprintf('  v_bitsprec done\n');

%% v_zerotrim
xz1 = [1 2 0; 0 3 0; 0 0 0];
yz1 = v_zerotrim(xz1);

xz2 = [0 0; 0 0];
yz2 = v_zerotrim(xz2);

xz3 = [1 0 0 0; 0 2 0 0];
yz3 = v_zerotrim(xz3);

save('-mat', fullfile(outdir, 'ref_zerotrim.mat'), ...
     'xz1', 'yz1', 'xz3', 'yz3');
fprintf('  v_zerotrim done\n');

%% v_choosenk
xc1 = v_choosenk(5, 3);
xc2 = v_choosenk(4, 2);
xc3 = v_choosenk(4, 4);

save('-mat', fullfile(outdir, 'ref_choosenk.mat'), 'xc1', 'xc2', 'xc3');
fprintf('  v_choosenk done\n');

%% v_choosrnk
xcr1 = v_choosrnk(3, 2);
xcr2 = v_choosrnk(2, 3);

save('-mat', fullfile(outdir, 'ref_choosrnk.mat'), 'xcr1', 'xcr2');
fprintf('  v_choosrnk done\n');

%% v_permutes
[xp3, sp3] = v_permutes(3);
[xp4, sp4] = v_permutes(4);

save('-mat', fullfile(outdir, 'ref_permutes.mat'), 'xp3', 'sp3', 'xp4', 'sp4');
fprintf('  v_permutes done\n');

%% v_sort
[bs1, is1, js1] = v_sort([3 1 4 1 5 9 2 6]);
[bs2, is2, js2] = v_sort([3 1; 4 2; 1 5]);

save('-mat', fullfile(outdir, 'ref_sort.mat'), 'bs1', 'is1', 'js1', 'bs2', 'is2', 'js2');
fprintf('  v_sort done\n');

%% v_windows
wh = v_windows('hamming', 64);
wn = v_windows('hanning', 64);
wr = v_windows('rectangle', 32);
wk = v_windows('kaiser', 64, 'uw', 8);
wg = v_windows('gaussian', 64, 'uw', 3);
wt = v_windows('triangle', 64);
wbl = v_windows('blackman', 64);
wh4 = v_windows('harris4', 128);
wtu = v_windows('tukey', 64, 'uw', 0.5);
wv = v_windows('vorbis', 64);

save('-mat', fullfile(outdir, 'ref_windows.mat'), ...
     'wh', 'wn', 'wr', 'wk', 'wg', 'wt', 'wbl', 'wh4', 'wtu', 'wv');
fprintf('  v_windows done\n');

%% v_huffman
ph = [0.4 0.3 0.2 0.1];
[cch, llh, lh] = v_huffman(ph);

save('-mat', fullfile(outdir, 'ref_huffman.mat'), 'ph', 'llh', 'lh');
fprintf('  v_huffman done\n');

fprintf('Phase 1 reference data generation complete.\n');
