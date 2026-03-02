% Generate reference data for Phase 4 Signal Processing Primitives
% Usage: octave --no-gui --silent gen_ref_phase4.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 4 reference data...\n');

%% v_enframe
rng(42);
x_enf = randn(1,200);

% basic framing: length 20, hop 10
[f1, t1, w1] = v_enframe(x_enf, 20, 10);

% with hamming window
win_ham = hamming(32, 'periodic')';
[f2, t2, w2] = v_enframe(x_enf, win_ham, 8);

% zero-pad mode
[f3, t3, w3] = v_enframe(x_enf, 30, 15, 'z');

% reflect mode
[f4, t4, w4] = v_enframe(x_enf, 30, 15, 'r');

% power spectrum mode
[f5, t5, w5] = v_enframe(x_enf, 32, 8, 'ps');

% DFT mode
[f6, t6, w6] = v_enframe(x_enf, 32, 8, 'f');

% overlap-add scaling
[f7, t7, w7] = v_enframe(x_enf, win_ham, 8, 'a');

% hop as fraction
[f8, t8, w8] = v_enframe(x_enf, 40, 0.25);

save('-mat', fullfile(outdir, 'ref_enframe.mat'), ...
     'x_enf', 'f1', 't1', 'w1', 'f2', 't2', 'w2', ...
     'f3', 't3', 'w3', 'f4', 't4', 'w4', ...
     'f5', 't5', 'w5', 'f6', 't6', 'w6', ...
     'f7', 't7', 'w7', 'f8', 't8', 'w8', 'win_ham');
fprintf('  v_enframe done\n');

%% v_overlapadd
rng(42);
frames_ola = randn(10, 32);
win_ola = hamming(32, 'periodic')';
x_ola1 = v_overlapadd(frames_ola, win_ola, 8);
x_ola2 = v_overlapadd(frames_ola, win_ola, 16);
x_ola3 = v_overlapadd(frames_ola);  % no window, no inc

save('-mat', fullfile(outdir, 'ref_overlapadd.mat'), ...
     'frames_ola', 'win_ola', 'x_ola1', 'x_ola2', 'x_ola3');
fprintf('  v_overlapadd done\n');

%% v_fram2wav
rng(42);
% Simple case: 5 contiguous frames
x_f2w = [1.0; 2.0; 3.0; 2.5; 1.5];
tt_f2w = [1 10; 11 20; 21 30; 31 40; 41 50];
[w_f2w_z, s_f2w_z] = v_fram2wav(x_f2w, tt_f2w, 'z');
[w_f2w_l, s_f2w_l] = v_fram2wav(x_f2w, tt_f2w, 'l');

save('-mat', fullfile(outdir, 'ref_fram2wav.mat'), ...
     'x_f2w', 'tt_f2w', 'w_f2w_z', 's_f2w_z', 'w_f2w_l', 's_f2w_l');
fprintf('  v_fram2wav done\n');

%% v_ditherq
% Since ditherq uses random numbers, we test the 'n' (no dither) mode
x_dith = [0.3 0.7 1.2 1.8 2.5 -0.3 -0.7 -1.2];
y_dith_n = v_ditherq(x_dith, 'n');

save('-mat', fullfile(outdir, 'ref_ditherq.mat'), ...
     'x_dith', 'y_dith_n');
fprintf('  v_ditherq done\n');

%% v_findpeaks
y_fp = [1 3 2 4 3 5 4 3 2 4 5 3 1 2 4 3 2];

% basic peaks
[k_fp1, v_fp1] = v_findpeaks(y_fp);

% valleys
[k_fp2, v_fp2] = v_findpeaks(y_fp, 'v');

% quadratic interpolation
[k_fp3, v_fp3] = v_findpeaks(y_fp, 'q');

% with first and last
[k_fp4, v_fp4] = v_findpeaks(y_fp, 'fl');

% maximum only
[k_fp5, v_fp5] = v_findpeaks(y_fp, 'm');

% with width tolerance
[k_fp6, v_fp6] = v_findpeaks(y_fp, '', 3);

% with x-axis
x_fp = linspace(0, 10, length(y_fp));
[k_fp7, v_fp7] = v_findpeaks(y_fp, '', [], x_fp);

% quadratic with x-axis
[k_fp8, v_fp8] = v_findpeaks(y_fp, 'q', [], x_fp);

% plateau test
y_plat = [1 2 3 3 3 2 1];
[k_plat, v_plat] = v_findpeaks(y_plat);

save('-mat', fullfile(outdir, 'ref_findpeaks.mat'), ...
     'y_fp', 'k_fp1', 'v_fp1', 'k_fp2', 'v_fp2', ...
     'k_fp3', 'v_fp3', 'k_fp4', 'v_fp4', ...
     'k_fp5', 'v_fp5', 'k_fp6', 'v_fp6', ...
     'x_fp', 'k_fp7', 'v_fp7', 'k_fp8', 'v_fp8', ...
     'y_plat', 'k_plat', 'v_plat');
fprintf('  v_findpeaks done\n');

%% v_maxfilt
rng(42);
x_mf = randn(1, 20);

% basic max filter (default: f=1, n=Inf)
[y_mf1, k_mf1] = v_maxfilt(x_mf);

% with window length
[y_mf2, k_mf2] = v_maxfilt(x_mf, 1, 5);

% with forgetting factor
[y_mf3, k_mf3] = v_maxfilt(x_mf, 0.95, 10);

% 2D matrix along dim 1
x_mf2d = randn(8, 3);
[y_mf2d, k_mf2d] = v_maxfilt(x_mf2d, 1, 4);

save('-mat', fullfile(outdir, 'ref_maxfilt.mat'), ...
     'x_mf', 'y_mf1', 'k_mf1', 'y_mf2', 'k_mf2', ...
     'y_mf3', 'k_mf3', 'x_mf2d', 'y_mf2d', 'k_mf2d');
fprintf('  v_maxfilt done\n');

%% v_teager
x_teag = sin(2*pi*(1:100)/20);
y_teag1 = v_teager(x_teag);
y_teag2 = v_teager(x_teag, 2, 'x');

% 2D matrix
x_teag2d = [sin(2*pi*(1:50)/10); cos(2*pi*(1:50)/10)]';
y_teag2d = v_teager(x_teag2d);

save('-mat', fullfile(outdir, 'ref_teager.mat'), ...
     'x_teag', 'y_teag1', 'y_teag2', 'x_teag2d', 'y_teag2d');
fprintf('  v_teager done\n');

%% v_zerocros
y_zc = sin(2*pi*(0:99)/20);

% both crossings
[t_zc1, s_zc1] = v_zerocros(y_zc);

% positive only
[t_zc2, s_zc2] = v_zerocros(y_zc, 'p');

% negative only
[t_zc3, s_zc3] = v_zerocros(y_zc, 'n');

% rounded
[t_zc4, s_zc4] = v_zerocros(y_zc, 'br');

% with x-axis
x_zc = linspace(0, 5, 100);
[t_zc5, s_zc5] = v_zerocros(y_zc, 'b', x_zc);

save('-mat', fullfile(outdir, 'ref_zerocros.mat'), ...
     'y_zc', 't_zc1', 's_zc1', 't_zc2', 's_zc2', ...
     't_zc3', 's_zc3', 't_zc4', 's_zc4', ...
     'x_zc', 't_zc5', 's_zc5');
fprintf('  v_zerocros done\n');

%% v_schmitt
rng(42);
x_sc = cumsum(randn(1, 200));

% default hysteresis
y_sc1 = v_schmitt(x_sc);

% explicit thresholds
y_sc2 = v_schmitt(x_sc, [-2 2]);

% with transition output
[y_sc3, t_sc3] = v_schmitt(x_sc, 0.5);

% with minimum width
y_sc4 = v_schmitt(x_sc, 0.5, 5);

save('-mat', fullfile(outdir, 'ref_schmitt.mat'), ...
     'x_sc', 'y_sc1', 'y_sc2', 'y_sc3', 't_sc3', 'y_sc4');
fprintf('  v_schmitt done\n');

%% v_sigalign
rng(42);
r_sa = randn(1, 100);
s_sa = [zeros(1,10), 1.5*r_sa + 0.1*randn(1,100), zeros(1,10)];

% default mode
[d_sa1, g_sa1, rr_sa1, ss_sa1] = v_sigalign(s_sa, r_sa);

% unity gain
[d_sa2, g_sa2, rr_sa2, ss_sa2] = v_sigalign(s_sa, r_sa, [], 'us');

save('-mat', fullfile(outdir, 'ref_sigalign.mat'), ...
     'r_sa', 's_sa', 'd_sa1', 'g_sa1', 'rr_sa1', 'ss_sa1', ...
     'd_sa2', 'g_sa2', 'rr_sa2', 'ss_sa2');
fprintf('  v_sigalign done\n');

%% v_nearnonz
x_nn = [0 3 0 0 5 0 7 0 0 0];
[v_nn, y_nn, w_nn] = v_nearnonz(x_nn);

% 2D case
x_nn2d = [0 1 0; 3 0 0; 0 0 4; 0 2 0];
[v_nn2d, y_nn2d, w_nn2d] = v_nearnonz(x_nn2d);

save('-mat', fullfile(outdir, 'ref_nearnonz.mat'), ...
     'x_nn', 'v_nn', 'y_nn', 'w_nn', ...
     'x_nn2d', 'v_nn2d', 'y_nn2d', 'w_nn2d');
fprintf('  v_nearnonz done\n');

%% v_rangelim
x_rl = [1 5 3 8 2 9 4 7 6];

% explicit range [3 7]
y_rl1 = v_rangelim(x_rl, [3 7]);

% NaN mode
y_rl2 = v_rangelim(x_rl, [3 7], 'n');

% linear range, peak reference
y_rl3 = v_rangelim(x_rl, 5);

% linear range, trough reference
y_rl4 = v_rangelim(x_rl, 5, 'lt');

% ratio range
x_rl_pos = [1 2 3 4 5 6 7 8 9 10];
y_rl5 = v_rangelim(x_rl_pos, 3, 'r');

save('-mat', fullfile(outdir, 'ref_rangelim.mat'), ...
     'x_rl', 'y_rl1', 'y_rl2', 'y_rl3', 'y_rl4', ...
     'x_rl_pos', 'y_rl5');
fprintf('  v_rangelim done\n');

%% v_interval
x_iv = [1.6 2 3.8 0 6.5];
y_iv = [1 2 3 5 6];

[i_iv1, f_iv1] = v_interval(x_iv, y_iv);

% clip low
[i_iv2, f_iv2] = v_interval(x_iv, y_iv, 'c');

% NaN low
[i_iv3, f_iv3] = v_interval(x_iv, y_iv, 'n');

% zero low
[i_iv4, f_iv4] = v_interval(x_iv, y_iv, 'z');

% Clip high
[i_iv5, f_iv5] = v_interval(x_iv, y_iv, 'C');

% NaN high
[i_iv6, f_iv6] = v_interval(x_iv, y_iv, 'N');

% Zero high
[i_iv7, f_iv7] = v_interval(x_iv, y_iv, 'Z');

save('-mat', fullfile(outdir, 'ref_interval.mat'), ...
     'x_iv', 'y_iv', ...
     'i_iv1', 'f_iv1', 'i_iv2', 'f_iv2', ...
     'i_iv3', 'f_iv3', 'i_iv4', 'f_iv4', ...
     'i_iv5', 'f_iv5', 'i_iv6', 'f_iv6', ...
     'i_iv7', 'f_iv7');
fprintf('  v_interval done\n');

fprintf('Phase 4 reference data generation complete.\n');
