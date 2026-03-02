% Generate reference data for Phase 3 FFT/Transform functions
% Usage: octave --no-gui --silent gen_ref_phase3.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 3 reference data...\n');

%% v_rfft
% 1-D vector, even length
x_rfft_1d_even = [1 2 3 4 5 6 7 8];
y_rfft_1d_even = v_rfft(x_rfft_1d_even);

% 1-D vector, odd length
x_rfft_1d_odd = [1 2 3 4 5 6 7];
y_rfft_1d_odd = v_rfft(x_rfft_1d_odd);

% 1-D with specified n (zero-pad)
y_rfft_1d_n16 = v_rfft(x_rfft_1d_even, 16);

% 1-D with specified n (truncate)
y_rfft_1d_n4 = v_rfft(x_rfft_1d_even, 4);

% 2-D matrix, default dim (columns)
x_rfft_2d = [1 5; 2 6; 3 7; 4 8; 5 9; 6 10];
y_rfft_2d = v_rfft(x_rfft_2d);

% 2-D matrix, dim=2 (along rows)
y_rfft_2d_d2 = v_rfft(x_rfft_2d, [], 2);

% scalar
y_rfft_scalar = v_rfft(42);

save('-mat', fullfile(outdir, 'ref_rfft.mat'), ...
     'x_rfft_1d_even', 'y_rfft_1d_even', ...
     'x_rfft_1d_odd', 'y_rfft_1d_odd', ...
     'y_rfft_1d_n16', 'y_rfft_1d_n4', ...
     'x_rfft_2d', 'y_rfft_2d', 'y_rfft_2d_d2', ...
     'y_rfft_scalar');
fprintf('  v_rfft done\n');

%% v_irfft
% 1-D even output
y_irfft_in_even = v_rfft([1 2 3 4 5 6 7 8]);
x_irfft_even = v_irfft(y_irfft_in_even, 8);

% 1-D odd output
y_irfft_in_odd = v_rfft([1 2 3 4 5 6 7], 7);
x_irfft_odd = v_irfft(y_irfft_in_odd, 7);

% Default n (2*M-2)
x_irfft_default = v_irfft(y_irfft_in_even);

% 2-D matrix
y_irfft_2d_in = v_rfft([1 5; 2 6; 3 7; 4 8; 5 9; 6 10]);
x_irfft_2d = v_irfft(y_irfft_2d_in, 6);

% roundtrip: rfft then irfft
x_roundtrip = [3 1 4 1 5 9 2 6];
y_roundtrip = v_rfft(x_roundtrip, 8);
x_roundtrip_back = v_irfft(y_roundtrip, 8);

save('-mat', fullfile(outdir, 'ref_irfft.mat'), ...
     'y_irfft_in_even', 'x_irfft_even', ...
     'y_irfft_in_odd', 'x_irfft_odd', ...
     'x_irfft_default', ...
     'y_irfft_2d_in', 'x_irfft_2d', ...
     'x_roundtrip', 'x_roundtrip_back');
fprintf('  v_irfft done\n');

%% v_rsfft
% 1-D default n
x_rsfft_1d = [1 2 3 4 5];
y_rsfft_1d = v_rsfft(x_rsfft_1d);

% 1-D with explicit n even
y_rsfft_1d_n8 = v_rsfft(x_rsfft_1d, 8);

% 1-D with explicit n odd
y_rsfft_1d_n9 = v_rsfft(x_rsfft_1d, 9);

% 2-D matrix (column-wise)
x_rsfft_2d = [1 4; 2 5; 3 6; 4 7; 5 8];
y_rsfft_2d = v_rsfft(x_rsfft_2d);

% self-inverse property: v_rsfft(v_rsfft(x, n), n)/n should return x
x_rsfft_si = [1 3 5 7 9];
n_rsfft_si = 8;
y_rsfft_si = v_rsfft(x_rsfft_si, n_rsfft_si);
x_rsfft_si_back = v_rsfft(y_rsfft_si, n_rsfft_si) / n_rsfft_si;

save('-mat', fullfile(outdir, 'ref_rsfft.mat'), ...
     'x_rsfft_1d', 'y_rsfft_1d', ...
     'y_rsfft_1d_n8', 'y_rsfft_1d_n9', ...
     'x_rsfft_2d', 'y_rsfft_2d', ...
     'x_rsfft_si', 'n_rsfft_si', 'y_rsfft_si', 'x_rsfft_si_back');
fprintf('  v_rsfft done\n');

%% v_rdct
% 1-D default
x_rdct_1d = [1 2 3 4 5 6 7 8];
y_rdct_1d = v_rdct(x_rdct_1d);

% 1-D with n
y_rdct_1d_n4 = v_rdct(x_rdct_1d, 4);
y_rdct_1d_n16 = v_rdct(x_rdct_1d, 16);

% 1-D with custom a,b
y_rdct_1d_ab = v_rdct(x_rdct_1d, 8, 1, 1);

% 2-D matrix
x_rdct_2d = [1 5; 2 6; 3 7; 4 8];
y_rdct_2d = v_rdct(x_rdct_2d);

% Odd-length input
x_rdct_odd = [1 2 3 4 5];
y_rdct_odd = v_rdct(x_rdct_odd);

save('-mat', fullfile(outdir, 'ref_rdct.mat'), ...
     'x_rdct_1d', 'y_rdct_1d', ...
     'y_rdct_1d_n4', 'y_rdct_1d_n16', ...
     'y_rdct_1d_ab', ...
     'x_rdct_2d', 'y_rdct_2d', ...
     'x_rdct_odd', 'y_rdct_odd');
fprintf('  v_rdct done\n');

%% v_irdct
% 1-D default
y_irdct_1d = v_rdct([1 2 3 4 5 6 7 8]);
x_irdct_1d = v_irdct(y_irdct_1d);

% roundtrip for even-length
x_irdct_rt_even = [3 1 4 1 5 9 2 6];
y_irdct_rt_even = v_rdct(x_irdct_rt_even, 8);
x_irdct_rt_even_back = v_irdct(y_irdct_rt_even, 8);

% roundtrip for odd-length
x_irdct_rt_odd = [3 1 4 1 5 9 2];
y_irdct_rt_odd = v_rdct(x_irdct_rt_odd, 7);
x_irdct_rt_odd_back = v_irdct(y_irdct_rt_odd, 7);

% with custom a, b
x_irdct_ab = [1 2 3 4];
y_irdct_ab = v_rdct(x_irdct_ab, 4, 1, 1);
x_irdct_ab_back = v_irdct(y_irdct_ab, 4, 1, 1);

% 2-D
x_irdct_2d = [1 5; 2 6; 3 7; 4 8];
y_irdct_2d = v_rdct(x_irdct_2d);
x_irdct_2d_back = v_irdct(y_irdct_2d);

save('-mat', fullfile(outdir, 'ref_irdct.mat'), ...
     'y_irdct_1d', 'x_irdct_1d', ...
     'x_irdct_rt_even', 'x_irdct_rt_even_back', ...
     'x_irdct_rt_odd', 'x_irdct_rt_odd_back', ...
     'x_irdct_ab', 'x_irdct_ab_back', ...
     'x_irdct_2d', 'x_irdct_2d_back');
fprintf('  v_irdct done\n');

%% v_rhartley
% 1-D default
x_hart_1d = [1 2 3 4 5 6 7 8];
y_hart_1d = v_rhartley(x_hart_1d);

% 1-D with n
y_hart_1d_n16 = v_rhartley(x_hart_1d, 16);
y_hart_1d_n4 = v_rhartley(x_hart_1d, 4);

% 2-D matrix
x_hart_2d = [1 5; 2 6; 3 7; 4 8];
y_hart_2d = v_rhartley(x_hart_2d);

% self-inverse: v_rhartley(v_rhartley(x,n),n)/n == x
y_hart_si = v_rhartley(x_hart_1d, 8);
x_hart_si_back = v_rhartley(y_hart_si, 8) / 8;

save('-mat', fullfile(outdir, 'ref_rhartley.mat'), ...
     'x_hart_1d', 'y_hart_1d', ...
     'y_hart_1d_n16', 'y_hart_1d_n4', ...
     'x_hart_2d', 'y_hart_2d', ...
     'x_hart_si_back');
fprintf('  v_rhartley done\n');

%% v_zoomfft
% Basic: equivalent to fft
x_zoom_1d = [1 2 3 4 5 6 7 8];
[y_zoom_fft, f_zoom_fft] = v_zoomfft(x_zoom_1d, 8, 8, 0);

% Zoom into a range
[y_zoom_range, f_zoom_range] = v_zoomfft(x_zoom_1d, 16, 4, 2);

% Non-integer n (chirp z-transform path)
[y_zoom_chirp, f_zoom_chirp] = v_zoomfft(x_zoom_1d, 10.5, 6, 1.5);

% 2-D matrix, default dim
x_zoom_2d = [1 5; 2 6; 3 7; 4 8];
[y_zoom_2d, f_zoom_2d] = v_zoomfft(x_zoom_2d, 4, 4, 0);

save('-mat', fullfile(outdir, 'ref_zoomfft.mat'), ...
     'x_zoom_1d', 'y_zoom_fft', 'f_zoom_fft', ...
     'y_zoom_range', 'f_zoom_range', ...
     'y_zoom_chirp', 'f_zoom_chirp', ...
     'x_zoom_2d', 'y_zoom_2d', 'f_zoom_2d');
fprintf('  v_zoomfft done\n');

%% v_convfft
% Basic convolution (equivalent to filter)
x_conv = [1 2 3 4 5 6 7 8];
h_conv = [1 0.5 0.25];
z_conv_filter = v_convfft(x_conv', h_conv);

% Full convolution (equivalent to conv)
z_conv_full = v_convfft(x_conv', h_conv, 1, '', 1, 1, length(x_conv)+length(h_conv)-1);

% Correlation mode
z_conv_xcorr = v_convfft(x_conv', h_conv, 1, 'x', 1, 1, length(x_conv));

save('-mat', fullfile(outdir, 'ref_convfft.mat'), ...
     'x_conv', 'h_conv', ...
     'z_conv_filter', 'z_conv_full', 'z_conv_xcorr');
fprintf('  v_convfft done\n');

%% v_frac2bin
% Basic integer
s_frac2bin_1 = v_frac2bin(5);

% Fractional
s_frac2bin_2 = v_frac2bin(5.75, 1, 4);

% Vector
s_frac2bin_3 = v_frac2bin([3; 5; 7], 4, 0);

% Negative n (leading spaces)
s_frac2bin_4 = v_frac2bin([1; 8], -1, 0);

% Truncation mode (m < 0)
s_frac2bin_5 = v_frac2bin(5.75, 1, -4);

save('-mat', fullfile(outdir, 'ref_frac2bin.mat'), ...
     's_frac2bin_1', 's_frac2bin_2', 's_frac2bin_3', ...
     's_frac2bin_4', 's_frac2bin_5');
fprintf('  v_frac2bin done\n');

fprintf('Phase 3 reference data generation complete.\n');
