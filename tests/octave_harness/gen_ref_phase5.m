% Generate reference data for Phase 5 Audio I/O & Codec functions
% Usage: octave --no-gui --silent gen_ref_phase5.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 5 reference data...\n');

%% v_lin2pcmu / v_pcmu2lin
% Test with all 256 possible PCM-u values (roundtrip)
pcmu_all = 0:255;
lin_from_pcmu_default = v_pcmu2lin(pcmu_all);
lin_from_pcmu_s1 = v_pcmu2lin(pcmu_all, 1);
lin_from_pcmu_s8031 = v_pcmu2lin(pcmu_all, 8031);

% Encode some test signals
x_pcmu_test = [-8159 -4000 -1000 -100 -10 -1 0 1 10 100 1000 4000 8159];
pcmu_from_lin_s1 = v_lin2pcmu(x_pcmu_test, 1);
pcmu_from_lin_default = v_lin2pcmu(x_pcmu_test / 4004.189931);

% Roundtrip: encode then decode
rng(42);
x_pcmu_rt = randn(1, 100) * 2000;
pcmu_encoded = v_lin2pcmu(x_pcmu_rt, 1);
pcmu_decoded = v_pcmu2lin(pcmu_encoded, 1);

% Test with sine wave (ITU standard reference)
pcmu_sine = [158 139 139 158 30 11 11 30];
lin_sine_pcmu = v_pcmu2lin(pcmu_sine);

save('-mat', fullfile(outdir, 'ref_lin2pcmu.mat'), ...
     'pcmu_all', 'lin_from_pcmu_default', 'lin_from_pcmu_s1', 'lin_from_pcmu_s8031', ...
     'x_pcmu_test', 'pcmu_from_lin_s1', 'pcmu_from_lin_default', ...
     'x_pcmu_rt', 'pcmu_encoded', 'pcmu_decoded', ...
     'pcmu_sine', 'lin_sine_pcmu');
fprintf('  v_lin2pcmu / v_pcmu2lin done\n');

%% v_lin2pcma / v_pcma2lin
% Test with all 256 possible PCM-a values (roundtrip)
pcma_all = 0:255;
lin_from_pcma_default = v_pcma2lin(pcma_all);
lin_from_pcma_s1 = v_pcma2lin(pcma_all, 85, 1);
lin_from_pcma_s4032 = v_pcma2lin(pcma_all, 85, 4032);

% Encode some test signals
x_pcma_test = [-4096 -2000 -500 -50 -5 -1 0 1 5 50 500 2000 4096];
pcma_from_lin_s1 = v_lin2pcma(x_pcma_test, 85, 1);
pcma_from_lin_default = v_lin2pcma(x_pcma_test / 2017.396342);

% Decode with no XOR mask (m=0)
lin_from_pcma_nomask = v_pcma2lin(pcma_all, 0, 1);

% Encode with no XOR mask (m=0)
pcma_nomask = v_lin2pcma(x_pcma_test, 0, 1);

% Roundtrip: encode then decode
rng(42);
x_pcma_rt = randn(1, 100) * 1000;
pcma_encoded = v_lin2pcma(x_pcma_rt, 85, 1);
pcma_decoded = v_pcma2lin(pcma_encoded, 85, 1);

% Test with sine wave (ITU standard reference)
pcma_sine = [225 244 244 225 97 116 116 97];
lin_sine_pcma = v_pcma2lin(pcma_sine);

save('-mat', fullfile(outdir, 'ref_lin2pcma.mat'), ...
     'pcma_all', 'lin_from_pcma_default', 'lin_from_pcma_s1', 'lin_from_pcma_s4032', ...
     'x_pcma_test', 'pcma_from_lin_s1', 'pcma_from_lin_default', ...
     'lin_from_pcma_nomask', 'pcma_nomask', ...
     'x_pcma_rt', 'pcma_encoded', 'pcma_decoded', ...
     'pcma_sine', 'lin_sine_pcma');
fprintf('  v_lin2pcma / v_pcma2lin done\n');

%% v_readwav / v_writewav
% Create test WAV files
rng(42);
wav_data_mono = randn(100, 1);
wav_data_stereo = randn(100, 2);
wav_fs = 16000;

% Write mono 16-bit WAV
v_writewav(wav_data_mono, wav_fs, fullfile(outdir, 'test_mono16.wav'), 'sp16');
% Write stereo 16-bit WAV
v_writewav(wav_data_stereo, wav_fs, fullfile(outdir, 'test_stereo16.wav'), 'sp16');

% Read them back
[wav_read_mono, wav_fs_mono] = v_readwav(fullfile(outdir, 'test_mono16.wav'), 'p');
[wav_read_stereo, wav_fs_stereo] = v_readwav(fullfile(outdir, 'test_stereo16.wav'), 'p');

% Read in raw mode
[wav_read_raw, wav_fs_raw] = v_readwav(fullfile(outdir, 'test_mono16.wav'), 'r');

% Read with nmax and nskip
[wav_read_partial, wav_fs_partial] = v_readwav(fullfile(outdir, 'test_mono16.wav'), 'p', 20, 10);

save('-mat', fullfile(outdir, 'ref_wav.mat'), ...
     'wav_data_mono', 'wav_data_stereo', 'wav_fs', ...
     'wav_read_mono', 'wav_fs_mono', ...
     'wav_read_stereo', 'wav_fs_stereo', ...
     'wav_read_raw', 'wav_fs_raw', ...
     'wav_read_partial', 'wav_fs_partial');
fprintf('  v_readwav / v_writewav done\n');

%% v_readhtk / v_writehtk
% Create test HTK files with different types
rng(42);

% USER type (9) with float data
htk_user_data = randn(20, 13);
htk_user_fp = 0.01;
htk_user_tc = 9;
v_writehtk(fullfile(outdir, 'test_user.htk'), htk_user_data, htk_user_fp, htk_user_tc);
[htk_user_read, htk_user_fp_read, htk_user_dt_read, htk_user_tc_read] = v_readhtk(fullfile(outdir, 'test_user.htk'));

% MFCC type (6) with energy (6+64=70)
htk_mfcc_data = randn(30, 13);
htk_mfcc_fp = 0.01;
htk_mfcc_tc = 70;  % MFCC + _E
v_writehtk(fullfile(outdir, 'test_mfcc.htk'), htk_mfcc_data, htk_mfcc_fp, htk_mfcc_tc);
[htk_mfcc_read, htk_mfcc_fp_read, htk_mfcc_dt_read, htk_mfcc_tc_read] = v_readhtk(fullfile(outdir, 'test_mfcc.htk'));

% WAVEFORM type (0) - 16-bit int data
htk_wave_data = round(randn(1, 200) * 1000);
htk_wave_fp = 1/16000;
htk_wave_tc = 0;
v_writehtk(fullfile(outdir, 'test_wave.htk'), htk_wave_data, htk_wave_fp, htk_wave_tc);
[htk_wave_read, htk_wave_fp_read, htk_wave_dt_read, htk_wave_tc_read] = v_readhtk(fullfile(outdir, 'test_wave.htk'));

% PLP type (11) with _D _A _Z modifiers (11+256+512+2048=2827)
htk_plp_data = randn(15, 39);
htk_plp_fp = 0.01;
htk_plp_tc = 2827;
v_writehtk(fullfile(outdir, 'test_plp.htk'), htk_plp_data, htk_plp_fp, htk_plp_tc);
[htk_plp_read, htk_plp_fp_read, htk_plp_dt_read, htk_plp_tc_read, htk_plp_t_read] = v_readhtk(fullfile(outdir, 'test_plp.htk'));

save('-mat', fullfile(outdir, 'ref_htk.mat'), ...
     'htk_user_data', 'htk_user_fp', 'htk_user_tc', ...
     'htk_user_read', 'htk_user_fp_read', 'htk_user_dt_read', 'htk_user_tc_read', ...
     'htk_mfcc_data', 'htk_mfcc_fp', 'htk_mfcc_tc', ...
     'htk_mfcc_read', 'htk_mfcc_fp_read', 'htk_mfcc_dt_read', 'htk_mfcc_tc_read', ...
     'htk_wave_data', 'htk_wave_fp', 'htk_wave_tc', ...
     'htk_wave_read', 'htk_wave_fp_read', 'htk_wave_dt_read', 'htk_wave_tc_read', ...
     'htk_plp_data', 'htk_plp_fp', 'htk_plp_tc', ...
     'htk_plp_read', 'htk_plp_fp_read', 'htk_plp_dt_read', 'htk_plp_tc_read', 'htk_plp_t_read');
fprintf('  v_readhtk / v_writehtk done\n');

fprintf('Phase 5 reference data generation complete.\n');
