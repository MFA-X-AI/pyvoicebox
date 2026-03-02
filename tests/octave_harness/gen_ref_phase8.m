% Generate reference data for Phase 8 Speech Analysis & Enhancement functions
% Usage: octave --no-gui --silent gen_ref_phase8.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 8 reference data...\n');

%% v_phon2sone / v_sone2phon
p_test = [10 20 30 40 50 60 70 80];
s_phon = v_phon2sone(p_test);
p_back = v_sone2phon(s_phon);
save('-mat', fullfile(outdir, 'ref_phon2sone.mat'), 'p_test', 's_phon', 'p_back');
fprintf('  v_phon2sone / v_sone2phon done\n');

%% v_pesq2mos / v_mos2pesq
pesq_test = [-0.5 0 1 2 3 4 4.5];
mos_vals = v_pesq2mos(pesq_test);
pesq_back = v_mos2pesq(mos_vals);
save('-mat', fullfile(outdir, 'ref_pesq2mos.mat'), 'pesq_test', 'mos_vals', 'pesq_back');
fprintf('  v_pesq2mos / v_mos2pesq done\n');

%% v_stoi2prob
stoi_test = [0 0.2 0.4 0.6 0.8 1.0];
prob_ieee = v_stoi2prob(stoi_test, 'i');
prob_dant = v_stoi2prob(stoi_test, 'd');
save('-mat', fullfile(outdir, 'ref_stoi2prob.mat'), 'stoi_test', 'prob_ieee', 'prob_dant');
fprintf('  v_stoi2prob done\n');

%% v_glotros
t_gr = (0:99)'/100;
gr0 = v_glotros(0, t_gr);
gr1 = v_glotros(1, t_gr);
gr2 = v_glotros(2, t_gr);
gr0b = v_glotros(0, t_gr, [0.5 0.3]);
save('-mat', fullfile(outdir, 'ref_glotros.mat'), 't_gr', 'gr0', 'gr1', 'gr2', 'gr0b');
fprintf('  v_glotros done\n');

%% v_glotlf
t_gl = (0:99)'/100;
[gl0, q_gl] = v_glotlf(0, t_gl);
[gl1, ~] = v_glotlf(1, t_gl);
[gl2, ~] = v_glotlf(2, t_gl);
q_Up = q_gl.Up;
q_Ee = q_gl.Ee;
q_alpha = q_gl.alpha;
q_epsilon = q_gl.epsilon;
q_omega = q_gl.omega;
q_ti = q_gl.ti;
q_tp = q_gl.tp;
q_te = q_gl.te;
q_ta = q_gl.ta;
save('-mat', fullfile(outdir, 'ref_glotlf.mat'), 't_gl', 'gl0', 'gl1', 'gl2', ...
     'q_Up', 'q_Ee', 'q_alpha', 'q_epsilon', 'q_omega', 'q_ti', 'q_tp', 'q_te', 'q_ta');
fprintf('  v_glotlf done\n');

%% v_cep2pow / v_pow2cep
rng(42);
u_cp = [1 0.5 -0.3 0.1];
v_cp = diag([0.2 0.1 0.05 0.02]);
[m_cp, c_cp] = v_cep2pow(u_cp, v_cp, 'i');
[u_back, v_back] = v_pow2cep(m_cp, c_cp, 'i');
save('-mat', fullfile(outdir, 'ref_cep2pow.mat'), 'u_cp', 'v_cp', 'm_cp', 'c_cp', 'u_back', 'v_back');
fprintf('  v_cep2pow / v_pow2cep done\n');

%% v_importsii
f_sii = [200 500 1000 2000 3000 5000 8000];
q_sii = v_importsii(f_sii);
q_sii_c = v_importsii(f_sii, 'c');
f_sii_d = [200 500 1000 2000 3000 5000 8000];
q_sii_d = v_importsii(f_sii_d, 'd');
save('-mat', fullfile(outdir, 'ref_importsii.mat'), 'f_sii', 'q_sii', 'q_sii_c', 'f_sii_d', 'q_sii_d');
fprintf('  v_importsii done\n');

%% v_ldatrace
rng(42);
b_lda = [3 1; 1 2];
w_lda = [2 0.5; 0.5 1.5];
[a_lda, f_lda, B_lda, W_lda] = v_ldatrace(b_lda, w_lda, 2);
save('-mat', fullfile(outdir, 'ref_ldatrace.mat'), 'b_lda', 'w_lda', 'a_lda', 'f_lda', 'B_lda', 'W_lda');
fprintf('  v_ldatrace done\n');

%% v_melbankm
[x_mb, mc_mb, mn_mb, mx_mb] = v_melbankm(10, 256, 8000);
x_mb_full = full(x_mb);
save('-mat', fullfile(outdir, 'ref_melbankm.mat'), 'x_mb_full', 'mc_mb', 'mn_mb', 'mx_mb');
fprintf('  v_melbankm done\n');

%% v_melcepst - using synthetic signal
rng(42);
fs_mc = 8000;
t_mc = (0:fs_mc*0.1-1)/fs_mc;
s_mc = sin(2*pi*200*t_mc) + 0.5*sin(2*pi*400*t_mc) + 0.1*randn(size(t_mc));
[c_mc, tc_mc] = v_melcepst(s_mc, fs_mc, 'M', 12, 26, 256, 128);
save('-mat', fullfile(outdir, 'ref_melcepst.mat'), 's_mc', 'fs_mc', 'c_mc', 'tc_mc');
fprintf('  v_melcepst done\n');

%% v_estnoiseg - using synthetic noisy signal
rng(42);
fs_en = 8000;
ninc = round(0.016*fs_en);
ovf = 2;
nfft = ovf*ninc;
s_en = randn(fs_en*0.5, 1);  % 0.5 second of noise
f_en = v_rfft(v_enframe(s_en, v_windows(2,nfft,'l'), ninc), nfft, 2);
pspec_en = f_en.*conj(f_en);
x_eng = v_estnoiseg(pspec_en, ninc/fs_en);
save('-mat', fullfile(outdir, 'ref_estnoiseg.mat'), 'pspec_en', 'x_eng', 'ninc', 'fs_en');
fprintf('  v_estnoiseg done\n');

%% v_estnoisem
x_enm = v_estnoisem(pspec_en, ninc/fs_en);
save('-mat', fullfile(outdir, 'ref_estnoisem.mat'), 'pspec_en', 'x_enm');
fprintf('  v_estnoisem done\n');

%% v_snrseg - simple test
rng(42);
fs_snr = 8000;
r_snr = sin(2*pi*200*(0:fs_snr*0.1-1)/fs_snr)';
n_snr = 0.1*randn(length(r_snr),1);
s_snr = r_snr + n_snr;
[seg_snr, glo_snr] = v_snrseg(s_snr, r_snr, fs_snr, 'wz');
save('-mat', fullfile(outdir, 'ref_snrseg.mat'), 's_snr', 'r_snr', 'fs_snr', 'seg_snr', 'glo_snr');
fprintf('  v_snrseg done\n');

fprintf('Phase 8 reference data generation complete.\n');
