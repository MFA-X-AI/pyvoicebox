% Generate reference data for Phase 7 Gaussian Mixture & Probability functions
% Usage: octave --no-gui --silent gen_ref_phase7.m <voicebox_dir> <output_dir> [stubs_dir]

args = argv();
addpath(args{1});
outdir = args{2};
if length(args) >= 3
    addpath(args{3});  % normcdf/normpdf stubs for Octave
else
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'tmp'));
end
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 7 reference data...\n');

%% v_berk2prob and v_prob2berk
b_test = [-3 -1 0 1 3];
[p_berk, d_berk] = v_berk2prob(b_test);
[b_back, d_back] = v_prob2berk(p_berk);
save('-mat', fullfile(outdir, 'ref_berk.mat'), 'b_test', 'p_berk', 'd_berk', 'b_back', 'd_back');
fprintf('  v_berk2prob / v_prob2berk done\n');

%% v_normcdflog
x_ncl = [-30 -10 -5 -1 0 1 5];
p_ncl = v_normcdflog(x_ncl);
p_ncl2 = v_normcdflog(x_ncl, 2, 3);
save('-mat', fullfile(outdir, 'ref_normcdflog.mat'), 'x_ncl', 'p_ncl', 'p_ncl2');
fprintf('  v_normcdflog done\n');

%% v_vonmisespdf
x_vm = linspace(-pi, pi, 20);
p_vm = v_vonmisespdf(x_vm, 0, 2);
p_vm2 = v_vonmisespdf(x_vm, 1, 5);
save('-mat', fullfile(outdir, 'ref_vonmises.mat'), 'x_vm', 'p_vm', 'p_vm2');
fprintf('  v_vonmisespdf done\n');

%% v_besselratio
x_br = [0 0.5 1 2 5 10 50];
y_br = v_besselratio(x_br, 0, 10);
y_br1 = v_besselratio(x_br, 1, 10);
save('-mat', fullfile(outdir, 'ref_besselratio.mat'), 'x_br', 'y_br', 'y_br1');
fprintf('  v_besselratio done\n');

%% v_besselratioi
r_bri = [0.1 0.3 0.5 0.7 0.9];
s_bri = v_besselratioi(r_bri, 0, 10);
save('-mat', fullfile(outdir, 'ref_besselratioi.mat'), 'r_bri', 's_bri');
fprintf('  v_besselratioi done\n');

%% v_besratinv0
r_bi0 = [0.1 0.3 0.5 0.7 0.85 0.9 0.95];
k_bi0 = v_besratinv0(r_bi0);
save('-mat', fullfile(outdir, 'ref_besratinv0.mat'), 'r_bi0', 'k_bi0');
fprintf('  v_besratinv0 done\n');

%% v_chimv
[m_chi1, v_chi1] = v_chimv(1, 0, 1);
[m_chi2, v_chi2] = v_chimv(3, 2, 1);
[m_chi3, v_chi3] = v_chimv(5, [0 1 3], 2);
save('-mat', fullfile(outdir, 'ref_chimv.mat'), 'm_chi1', 'v_chi1', 'm_chi2', 'v_chi2', 'm_chi3', 'v_chi3');
fprintf('  v_chimv done\n');

%% v_maxgauss
m_mg = [0; 1; 2];
c_mg = [1 0.2 0; 0.2 2 0.1; 0 0.1 1.5];
[u_mg, v_mg, p_mg] = v_maxgauss(m_mg, c_mg);
save('-mat', fullfile(outdir, 'ref_maxgauss.mat'), 'm_mg', 'c_mg', 'u_mg', 'v_mg', 'p_mg');
fprintf('  v_maxgauss done\n');

%% v_gausprod - scalar covariance
m_gp1 = [0 2; 1 3];  % 2D, 2 gaussians
c_gp1 = [1; 2];  % scalar * I
[g_gp1, u_gp1, k_gp1] = v_gausprod(m_gp1, c_gp1);
save('-mat', fullfile(outdir, 'ref_gausprod.mat'), 'm_gp1', 'c_gp1', 'g_gp1', 'u_gp1', 'k_gp1');
fprintf('  v_gausprod done\n');

%% v_gaussmixp - diagonal covariance
rng(42);
m_gmp = [0 0; 3 3; -2 1];
v_gmp = [1 0.5; 0.5 2; 1.5 1];
w_gmp = [0.3; 0.5; 0.2];
y_gmp = randn(10, 2);
[lp_gmp, rp_gmp, kh_gmp, kp_gmp] = v_gaussmixp(y_gmp, m_gmp, v_gmp, w_gmp);
save('-mat', fullfile(outdir, 'ref_gaussmixp.mat'), ...
     'y_gmp', 'm_gmp', 'v_gmp', 'w_gmp', 'lp_gmp', 'rp_gmp', 'kh_gmp', 'kp_gmp');
fprintf('  v_gaussmixp (diag) done\n');

%% v_gaussmixp - full covariance
m_gmpf = [0 0; 3 3];
v_gmpf = zeros(2,2,2);
v_gmpf(:,:,1) = [1 0.3; 0.3 0.5];
v_gmpf(:,:,2) = [2 -0.1; -0.1 1];
w_gmpf = [0.6; 0.4];
y_gmpf = [0 0; 1 1; 3 3; -1 0];
[lp_gmpf, rp_gmpf, kh_gmpf, kp_gmpf] = v_gaussmixp(y_gmpf, m_gmpf, v_gmpf, w_gmpf);
save('-mat', fullfile(outdir, 'ref_gaussmixp_full.mat'), ...
     'y_gmpf', 'm_gmpf', 'v_gmpf', 'w_gmpf', 'lp_gmpf', 'rp_gmpf', 'kh_gmpf', 'kp_gmpf');
fprintf('  v_gaussmixp (full) done\n');

%% v_gaussmixk - KL divergence
m_kl1 = [0 0; 2 2];
v_kl1 = [1 1; 1 1];
w_kl1 = [0.5; 0.5];
m_kl2 = [1 1; 3 3];
v_kl2 = [1.5 1.5; 0.5 0.5];
w_kl2 = [0.4; 0.6];
[d_kl, klfg_kl] = v_gaussmixk(m_kl1, v_kl1, w_kl1, m_kl2, v_kl2, w_kl2);
[d_kl_self, klfg_self] = v_gaussmixk(m_kl1, v_kl1, w_kl1);
save('-mat', fullfile(outdir, 'ref_gaussmixk.mat'), ...
     'm_kl1', 'v_kl1', 'w_kl1', 'm_kl2', 'v_kl2', 'w_kl2', ...
     'd_kl', 'klfg_kl', 'd_kl_self', 'klfg_self');
fprintf('  v_gaussmixk done\n');

%% v_gaussmixg - GMM mean, variance, mode
m_gg = [0 0; 5 0];
v_gg = [1 1; 1 1];
w_gg = [0.5; 0.5];
[mg_gg, vg_gg, pg_gg, pv_gg] = v_gaussmixg(m_gg, v_gg, w_gg, 2);
save('-mat', fullfile(outdir, 'ref_gaussmixg.mat'), ...
     'm_gg', 'v_gg', 'w_gg', 'mg_gg', 'vg_gg', 'pg_gg', 'pv_gg');
fprintf('  v_gaussmixg done\n');

%% v_gaussmixt - multiply two GMMs
m_t1 = [0; 2];
v_t1 = [1; 0.5];
w_t1 = [0.6; 0.4];
m_t2 = [1; 3];
v_t2 = [0.8; 1.2];
w_t2 = [0.5; 0.5];
[m_tp, v_tp, w_tp] = v_gaussmixt(m_t1, v_t1, w_t1, m_t2, v_t2, w_t2);
save('-mat', fullfile(outdir, 'ref_gaussmixt.mat'), ...
     'm_t1', 'v_t1', 'w_t1', 'm_t2', 'v_t2', 'w_t2', 'm_tp', 'v_tp', 'w_tp');
fprintf('  v_gaussmixt done\n');

%% v_gaussmixm - magnitude of GMM
m_gm = [1 2; -1 3];
v_gm = [0.5 1; 1 0.5];
w_gm = [0.6; 0.4];
z_gm = [0 0];
[mm_gm, mc_gm] = v_gaussmixm(m_gm, v_gm, w_gm, z_gm);
save('-mat', fullfile(outdir, 'ref_gaussmixm.mat'), ...
     'm_gm', 'v_gm', 'w_gm', 'z_gm', 'mm_gm', 'mc_gm');
fprintf('  v_gaussmixm done\n');

%% v_histndim
rng(42);
x_hist = randn(1000, 2);
b_hist = [10; -3; 3];
[v_hist, t_hist] = v_histndim(x_hist, b_hist);
save('-mat', fullfile(outdir, 'ref_histndim.mat'), 'x_hist', 'b_hist', 'v_hist');
fprintf('  v_histndim done\n');

%% v_gaussmixb - Bhattacharyya divergence
m_b1 = [0; 3];
v_b1 = [1; 1];
w_b1 = [0.5; 0.5];
m_b2 = [1; 4];
v_b2 = [1.5; 0.5];
w_b2 = [0.5; 0.5];
d_b = v_gaussmixb(m_b1, v_b1, w_b1, m_b2, v_b2, w_b2, 0);
[d_bself, dbfg_self] = v_gaussmixb(m_b1, v_b1, w_b1);
save('-mat', fullfile(outdir, 'ref_gaussmixb.mat'), ...
     'm_b1', 'v_b1', 'w_b1', 'm_b2', 'v_b2', 'w_b2', 'd_b', 'd_bself', 'dbfg_self');
fprintf('  v_gaussmixb done\n');

fprintf('Phase 7 reference data generation complete.\n');
