% Generate reference data for Phase 10 functions
% Usage: octave --no-gui --silent gen_ref_phase10.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 10 reference data...\n');

%% v_modsym
x1 = [3.7, -1.2, 5.5, 0.25];
[z1, k1] = v_modsym(x1, -2*pi);
[z2, k2] = v_modsym(2.3, 1);
[z3, k3] = v_modsym([1.1 2.5 3.8], 2, 1);
save('-mat', fullfile(outdir, 'ref_modsym.mat'), 'x1', 'z1', 'k1', 'z2', 'k2', 'z3', 'k3');
fprintf('  v_modsym done\n');

%% v_soundspeed
[v1, d1, z1s] = v_soundspeed(20);
[v2, d2, z2s] = v_soundspeed(25, 1.0);
[v3, d3, z3s] = v_soundspeed(0);
save('-mat', fullfile(outdir, 'ref_soundspeed.mat'), 'v1', 'd1', 'z1s', 'v2', 'd2', 'z2s', 'v3', 'd3', 'z3s');
fprintf('  v_soundspeed done\n');

%% v_mintrace
x_mt = [5 3 1; 2 8 4; 7 6 9];
p_mt = v_mintrace(x_mt);
save('-mat', fullfile(outdir, 'ref_mintrace.mat'), 'x_mt', 'p_mt');
fprintf('  v_mintrace done\n');

%% Distance measures - generate AR coefficients
rng(42);
ar1 = [1 0.6 -0.3; 1 0.4 0.2; 1 -0.5 0.1];
ar2 = [1 0.3 -0.1; 1 0.7 0.3; 1 -0.2 -0.4];

% v_distchar
d_char_d = v_distchar(ar1, ar2, 'd');
d_char_x = v_distchar(ar1(1:2,:), ar2, 'x');
save('-mat', fullfile(outdir, 'ref_distchar.mat'), 'ar1', 'ar2', 'd_char_d', 'd_char_x');
fprintf('  v_distchar done\n');

% v_distisar
d_isar_d = v_distisar(ar1, ar2, 'd');
d_isar_x = v_distisar(ar1(1:2,:), ar2, 'x');
save('-mat', fullfile(outdir, 'ref_distisar.mat'), 'ar1', 'ar2', 'd_isar_d', 'd_isar_x');
fprintf('  v_distisar done\n');

% v_distitar
d_itar_d = v_distitar(ar1, ar2, 'd');
d_itar_x = v_distitar(ar1(1:2,:), ar2, 'x');
d_itar_e = v_distitar(ar1, ar2, 'e');
save('-mat', fullfile(outdir, 'ref_distitar.mat'), 'ar1', 'ar2', 'd_itar_d', 'd_itar_x', 'd_itar_e');
fprintf('  v_distitar done\n');

%% Power spectrum distance measures
pf1 = abs([1 2 3 4 5; 3 2 1 4 2; 2 5 3 1 4]) + 0.1;
pf2 = abs([2 1 4 3 2; 1 3 2 5 1; 4 2 1 3 5]) + 0.1;

% v_distchpf
d_chpf_d = v_distchpf(pf1, pf2, 'd');
d_chpf_x = v_distchpf(pf1(1:2,:), pf2, 'x');
save('-mat', fullfile(outdir, 'ref_distchpf.mat'), 'pf1', 'pf2', 'd_chpf_d', 'd_chpf_x');
fprintf('  v_distchpf done\n');

% v_distispf
d_ispf_d = v_distispf(pf1, pf2, 'd');
d_ispf_x = v_distispf(pf1(1:2,:), pf2, 'x');
save('-mat', fullfile(outdir, 'ref_distispf.mat'), 'pf1', 'pf2', 'd_ispf_d', 'd_ispf_x');
fprintf('  v_distispf done\n');

% v_distitpf
d_itpf_d = v_distitpf(pf1, pf2, 'd');
d_itpf_x = v_distitpf(pf1(1:2,:), pf2, 'x');
save('-mat', fullfile(outdir, 'ref_distitpf.mat'), 'pf1', 'pf2', 'd_itpf_d', 'd_itpf_x');
fprintf('  v_distitpf done\n');

%% v_qrdotmult
q1 = [1; 2; 3; 4];
q2 = [5; 6; 7; 8];
qm = v_qrdotmult(q1, q2);
% batch
q1b = [1; 2; 3; 4; 5; 6; 7; 8];
q2b = [8; 7; 6; 5; 4; 3; 2; 1];
qmb = v_qrdotmult(q1b, q2b);
save('-mat', fullfile(outdir, 'ref_qrdotmult.mat'), 'q1', 'q2', 'qm', 'q1b', 'q2b', 'qmb');
fprintf('  v_qrdotmult done\n');

%% v_qrdotdiv
qd1 = v_qrdotdiv(q1, q2);
qd_inv = v_qrdotdiv(q1);
save('-mat', fullfile(outdir, 'ref_qrdotdiv.mat'), 'q1', 'q2', 'qd1', 'qd_inv');
fprintf('  v_qrdotdiv done\n');

%% v_qrdivide
qr1 = [1; 0; 0; 0];
qr2 = [0; 1; 0; 0];
qdiv = v_qrdivide(qr1, qr2);
qdiv2 = v_qrdivide(q1, q2);
qinv = v_qrdivide(q1);
save('-mat', fullfile(outdir, 'ref_qrdivide.mat'), 'qr1', 'qr2', 'qdiv', 'q1', 'q2', 'qdiv2', 'qinv');
fprintf('  v_qrdivide done\n');

%% v_potsband
[b_pots, a_pots] = v_potsband(8000);
[b_pots16, a_pots16] = v_potsband(16000);
save('-mat', fullfile(outdir, 'ref_potsband.mat'), 'b_pots', 'a_pots', 'b_pots16', 'a_pots16');
fprintf('  v_potsband done\n');

%% v_ewgrpdel
x_ew = zeros(200,1);
x_ew(50) = 1;
x_ew(100) = 1;
x_ew(150) = 1;
[y_ew, m_ew] = v_ewgrpdel(x_ew, 21);
save('-mat', fullfile(outdir, 'ref_ewgrpdel.mat'), 'x_ew', 'y_ew', 'm_ew');
fprintf('  v_ewgrpdel done\n');

%% v_quadpeak - 1D
z_1d = [1 3 5 4 2];
[v_qp, x_qp, t_qp, m_qp] = v_quadpeak(z_1d);
save('-mat', fullfile(outdir, 'ref_quadpeak.mat'), 'z_1d', 'v_qp', 'x_qp', 't_qp', 'm_qp');
fprintf('  v_quadpeak done\n');

%% v_quadpeak - 2D
z_2d = [1 2 1; 3 5 3; 2 3 2];
[v_qp2, x_qp2, t_qp2, m_qp2] = v_quadpeak(z_2d);
save('-mat', fullfile(outdir, 'ref_quadpeak2d.mat'), 'z_2d', 'v_qp2', 'x_qp2', 't_qp2', 'm_qp2');
fprintf('  v_quadpeak 2D done\n');

%% v_hypergeom1f1
h1 = v_hypergeom1f1(1, 2, 0.5);
h2 = v_hypergeom1f1(0.5, 1.5, -3);
h3 = v_hypergeom1f1(2, 3, [0.1, 0.5, 1.0, 5.0, -5.0]);
save('-mat', fullfile(outdir, 'ref_hypergeom1f1.mat'), 'h1', 'h2', 'h3');
fprintf('  v_hypergeom1f1 done\n');

%% v_txalign
x_ta = [1 11 21 27 31 42 51];
y_ta = [2 12 15 22 25 32 41 52 61];
[kx_ta, ky_ta, nxy_ta, mxy_ta, sxy_ta] = v_txalign(x_ta, y_ta, 1.1);
save('-mat', fullfile(outdir, 'ref_txalign.mat'), 'x_ta', 'y_ta', 'kx_ta', 'ky_ta', 'nxy_ta', 'mxy_ta', 'sxy_ta');
fprintf('  v_txalign done\n');

fprintf('Phase 10 reference data generation complete.\n');
