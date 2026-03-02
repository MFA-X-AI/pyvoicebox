% Generate reference data for Phase 9 Rotation, Quaternion & Geometry functions
% Usage: octave --no-gui --silent gen_ref_phase9.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 9 reference data...\n');

%% v_roteucode
mv_xyz = v_roteucode('xyz');
mv_zxz = v_roteucode('zxz');
mv_zyx = v_roteucode('zyx');
mv_d = v_roteucode('dxyz');
mv_O = v_roteucode('Ozyx');
save('-mat', fullfile(outdir, 'ref_roteucode.mat'), 'mv_xyz', 'mv_zxz', 'mv_zyx', 'mv_d', 'mv_O');
fprintf('  v_roteucode done\n');

%% v_roteu2qr - single and batched
e1 = [0.1; 0.2; 0.3];
q_xyz_1 = v_roteu2qr('xyz', e1);
q_zxz_1 = v_roteu2qr('zxz', e1);
q_zyx_1 = v_roteu2qr('zyx', e1);

% batched
e_batch = [0.1 0.4 0.7; 0.2 0.5 0.8; 0.3 0.6 0.9];
q_xyz_batch = v_roteu2qr('xyz', e_batch);
q_zyx_batch = v_roteu2qr('zyx', e_batch);

% fixed rotation
q_456 = v_roteu2qr('456');

% degrees
e_deg = [10; 20; 30];
q_deg = v_roteu2qr('dxyz', e_deg);

save('-mat', fullfile(outdir, 'ref_roteu2qr.mat'), ...
    'e1', 'q_xyz_1', 'q_zxz_1', 'q_zyx_1', ...
    'e_batch', 'q_xyz_batch', 'q_zyx_batch', ...
    'q_456', 'e_deg', 'q_deg');
fprintf('  v_roteu2qr done\n');

%% v_rotqr2ro
r_xyz_1 = v_rotqr2ro(q_xyz_1);
r_zxz_1 = v_rotqr2ro(q_zxz_1);
r_xyz_batch = v_rotqr2ro(q_xyz_batch);
save('-mat', fullfile(outdir, 'ref_rotqr2ro.mat'), ...
    'q_xyz_1', 'r_xyz_1', 'q_zxz_1', 'r_zxz_1', ...
    'q_xyz_batch', 'r_xyz_batch');
fprintf('  v_rotqr2ro done\n');

%% v_rotro2qr
q_back_1 = v_rotro2qr(r_xyz_1);
q_back_batch = v_rotro2qr(r_xyz_batch);
save('-mat', fullfile(outdir, 'ref_rotro2qr.mat'), ...
    'r_xyz_1', 'q_back_1', 'r_xyz_batch', 'q_back_batch');
fprintf('  v_rotro2qr done\n');

%% v_roteu2ro
r_eu2ro_1 = v_roteu2ro('xyz', e1);
r_eu2ro_batch = v_roteu2ro('xyz', e_batch);
r_eu2ro_zxz = v_roteu2ro('zxz', e1);
save('-mat', fullfile(outdir, 'ref_roteu2ro.mat'), ...
    'e1', 'r_eu2ro_1', 'e_batch', 'r_eu2ro_batch', 'r_eu2ro_zxz');
fprintf('  v_roteu2ro done\n');

%% v_rotro2eu
e_back_xyz_1 = v_rotro2eu('xyz', r_eu2ro_1);
e_back_xyz_batch = v_rotro2eu('xyz', r_eu2ro_batch);
e_back_zxz = v_rotro2eu('zxz', r_eu2ro_zxz);
save('-mat', fullfile(outdir, 'ref_rotro2eu.mat'), ...
    'r_eu2ro_1', 'e_back_xyz_1', 'r_eu2ro_batch', 'e_back_xyz_batch', ...
    'r_eu2ro_zxz', 'e_back_zxz');
fprintf('  v_rotro2eu done\n');

%% v_rotqr2eu
e_qr2eu_1 = v_rotqr2eu('xyz', q_xyz_1);
e_qr2eu_zxz = v_rotqr2eu('zxz', q_zxz_1);
save('-mat', fullfile(outdir, 'ref_rotqr2eu.mat'), ...
    'q_xyz_1', 'e_qr2eu_1', 'q_zxz_1', 'e_qr2eu_zxz');
fprintf('  v_rotqr2eu done\n');

%% v_rotqc2qr / v_rotqr2qc
qr1 = [0.5; 0.5; 0.5; 0.5];
qc1 = v_rotqr2qc(qr1);
qr1_back = v_rotqc2qr(qc1);
save('-mat', fullfile(outdir, 'ref_rotqc.mat'), 'qr1', 'qc1', 'qr1_back');
fprintf('  v_rotqc2qr / v_rotqr2qc done\n');

%% v_rotqc2mc / v_rotmc2qc
mc1 = v_rotqc2mc(qc1);
qc1_back = v_rotmc2qc(mc1);
save('-mat', fullfile(outdir, 'ref_rotmc.mat'), 'qc1', 'mc1', 'qc1_back');
fprintf('  v_rotqc2mc / v_rotmc2qc done\n');

%% v_rotqr2mr / v_rotmr2qr
mr1 = v_rotqr2mr(qr1);
qr1_mr_back = v_rotmr2qr(mr1);
save('-mat', fullfile(outdir, 'ref_rotmr.mat'), 'qr1', 'mr1', 'qr1_mr_back');
fprintf('  v_rotqr2mr / v_rotmr2qr done\n');

%% v_rotax2qr / v_rotqr2ax
ax1 = [1; 2; 3];
t1 = 0.5;
q_ax = v_rotax2qr(ax1, t1);
[ax_back, t_back] = v_rotqr2ax(q_ax);
save('-mat', fullfile(outdir, 'ref_rotax.mat'), 'ax1', 't1', 'q_ax', 'ax_back', 't_back');
fprintf('  v_rotax2qr / v_rotqr2ax done\n');

%% v_rotqrmean
rng(42);
q_mean_in = zeros(4, 5);
for i=1:5
    e_rand = randn(3,1) * 0.3;
    q_mean_in(:,i) = v_roteu2qr('xyz', e_rand);
end
[y_mean, s_mean, v_mean] = v_rotqrmean(q_mean_in);
save('-mat', fullfile(outdir, 'ref_rotqrmean.mat'), 'q_mean_in', 'y_mean', 's_mean', 'v_mean');
fprintf('  v_rotqrmean done\n');

%% v_rotqrvec
q_vec = v_roteu2qr('xyz', [0.1; 0.2; 0.3]);
x_vec = [1; 0; 0];
y_vec = v_rotqrvec(q_vec, x_vec);
x_multi = [1 0 0; 0 1 0; 0 0 1]';
y_multi = v_rotqrvec(q_vec, x_multi);
save('-mat', fullfile(outdir, 'ref_rotqrvec.mat'), 'q_vec', 'x_vec', 'y_vec', 'x_multi', 'y_multi');
fprintf('  v_rotqrvec done\n');

%% v_rotation
r_rot1 = v_rotation([1;0;0], [0;1;0], pi/4);
% axis-angle mode
r_rot2 = v_rotation([0;0;1], pi/3);
% axis*angle mode
r_rot3 = v_rotation([0;0;pi/6]);
save('-mat', fullfile(outdir, 'ref_rotation.mat'), 'r_rot1', 'r_rot2', 'r_rot3');
fprintf('  v_rotation done\n');

%% v_rotpl2ro / v_rotro2pl
u1 = [1; 0; 0];
v1 = [0; 1; 0];
r_pl = v_rotpl2ro(u1, v1, pi/4);
[u_back, v_back, t_plback] = v_rotro2pl(r_pl);
save('-mat', fullfile(outdir, 'ref_rotpl.mat'), 'u1', 'v1', 'r_pl', 'u_back', 'v_back', 't_plback');
fprintf('  v_rotpl2ro / v_rotro2pl done\n');

%% v_rotlu2ro / v_rotro2lu
l1 = [1; 2; 3];
u_lu = [0; 0; 1];
r_lu = v_rotlu2ro(l1, u_lu);
[l_back, u_back_lu] = v_rotro2lu(r_lu);
save('-mat', fullfile(outdir, 'ref_rotlu.mat'), 'l1', 'u_lu', 'r_lu', 'l_back', 'u_back_lu');
fprintf('  v_rotlu2ro / v_rotro2lu done\n');

%% v_polygonarea
p_tri = [0 0; 1 0; 0 1];
a_tri = v_polygonarea(p_tri);
p_sq = [0 0; 1 0; 1 1; 0 1];
a_sq = v_polygonarea(p_sq);
p_ccw = [0 0; 2 0; 2 3; 0 3];
a_ccw = v_polygonarea(p_ccw);
save('-mat', fullfile(outdir, 'ref_polygonarea.mat'), ...
    'p_tri', 'a_tri', 'p_sq', 'a_sq', 'p_ccw', 'a_ccw');
fprintf('  v_polygonarea done\n');

%% v_polygonwind
p_wind = [0 0; 4 0; 4 3; 0 3];
x_wind = [2 1.5; 5 1.5; 2 -1; -1 1.5; 0 0];
w_wind = v_polygonwind(p_wind, x_wind);
save('-mat', fullfile(outdir, 'ref_polygonwind.mat'), 'p_wind', 'x_wind', 'w_wind');
fprintf('  v_polygonwind done\n');

%% v_polygonxline
p_xline = [0 0; 4 0; 4 3; 0 3];
l_xline = [0 1 -1.5];  % y = 1.5
[xc_xl, ec_xl, tc_xl, xy0_xl] = v_polygonxline(p_xline, l_xline);
save('-mat', fullfile(outdir, 'ref_polygonxline.mat'), ...
    'p_xline', 'l_xline', 'xc_xl', 'ec_xl', 'tc_xl', 'xy0_xl');
fprintf('  v_polygonxline done\n');

%% v_skew3d
x_sk3 = [1; 2; 3];
y_sk3 = v_skew3d(x_sk3);
x_sk3_back = v_skew3d(y_sk3);
x_sk6 = [1; 2; 3; 4; 5; 6];
y_sk6 = v_skew3d(x_sk6);
x_sk6_back = v_skew3d(y_sk6);
save('-mat', fullfile(outdir, 'ref_skew3d.mat'), ...
    'x_sk3', 'y_sk3', 'x_sk3_back', 'x_sk6', 'y_sk6', 'x_sk6_back');
fprintf('  v_skew3d done\n');

fprintf('Phase 9 reference data generation complete.\n');
