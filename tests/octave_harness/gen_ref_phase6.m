% Generate reference data for Phase 6 LPC functions
% Usage: octave --no-gui --silent gen_ref_phase6.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 6 reference data...\n');

%% Core LPC: v_lpcauto
rng(42);
s_auto = randn(200, 1);
[ar_auto, e_auto, k_auto] = v_lpcauto(s_auto, 4);
[ar_auto_f, e_auto_f, k_auto_f] = v_lpcauto(s_auto, 4, [50 50]);
save('-mat', fullfile(outdir, 'ref_lpcauto.mat'), ...
     's_auto', 'ar_auto', 'e_auto', 'k_auto', ...
     'ar_auto_f', 'e_auto_f', 'k_auto_f');
fprintf('  v_lpcauto done\n');

%% v_lpccovar
rng(42);
s_covar = randn(200, 1);
[ar_covar, e_covar] = v_lpccovar(s_covar, 4);
save('-mat', fullfile(outdir, 'ref_lpccovar.mat'), ...
     's_covar', 'ar_covar', 'e_covar');
fprintf('  v_lpccovar done\n');

%% v_lpcbwexp
rng(42);
ar_bw = [1 -0.9 0.5 -0.2];
arx_bw = v_lpcbwexp(ar_bw, 0.1);
save('-mat', fullfile(outdir, 'ref_lpcbwexp.mat'), 'ar_bw', 'arx_bw');
fprintf('  v_lpcbwexp done\n');

%% v_lpcstable
ar_stable = [1 -0.5 0.3; 1 -1.5 0.9];
[m_stable, a_stable] = v_lpcstable(ar_stable);
save('-mat', fullfile(outdir, 'ref_lpcstable.mat'), ...
     'ar_stable', 'm_stable', 'a_stable');
fprintf('  v_lpcstable done\n');

%% v_lpcar2rf and v_lpcrf2ar roundtrip
rng(42);
ar_rf = [1 -0.7 0.3 -0.1; 1 -0.5 0.2 0.05];
rf_from_ar = v_lpcar2rf(ar_rf);
ar_from_rf = v_lpcrf2ar(rf_from_ar);
save('-mat', fullfile(outdir, 'ref_lpcar2rf.mat'), ...
     'ar_rf', 'rf_from_ar', 'ar_from_rf');
fprintf('  v_lpcar2rf / v_lpcrf2ar done\n');

%% v_lpcar2cc and v_lpccc2ar roundtrip
ar_cc = [1 -0.7 0.3 -0.1];
[cc_from_ar, c0_from_ar] = v_lpcar2cc(ar_cc);
cc_from_ar6 = v_lpcar2cc(ar_cc, 6);
ar_from_cc = v_lpccc2ar(cc_from_ar);
save('-mat', fullfile(outdir, 'ref_lpcar2cc.mat'), ...
     'ar_cc', 'cc_from_ar', 'c0_from_ar', 'cc_from_ar6', 'ar_from_cc');
fprintf('  v_lpcar2cc / v_lpccc2ar done\n');

%% v_lpcar2ff, v_lpcar2pf, v_lpcar2db
ar_spec = [1 -0.7 0.3 -0.1];
[ff_spec, f_ff] = v_lpcar2ff(ar_spec, 8);
[pf_spec, f_pf] = v_lpcar2pf(ar_spec, 8);
[db_spec, f_db] = v_lpcar2db(ar_spec, 8);
save('-mat', fullfile(outdir, 'ref_lpcar2spec.mat'), ...
     'ar_spec', 'ff_spec', 'f_ff', 'pf_spec', 'f_pf', 'db_spec', 'f_db');
fprintf('  v_lpcar2ff/pf/db done\n');

%% v_lpcar2im and v_lpcim2ar roundtrip
ar_im = [1 -0.7 0.3 -0.1];
im_from_ar = v_lpcar2im(ar_im, 10);
ar_from_im = v_lpcim2ar(im_from_ar);
save('-mat', fullfile(outdir, 'ref_lpcar2im.mat'), ...
     'ar_im', 'im_from_ar', 'ar_from_im');
fprintf('  v_lpcar2im / v_lpcim2ar done\n');

%% v_lpcar2rr and v_lpcrr2ar roundtrip
ar_rr = [1 -0.7 0.3 -0.1];
rr_from_ar = v_lpcar2rr(ar_rr);
[ar_from_rr, e_from_rr] = v_lpcrr2ar(rr_from_ar);
save('-mat', fullfile(outdir, 'ref_lpcar2rr.mat'), ...
     'ar_rr', 'rr_from_ar', 'ar_from_rr', 'e_from_rr');
fprintf('  v_lpcar2rr / v_lpcrr2ar done\n');

%% v_lpcar2ra
ar_ra = [1 -0.7 0.3 -0.1];
ra_from_ar = v_lpcar2ra(ar_ra);
save('-mat', fullfile(outdir, 'ref_lpcar2ra.mat'), 'ar_ra', 'ra_from_ar');
fprintf('  v_lpcar2ra done\n');

%% v_lpcar2zz and v_lpczz2ar roundtrip
ar_zz = [1 -0.7 0.3 -0.1];
zz_from_ar = v_lpcar2zz(ar_zz);
ar_from_zz = v_lpczz2ar(zz_from_ar);
save('-mat', fullfile(outdir, 'ref_lpcar2zz.mat'), ...
     'ar_zz', 'zz_from_ar', 'ar_from_zz');
fprintf('  v_lpcar2zz / v_lpczz2ar done\n');

%% v_lpcar2pp and v_lpcra2pp
ar_pp = [1 -0.7 0.3 -0.1];
pp_from_ar = v_lpcar2pp(ar_pp);
ra_pp = v_lpcar2ra(ar_pp);
pp_from_ra = v_lpcra2pp(ra_pp);
save('-mat', fullfile(outdir, 'ref_lpcar2pp.mat'), ...
     'ar_pp', 'pp_from_ar', 'ra_pp', 'pp_from_ra');
fprintf('  v_lpcar2pp / v_lpcra2pp done\n');

%% v_lpcar2ls and v_lpcls2ar roundtrip
ar_ls = [1 -0.7 0.3 -0.1 0.05];
ls_from_ar = v_lpcar2ls(ar_ls);
ar_from_ls = v_lpcls2ar(ls_from_ar);
save('-mat', fullfile(outdir, 'ref_lpcar2ls.mat'), ...
     'ar_ls', 'ls_from_ar', 'ar_from_ls');
fprintf('  v_lpcar2ls / v_lpcls2ar done\n');

%% Reflection coefficient conversions
rf_test = [1 -0.5 0.3 -0.2];
aa_from_rf = v_lpcrf2aa(rf_test);
ao_from_rf = v_lpcrf2ao(rf_test);
is_from_rf = v_lpcrf2is(rf_test);
la_from_rf = v_lpcrf2la(rf_test);
lo_from_rf = v_lpcrf2lo(rf_test);
[rr_from_rf, ar_from_rf_rr] = v_lpcrf2rr(rf_test);
save('-mat', fullfile(outdir, 'ref_lpcrf_conv.mat'), ...
     'rf_test', 'aa_from_rf', 'ao_from_rf', 'is_from_rf', ...
     'la_from_rf', 'lo_from_rf', 'rr_from_rf', 'ar_from_rf_rr');
fprintf('  v_lpcrf2* done\n');

%% Area conversions
aa_test = aa_from_rf;
ao_from_aa = v_lpcaa2ao(aa_test);
rf_from_aa = v_lpcaa2rf(aa_test);
rf_from_ao = v_lpcao2rf(ao_from_aa);
save('-mat', fullfile(outdir, 'ref_lpcaa_conv.mat'), ...
     'aa_test', 'ao_from_aa', 'rf_from_aa', 'rf_from_ao');
fprintf('  v_lpcaa2* / v_lpcao2rf done\n');

%% Log area conversions
rf_from_is = v_lpcis2rf(is_from_rf);
rf_from_la = v_lpcla2rf(la_from_rf);
rf_from_lo = v_lpclo2rf(lo_from_rf);
save('-mat', fullfile(outdir, 'ref_lpc_inv_conv.mat'), ...
     'is_from_rf', 'rf_from_is', 'la_from_rf', 'rf_from_la', ...
     'lo_from_rf', 'rf_from_lo');
fprintf('  v_lpcis2rf / v_lpcla2rf / v_lpclo2rf done\n');

%% v_lpccc2cc, v_lpccc2db, v_lpccc2pf
cc_test = cc_from_ar;
cc_ext = v_lpccc2cc(cc_test, 6);
[db_from_cc, f_db_cc] = v_lpccc2db(cc_test, 8);
[pf_from_cc, f_pf_cc] = v_lpccc2pf(cc_test, 8);
[ff_from_cc, f_ff_cc] = v_lpccc2ff(cc_test, 8);
save('-mat', fullfile(outdir, 'ref_lpccc_conv.mat'), ...
     'cc_test', 'cc_ext', 'db_from_cc', 'f_db_cc', ...
     'pf_from_cc', 'f_pf_cc', 'ff_from_cc', 'f_ff_cc');
fprintf('  v_lpccc2* done\n');

%% v_lpcdb2pf and v_lpcff2pf
db_test = db_spec;
pf_from_db = v_lpcdb2pf(db_test);
ff_test = ff_spec;
pf_from_ff = v_lpcff2pf(ff_test);
save('-mat', fullfile(outdir, 'ref_lpc_pf_conv.mat'), ...
     'db_test', 'pf_from_db', 'ff_test', 'pf_from_ff');
fprintf('  v_lpcdb2pf / v_lpcff2pf done\n');

%% v_lpcpf2rr and v_lpcpf2cc
pf_test = pf_spec;
rr_from_pf = v_lpcpf2rr(pf_test, 3);
[cc_from_pf, c0_from_pf] = v_lpcpf2cc(pf_test, 3);
save('-mat', fullfile(outdir, 'ref_lpcpf_conv.mat'), ...
     'pf_test', 'rr_from_pf', 'cc_from_pf', 'c0_from_pf');
fprintf('  v_lpcpf2rr / v_lpcpf2cc done\n');

%% v_lpcrr2am and v_lpcar2am
rr_am = rr_from_ar;
[am_from_rr, em_from_rr] = v_lpcrr2am(rr_am);
ar_am = ar_rr;
[am_from_ar, em_from_ar] = v_lpcar2am(ar_am);
save('-mat', fullfile(outdir, 'ref_lpc_am.mat'), ...
     'rr_am', 'am_from_rr', 'em_from_rr', ...
     'ar_am', 'am_from_ar', 'em_from_ar');
fprintf('  v_lpcrr2am / v_lpcar2am done\n');

%% v_lpcss2zz and v_lpczz2ss
ss_test = [-0.05+0.1i -0.05-0.1i -0.02+0.3i -0.02-0.3i];
zz_from_ss = v_lpcss2zz(ss_test);
ss_from_zz = v_lpczz2ss(zz_from_ss);
save('-mat', fullfile(outdir, 'ref_lpc_sszz.mat'), ...
     'ss_test', 'zz_from_ss', 'ss_from_zz');
fprintf('  v_lpcss2zz / v_lpczz2ss done\n');

%% v_lpczz2cc
zz_cc = zz_from_ar;
cc_from_zz = v_lpczz2cc(zz_cc, 5);
save('-mat', fullfile(outdir, 'ref_lpczz2cc.mat'), 'zz_cc', 'cc_from_zz');
fprintf('  v_lpczz2cc done\n');

%% v_lpccw2zz and v_lpcpz2zz
cw_test = [0.5+0.3i 0.5-0.3i -0.2+0.1i];
zz_from_cw = v_lpccw2zz(cw_test);
pz_test = cw_test;
zz_from_pz = v_lpcpz2zz(pz_test);
save('-mat', fullfile(outdir, 'ref_lpc_cwpz.mat'), ...
     'cw_test', 'zz_from_cw', 'pz_test', 'zz_from_pz');
fprintf('  v_lpccw2zz / v_lpcpz2zz done\n');

%% v_lpcra2ar (Wilson spectral factorization)
ar_ra2 = [1 -0.7 0.1];
ra_for_wilson = v_lpcar2ra(ar_ra2);
ar_from_ra = v_lpcra2ar(ra_for_wilson);
save('-mat', fullfile(outdir, 'ref_lpcra2ar.mat'), ...
     'ar_ra2', 'ra_for_wilson', 'ar_from_ra');
fprintf('  v_lpcra2ar done\n');

%% v_lpcra2pf
ra_pf = v_lpcar2ra([1 -0.7 0.3 -0.1]);
pf_from_ra = v_lpcra2pf(ra_pf, 8);
save('-mat', fullfile(outdir, 'ref_lpcra2pf.mat'), 'ra_pf', 'pf_from_ra');
fprintf('  v_lpcra2pf done\n');

%% v_lpcpp2cw and v_lpcpp2pz
pp_test = pp_from_ar;
cw_from_pp = v_lpcpp2cw(pp_test);
save('-mat', fullfile(outdir, 'ref_lpcpp2cw.mat'), 'pp_test', 'cw_from_pp');
fprintf('  v_lpcpp2cw done\n');

%% v_lpcaa2dl and v_lpcdl2aa roundtrip
aa_dl = aa_from_rf;
dl_from_aa = v_lpcaa2dl(aa_dl);
aa_from_dl = v_lpcdl2aa(dl_from_aa);
save('-mat', fullfile(outdir, 'ref_lpcaa2dl.mat'), ...
     'aa_dl', 'dl_from_aa', 'aa_from_dl');
fprintf('  v_lpcaa2dl / v_lpcdl2aa done\n');

%% v_rootstab
p_stable_poly = [1 -0.5 0.3];
[no_s, ni_s, nc_s] = v_rootstab(p_stable_poly);
p_unstable_poly = [1 -2 1.5];
[no_u, ni_u, nc_u] = v_rootstab(p_unstable_poly);
save('-mat', fullfile(outdir, 'ref_rootstab.mat'), ...
     'p_stable_poly', 'no_s', 'ni_s', 'nc_s', ...
     'p_unstable_poly', 'no_u', 'ni_u', 'nc_u');
fprintf('  v_rootstab done\n');

%% v_lpcfq2zz
f_fq = [0.1 0.3];
q_fq = [5 10];
zz_from_fq = v_lpcfq2zz(f_fq, q_fq);
save('-mat', fullfile(outdir, 'ref_lpcfq2zz.mat'), 'f_fq', 'q_fq', 'zz_from_fq');
fprintf('  v_lpcfq2zz done\n');

fprintf('Phase 6 reference data generation complete.\n');
