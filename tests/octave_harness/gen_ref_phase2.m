% Generate reference data for Phase 2 frequency conversion functions
% Usage: octave --no-gui --silent gen_ref_phase2.m <voicebox_dir> <output_dir>

args = argv();
addpath(args{1});
outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating Phase 2 reference data...\n');

%% v_frq2mel
frq_mel = [0 100 200 500 1000 2000 4000 8000 -500 -1000];
[mel_out, mel_mr] = v_frq2mel(frq_mel);

save('-mat', fullfile(outdir, 'ref_frq2mel.mat'), ...
     'frq_mel', 'mel_out', 'mel_mr');
fprintf('  v_frq2mel done\n');

%% v_mel2frq
mel_in = [0 100 200 500 1000 1500 2000 3000 -500 -1000];
[mel2frq_out, mel2frq_mr] = v_mel2frq(mel_in);

save('-mat', fullfile(outdir, 'ref_mel2frq.mat'), ...
     'mel_in', 'mel2frq_out', 'mel2frq_mr');
fprintf('  v_mel2frq done\n');

%% v_frq2bark - default mode
frq_bark = [0 50 100 200 500 1000 2000 4000 8000 16000 -500 -1000];
[bark_def, bark_def_c] = v_frq2bark(frq_bark);

% mode 'z' (Zwicker)
[bark_z, bark_z_c] = v_frq2bark(frq_bark, 'z');

% mode 's' (Schroeder)
[bark_s, bark_s_c] = v_frq2bark(frq_bark, 's');

% mode 'lh' (Traunmuller low+high corrections)
[bark_lh, bark_lh_c] = v_frq2bark(frq_bark, 'lh');

% mode 'LH' (no corrections)
[bark_LH, bark_LH_c] = v_frq2bark(frq_bark, 'LH');

% mode 'u' (unipolar)
frq_bark_u = [0 50 100 500 1000 4000 8000];
[bark_u, bark_u_c] = v_frq2bark(frq_bark_u, 'u');

save('-mat', fullfile(outdir, 'ref_frq2bark.mat'), ...
     'frq_bark', 'bark_def', 'bark_def_c', ...
     'bark_z', 'bark_z_c', 'bark_s', 'bark_s_c', ...
     'bark_lh', 'bark_lh_c', 'bark_LH', 'bark_LH_c', ...
     'frq_bark_u', 'bark_u', 'bark_u_c');
fprintf('  v_frq2bark done\n');

%% v_bark2frq - default mode
bark_in = [0 1 2 3 5 8 10 15 20 24 -5 -10];
[b2f_def, b2f_def_c] = v_bark2frq(bark_in);

% mode 's' (Schroeder)
[b2f_s, b2f_s_c] = v_bark2frq(bark_in, 's');

% mode 'lh' (Traunmuller low+high corrections)
[b2f_lh, b2f_lh_c] = v_bark2frq(bark_in, 'lh');

% mode 'LH' (no corrections)
[b2f_LH, b2f_LH_c] = v_bark2frq(bark_in, 'LH');

% mode 'u' (unipolar)
bark_in_u = [0 1 3 5 10 15 20];
[b2f_u, b2f_u_c] = v_bark2frq(bark_in_u, 'u');

save('-mat', fullfile(outdir, 'ref_bark2frq.mat'), ...
     'bark_in', 'b2f_def', 'b2f_def_c', ...
     'b2f_s', 'b2f_s_c', 'b2f_lh', 'b2f_lh_c', ...
     'b2f_LH', 'b2f_LH_c', ...
     'bark_in_u', 'b2f_u', 'b2f_u_c');
fprintf('  v_bark2frq done\n');

%% v_frq2erb
frq_erb = [0 50 100 200 500 1000 2000 4000 8000 -500 -1000];
[erb_out, erb_bnd] = v_frq2erb(frq_erb);

save('-mat', fullfile(outdir, 'ref_frq2erb.mat'), ...
     'frq_erb', 'erb_out', 'erb_bnd');
fprintf('  v_frq2erb done\n');

%% v_erb2frq
erb_in = [0 1 5 10 15 20 25 30 35 40 -5 -10];
[e2f_out, e2f_bnd] = v_erb2frq(erb_in);

save('-mat', fullfile(outdir, 'ref_erb2frq.mat'), ...
     'erb_in', 'e2f_out', 'e2f_bnd');
fprintf('  v_erb2frq done\n');

%% v_frq2cent
frq_cent = [100 200 440 500 1000 2000 4000 8000 -440 -1000];
[cent_out, cent_cr] = v_frq2cent(frq_cent);

save('-mat', fullfile(outdir, 'ref_frq2cent.mat'), ...
     'frq_cent', 'cent_out', 'cent_cr');
fprintf('  v_frq2cent done\n');

%% v_cent2frq
cent_in = [0 1200 2400 3600 4800 5700 6000 6900 -5700 -6900];
[c2f_out, c2f_cr] = v_cent2frq(cent_in);

save('-mat', fullfile(outdir, 'ref_cent2frq.mat'), ...
     'cent_in', 'c2f_out', 'c2f_cr');
fprintf('  v_cent2frq done\n');

%% v_frq2midi
frq_midi = [261.63 293.66 329.63 349.23 392.00 440.00 493.88 523.25 -440.00];
[midi_out, midi_text] = v_frq2midi(frq_midi);

save('-mat', fullfile(outdir, 'ref_frq2midi.mat'), ...
     'frq_midi', 'midi_out', 'midi_text');
fprintf('  v_frq2midi done\n');

%% v_midi2frq - equal tempered (default)
midi_in = [60 61 62 63 64 65 66 67 68 69 70 71 72];
midi2frq_e = v_midi2frq(midi_in);

% pythagorean
midi2frq_p = v_midi2frq(midi_in, 'p');

% just intonation
midi2frq_j = v_midi2frq(midi_in, 'j');

% fractional notes for pythagorean
midi_frac = [60.5 61.5 69.25];
midi2frq_p_frac = v_midi2frq(midi_frac, 'p');
midi2frq_j_frac = v_midi2frq(midi_frac, 'j');

save('-mat', fullfile(outdir, 'ref_midi2frq.mat'), ...
     'midi_in', 'midi2frq_e', 'midi2frq_p', 'midi2frq_j', ...
     'midi_frac', 'midi2frq_p_frac', 'midi2frq_j_frac');
fprintf('  v_midi2frq done\n');

fprintf('Phase 2 reference data generation complete.\n');
