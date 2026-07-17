function stokes_analyser()
% STOKES_ANALYSER  Stokes polarimetry from FOUR hyperspectral QWP measurements.
%
% MATLAB version of stokes_app.py. Loads a stack of four hyperspectral
% measurements taken through a quarter-wave plate at four angles, averages each
% spectral cube over a user wavelength range to get one intensity image per
% angle, and computes the Stokes parameters S0,S1,S2,S3 per pixel.
%
% Each measurement is a .npz saved by the acquisition app: a spectral cube
% `spectrum_cube` (n_wavelengths, h, w) and a `wavelengths` axis (um). The QWP
% angle is stored (angle_value_deg) when the acquisition scanned angle; the
% measurement frame = filename QWP angle - 45 deg.
%
% Features:
%   * Formula selector -- two methods, each with its own 4 angles + Stokes eqns.
%   * Load folder (auto-assign by angle): reads each file's angle/z metadata and
%     fills I1..I4 at the method's target angles; a z-position dropdown lets you
%     switch z within a z-stack. Or load each slot by hand (filename shown).
%   * Wavelength averaging range + optional FLAT-FIELD correction (divide by the
%     frame at a chosen lambda). When on, Stokes use the corrected intensities.
%   * S1..S3 on a diverging blue-white-red map with adjustable colour limits;
%     S0 and the intensities on viridis.
%
% Reads acquisition_app .npz directly (no Python). Place anywhere and run:
%     stokes_analyser

ANGLE_OFFSET = -45;    % measurement frame = filename QWP angle - 45 deg

% ---- calculation methods (name, target filename angles, box titles, Stokes) ----
titlesA = {'S_0 = I_1 + I_4', ...
           'S_1 = (2 I_2 - I_1 - I_4) / S_0', ...
           'S_2 = [(I_1 - I_4)\surd2 - I_1 - 2 I_2 + 4 I_3 - I_4] / S_0', ...
           'S_3 = (I_1 - I_4) / S_0'};
titlesB = {'S_0 = I_1 + I_4', ...
           'S_1 = 2 (I_2 + I_3 - I_1 - I_4) / S_0', ...
           'S_2 = [(I_1 - I_4)\surd2 - 2 I_2 + 2 I_3] / S_0', ...
           'S_3 = (I_1 - I_4) / S_0'};
methods_(1).name    = '0, 45, 67.5, 90  ->  frame -45, 0, 22.5, 45';
methods_(1).fangles = [0 45 67.5 90];
methods_(1).titles  = titlesA;
methods_(1).fn      = @stokesA;
methods_(2).name    = '0, 22.5, 67.5, 90  ->  frame -45, -22.5, 22.5, 45';
methods_(2).fangles = [0 22.5 67.5 90];
methods_(2).titles  = titlesB;
methods_(2).fn      = @stokesB;

VIR = viridisMap();
BWR = bwrMap();

% ------------------------------------------------------------------ state
slots = struct('loaded', {false,false,false,false}, ...
               'cube', {[],[],[],[]}, 'wl', {[],[],[],[]}, 'name', {'','','',''});
Icell    = cell(1, 4);      % raw averaged intensity images
Ibkgcell = cell(1, 4);      % flat-field-corrected intensity images
Scell    = cell(1, 4);      % Stokes maps
metaPaths = {}; metaAngle = []; metaZ = [];   % last scanned folder
ranges_init = false;
first_stokes = true;
syncing = false;
if isfolder('D:\CAMERA'), last_dir = 'D:\CAMERA'; else, last_dir = pwd; end

% handle arrays (assigned during UI build)
angle_fields = gobjects(1, 4);
file_labels  = gobjects(1, 4);
axI = gobjects(1, 4);  imI = gobjects(1, 4);
axS = gobjects(1, 4);  imS = gobjects(1, 4);
s_min = gobjects(1, 4);  s_max = gobjects(1, 4);

% ------------------------------------------------------------------ build UI
fig = uifigure('Name', 'Stokes Polarimetry — hyperspectral QWP analyzer', ...
    'Color', 'w', 'Position', [80, 60, 1320, 880]);
G = uigridlayout(fig, [5, 1]);
G.RowHeight = {38, 96, 38, '1x', 24};
G.Padding = [8 8 8 8]; G.RowSpacing = 6;

% Row 1: formula + folder + z
c1 = uigridlayout(G, [1, 6]);
c1.Layout.Row = 1; c1.ColumnWidth = {55, 320, 210, 75, 130, '1x'};
c1.Padding = [0 0 0 0];
uilabel(c1, 'Text', 'Formula:');
method_dd = uidropdown(c1, 'Items', {methods_.name}, 'ItemsData', 1:numel(methods_), ...
    'Value', 1, 'ValueChangedFcn', @(s,e) onMethodChanged());
btn_folder = uibutton(c1, 'Text', 'Load folder (auto by angle)…', ...
    'ButtonPushedFcn', @(s,e) loadFolder());  %#ok<NASGU>
uilabel(c1, 'Text', 'z-position:', 'HorizontalAlignment', 'right');
z_dd = uidropdown(c1, 'Items', {'—'}, 'ItemsData', [], 'Enable', 'off', ...
    'ValueChangedFcn', @(s,e) onZChanged());
uilabel(c1, 'Text', 'reads each .npz angle to fill I1..I4; or load slots by hand', ...
    'FontColor', [.5 .5 .5]);

% Row 2: four slot loaders
srow = uigridlayout(G, [1, 4]); srow.Layout.Row = 2; srow.Padding = [0 0 0 0];
for ii = 1:4
    p = uipanel(srow, 'Title', sprintf('I%d', ii));
    pgl = uigridlayout(p, [2, 3]);
    pgl.RowHeight = {24, '1x'}; pgl.ColumnWidth = {64, 34, '1x'};
    pgl.Padding = [6 4 6 4]; pgl.RowSpacing = 2;
    b = uibutton(pgl, 'Text', 'Load…', 'ButtonPushedFcn', @(s,e) loadSlot(ii)); %#ok<NASGU>
    uilabel(pgl, 'Text', 'QWP');
    angle_fields(ii) = uieditfield(pgl, 'numeric', 'ValueDisplayFormat', '%.1f°', ...
        'ValueChangedFcn', @(s,e) recompute());
    file_labels(ii) = uilabel(pgl, 'Text', '(not loaded)', 'FontColor', [.5 .5 .5], ...
        'WordWrap', 'on');
    file_labels(ii).Layout.Row = 2; file_labels(ii).Layout.Column = [1 3];
end

% Row 3: wavelength range + flat-field
r3 = uigridlayout(G, [1, 9]); r3.Layout.Row = 3; r3.Padding = [0 0 0 0];
r3.ColumnWidth = {105, 90, 25, 90, 200, 90, 30, 230, '1x'};
uilabel(r3, 'Text', 'Average λ from');
sp_wl0 = uieditfield(r3, 'numeric', 'ValueDisplayFormat', '%.4f', ...
    'ValueChangedFcn', @(s,e) recompute());
uilabel(r3, 'Text', 'to', 'HorizontalAlignment', 'center');
sp_wl1 = uieditfield(r3, 'numeric', 'ValueDisplayFormat', '%.4f', ...
    'ValueChangedFcn', @(s,e) recompute());
chk_ff = uicheckbox(r3, 'Text', 'Flat-field correction (÷ frame at λ)', ...
    'ValueChangedFcn', @(s,e) recompute());
sp_wlbkg = uieditfield(r3, 'numeric', 'ValueDisplayFormat', '%.4f', ...
    'ValueChangedFcn', @(s,e) recompute());
uilabel(r3, 'Text', 'µm');
uilabel(r3, 'Text', 'slot angle = filename QWP angle − 45°', 'FontColor', [.5 .5 .5]);

% Row 4: tabs
tg = uitabgroup(G); tg.Layout.Row = 4;
tabI = uitab(tg, 'Title', 'Intensities');
tabS = uitab(tg, 'Title', 'Stokes S0–S3');

ig = uigridlayout(tabI, [2, 2]); ig.Padding = [6 6 6 6];
for ii = 1:4
    axI(ii) = uiaxes(ig);
    imI(ii) = imagesc(axI(ii), zeros(2));
    axis(axI(ii), 'image'); colormap(axI(ii), VIR); colorbar(axI(ii));
    title(axI(ii), sprintf('I%d', ii), 'Interpreter', 'none');
end

sgTab = uigridlayout(tabS, [2, 1]); sgTab.RowHeight = {'1x', 46}; sgTab.Padding = [6 6 6 6];
sag = uigridlayout(sgTab, [2, 2]); sag.Layout.Row = 1; sag.Padding = [0 0 0 0];
climrow = uigridlayout(sgTab, [1, 4]); climrow.Layout.Row = 2; climrow.Padding = [0 0 0 0];
for ii = 1:4
    axS(ii) = uiaxes(sag);
    imS(ii) = imagesc(axS(ii), zeros(2));
    axis(axS(ii), 'image');
    if ii == 1, colormap(axS(ii), VIR); else, colormap(axS(ii), BWR); end
    colorbar(axS(ii));
    title(axS(ii), methods_(method_dd.Value).titles{ii});   % TeX
    % clim controls
    cc = uigridlayout(climrow, [1, 6]);
    cc.ColumnWidth = {32, 28, '1x', 28, '1x', 46}; cc.Padding = [2 2 2 2];
    uilabel(cc, 'Text', sprintf('S%d', ii-1), 'FontWeight', 'bold');
    uilabel(cc, 'Text', 'min', 'HorizontalAlignment', 'right');
    s_min(ii) = uieditfield(cc, 'numeric', 'ValueChangedFcn', @(s,e) onClimEdit(ii));
    uilabel(cc, 'Text', 'max', 'HorizontalAlignment', 'right');
    s_max(ii) = uieditfield(cc, 'numeric', 'ValueChangedFcn', @(s,e) onClimEdit(ii));
    uibutton(cc, 'Text', 'Auto', 'ButtonPushedFcn', @(s,e) autoClim(ii));
end

% Row 5: status
status_lbl = uilabel(G, 'Text', 'Load four QWP measurements (one per angle).');
status_lbl.Layout.Row = 5; status_lbl.FontColor = [.3 .3 .3];

% set default slot angles for the initial method
fr0 = frameAngles();
for ii = 1:4, angle_fields(ii).Value = fr0(ii); end

% ======================================================================
%                       NESTED CALLBACKS / LOGIC
% ======================================================================
    function fr = frameAngles()
        fr = methods_(method_dd.Value).fangles + ANGLE_OFFSET;
    end

    function onMethodChanged()
        mth = methods_(method_dd.Value);
        for kk = 1:4, title(axS(kk), mth.titles{kk}); end
        first_stokes = true;                       % re-autoscale for the new formula
        if ~isempty(metaPaths)
            assignFromFolder(currentZ());
        else
            fra = frameAngles();
            for kk = 1:4
                if ~slots(kk).loaded, angle_fields(kk).Value = fra(kk); end
            end
            recompute();
        end
    end

    % ------------------------------------------------------------- loading
    function ok = assignSlot(k, path)
        ok = false;
        try
            [cube, wl, angle] = loadMeasurement(path);
        catch ME
            uialert(fig, sprintf('%s\n%s', nameOf(path), ME.message), 'Load error');
            return;
        end
        slots(k).loaded = true; slots(k).cube = cube; slots(k).wl = wl;
        slots(k).name = nameOf(path);
        file_labels(k).Text = nameOf(path);
        file_labels(k).FontColor = [0.18 0.62 0.27];
        file_labels(k).Tooltip = path;
        if ~isnan(angle)
            angle_fields(k).Value = angle + ANGLE_OFFSET;
        end
        if ~ranges_init
            sp_wl0.Value = min(wl); sp_wl1.Value = max(wl); sp_wlbkg.Value = max(wl);
            ranges_init = true;
        end
        ok = true;
    end

    function loadSlot(k)
        [f, p] = uigetfile({'*.npz', 'Measurement (*.npz)'}, ...
            sprintf('QWP measurement for slot I%d', k), last_dir);
        if isequal(f, 0), return; end
        last_dir = p;
        if assignSlot(k, fullfile(p, f)), recompute(); end
    end

    function loadFolder()
        d = uigetdir(last_dir, 'Folder holding the QWP angle stack');
        if isequal(d, 0), return; end
        last_dir = d;
        listing = dir(fullfile(d, '*.npz'));
        if isempty(listing)
            uialert(fig, 'No .npz files directly in that folder.', 'No files'); return;
        end
        pths = arrayfun(@(f) fullfile(d, f.name), listing, 'UniformOutput', false);
        populateFromPaths(pths);
    end

    function populateFromPaths(paths)
        metaPaths = {}; metaAngle = []; metaZ = [];
        fig.Pointer = 'watch'; drawnow;
        for i = 1:numel(paths)
            [ang, zval] = readMeta(paths{i});
            metaPaths{end+1} = paths{i}; metaAngle(end+1) = ang; metaZ(end+1) = zval; %#ok<AGROW>
        end
        fig.Pointer = 'arrow';
        if all(isnan(metaAngle))
            status_lbl.Text = ['Files carry no angle (angle_value_deg) — cannot ', ...
                'auto-assign. Load the four measurements by hand.'];
            return;
        end
        zs = unique(round(metaZ(~isnan(metaZ)), 4));
        if isempty(zs)
            z_dd.Items = {'—'}; z_dd.ItemsData = []; z_dd.Enable = 'off';
            assignFromFolder(NaN);
        else
            items = arrayfun(@(z) sprintf('%.4f mm', z), zs, 'UniformOutput', false);
            z_dd.Items = items; z_dd.ItemsData = zs; z_dd.Value = zs(1);
            if numel(zs) > 1, z_dd.Enable = 'on'; else, z_dd.Enable = 'off'; end
            assignFromFolder(zs(1));
        end
    end

    function z = currentZ()
        if isempty(z_dd.ItemsData), z = NaN; else, z = z_dd.Value; end
    end

    function onZChanged()
        if isempty(metaPaths), return; end
        assignFromFolder(currentZ());
    end

    function assignFromFolder(zsel)
        mth = methods_(method_dd.Value);
        targets = mth.fangles;
        loaded = {}; missing = {};
        for q = 1:4
            ta = targets(q);
            mask = ~isnan(metaAngle);
            if ~isnan(zsel), mask = mask & (abs(metaZ - zsel) < 1e-3); end
            idxs = find(mask);
            if isempty(idxs), missing{end+1} = sprintf('%g°', ta); continue; end %#ok<AGROW>
            [dmin, jj] = min(abs(metaAngle(idxs) - ta));
            j = idxs(jj);
            if dmin <= 1.0 && assignSlot(q, metaPaths{j})
                loaded{end+1} = sprintf('I%d←%g°', q, metaAngle(j)); %#ok<AGROW>
            else
                missing{end+1} = sprintf('%g°', ta); %#ok<AGROW>
            end
        end
        recompute();
        if ~isnan(zsel), zmsg = sprintf(' at z = %.4f mm', zsel); else, zmsg = ''; end
        msg = sprintf('Auto-loaded %d/4%s: %s', numel(loaded), zmsg, strjoin(loaded, ', '));
        if ~isempty(missing)
            msg = [msg, '  |  no file near angle(s): ', strjoin(missing, ', ')];
        end
        status_lbl.Text = msg;
    end

    % ------------------------------------------------------------- compute
    function recompute()
        if ~all([slots.loaded])
            status_lbl.Text = sprintf('Loaded %d/4 measurements — load all four to compute Stokes.', ...
                sum([slots.loaded]));
            return;
        end
        s1 = size(slots(1).cube); H = s1(2); W = s1(3);
        for q = 2:4
            sk = size(slots(q).cube);
            if numel(sk) < 3 || sk(2) ~= H || sk(3) ~= W
                status_lbl.Text = 'Image sizes differ across slots — cannot combine.'; return;
            end
        end
        wl0 = sp_wl0.Value; wl1 = sp_wl1.Value; wlb = sp_wlbkg.Value;
        i0 = 1; i1 = 1; ibk = 1; wl = slots(4).wl;
        for q = 1:4
            wl = slots(q).wl; cube = slots(q).cube;
            [~, a] = min(abs(wl - wl0)); [~, b] = min(abs(wl - wl1));
            i0 = min(a, b); i1 = max(a, b);
            Icell{q} = double(squeeze(mean(cube(i0:i1, :, :), 1)));   % (h,w)
            [~, ibk] = min(abs(wl - wlb));
            Ibkgcell{q} = Icell{q} ./ double(squeeze(cube(ibk, :, :)));
        end
        if chk_ff.Value, X = Ibkgcell; else, X = Icell; end
        mth = methods_(method_dd.Value);
        [S0, S1, S2, S3] = mth.fn(X{1}, X{2}, X{3}, X{4});
        Scell = {S0, S1, S2, S3};
        for q = 1:4, mp = Scell{q}; mp(~isfinite(mp)) = NaN; Scell{q} = mp; end
        updateIntensities();
        updateStokes();
        status_lbl.Text = sprintf(['λ average %.4f–%.4f µm (%d bands) | ', ...
            'flat-field %s (÷ frame at %.4f µm) | Stokes from %s intensities'], ...
            wl(i0), wl(i1), i1 - i0 + 1, ternary(chk_ff.Value, 'ON', 'off'), ...
            wl(ibk), ternary(chk_ff.Value, 'flat-fielded', 'raw'));
    end

    function updateIntensities()
        if chk_ff.Value, disp_ = Ibkgcell; else, disp_ = Icell; end
        for q = 1:4
            d = disp_{q};
            imI(q).CData = d; setImgLimits(axI(q), d);
            [lo, hi] = finiteRange(d);
            clim(axI(q), [lo, hi]);
            ttl = sprintf('I%d - frame %g°', q, angle_fields(q).Value);
            if chk_ff.Value, ttl = [ttl, '  (flat-fielded)']; end
            title(axI(q), ttl, 'Interpreter', 'none');
        end
    end

    function updateStokes()
        for q = 1:4
            imS(q).CData = Scell{q}; setImgLimits(axS(q), Scell{q});
            if first_stokes
                autoClim(q);
            else
                lo = s_min(q).Value; hi = s_max(q).Value;
                if hi > lo, clim(axS(q), [lo, hi]); end
            end
        end
        first_stokes = false;
    end

    % ----------------------------------------------------- colour limits
    function autoClim(k)
        d = Scell{k}; fin = d(isfinite(d));
        if isempty(fin)
            lo = -1; hi = 1;
        elseif k == 1                              % S0: intensity -> data range
            lo = min(fin); hi = max(fin);
        else                                       % S1..S3: symmetric about 0
            mm = max(abs(fin));
            if isempty(mm) || ~isfinite(mm) || mm <= 0, mm = 1; end
            lo = -mm; hi = mm;
        end
        if hi <= lo, hi = lo + 1e-9; end
        syncing = true;
        s_min(k).Value = lo; s_max(k).Value = hi;
        clim(axS(k), [lo, hi]);
        syncing = false;
    end

    function onClimEdit(k)
        if syncing, return; end
        lo = s_min(k).Value; hi = s_max(k).Value;
        if hi > lo, clim(axS(k), [lo, hi]); end
    end
end

% ======================================================================
%                    STOKES FORMULAS (per method)
% ======================================================================
function [S0, S1, S2, S3] = stokesA(I1, I2, I3, I4)
% Filename angles 0, 45, 67.5, 90 (frame -45, 0, 22.5, 45).
S0 = I1 + I4;
S1 = (2*I2 - I1 - I4) ./ S0;
S2 = ((I1 - I4)*sqrt(2) - I1 - 2*I2 + 4*I3 - I4) ./ S0;
S3 = (I1 - I4) ./ S0;
end

function [S0, S1, S2, S3] = stokesB(I1, I2, I3, I4)
% Filename angles 0, 22.5, 67.5, 90 (frame -45, -22.5, 22.5, 45).
S0 = I1 + I4;
S1 = (2*I2 + 2*I3 - 2*I1 - 2*I4) ./ S0;
S2 = ((I1 - I4)*sqrt(2) - 2*I2 + 2*I3) ./ S0;
S3 = (I1 - I4) ./ S0;
end

% ======================================================================
%                    SMALL HELPERS
% ======================================================================
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function [lo, hi] = finiteRange(d)
fin = d(isfinite(d));
if isempty(fin), lo = 0; hi = 1; else, lo = double(min(fin)); hi = double(max(fin)); end
if hi <= lo, hi = lo + 1; end
end

function setImgLimits(ax, d)
ax.XLim = [0.5, size(d, 2) + 0.5];
ax.YLim = [0.5, size(d, 1) + 0.5];
end

function n = nameOf(path)
[~, b, e] = fileparts(path); n = [b, e];
end

function cm = viridisMap()
anchors = [ 68 1 84; 71 44 122; 59 81 139; 44 113 142; 33 144 141; ...
            39 173 129; 92 200 99; 170 220 50; 253 231 37] / 255;
x = linspace(0, 1, size(anchors, 1)); xi = linspace(0, 1, 256).';
cm = [interp1(x, anchors(:,1), xi), interp1(x, anchors(:,2), xi), interp1(x, anchors(:,3), xi)];
end

function cm = bwrMap()
cm = interp1([0 0.5 1], [0 0 1; 1 1 1; 1 0 0], linspace(0, 1, 256).');
end

% ======================================================================
%              MEASUREMENT + METADATA READING (.npz)
% ======================================================================
function [cube, wl, angle] = loadMeasurement(path)
M = readNPZmembers(path, {'spectrum_cube', 'spectrum_cubes', 'wavelengths', 'angle_value_deg'});
if isfield(M, 'spectrum_cube')
    cube = M.spectrum_cube;
elseif isfield(M, 'spectrum_cubes')
    cube = M.spectrum_cubes;
    if ndims(cube) == 4, cube = squeeze(cube(1, :, :, :)); end
else
    error('No spectrum_cube (or spectrum_cubes) in %s.', nameOf(path));
end
if ~isfield(M, 'wavelengths'), error('No wavelengths axis in %s.', nameOf(path)); end
if ndims(cube) ~= 3
    error('spectrum_cube must be 3-D (n_wl,h,w); got %s.', mat2str(size(cube)));
end
cube = single(cube);
wl = double(M.wavelengths(:));
angle = NaN;
if isfield(M, 'angle_value_deg') && ~isempty(M.angle_value_deg)
    angle = double(M.angle_value_deg(1));
end
end

function [angle, z] = readMeta(path)
% Read only the small angle/z members (no big cube).
angle = NaN; z = NaN;
try
    M = readNPZmembers(path, {'angle_value_deg', 'z_value_mm', 'metadata_json'});
    if isfield(M, 'angle_value_deg') && ~isempty(M.angle_value_deg)
        angle = double(M.angle_value_deg(1));
    end
    if isfield(M, 'z_value_mm') && ~isempty(M.z_value_mm)
        z = double(M.z_value_mm(1));
    end
    if (isnan(angle) || isnan(z)) && isfield(M, 'metadata_json')
        m = jsondecode(M.metadata_json);
        if isnan(angle) && isfield(m, 'angle_value_deg') && ~isempty(m.angle_value_deg)
            angle = double(m.angle_value_deg);
        end
        if isnan(z) && isfield(m, 'z_value_mm') && ~isempty(m.z_value_mm)
            z = double(m.z_value_mm);
        end
    end
catch
end
end

% ======================================================================
%              NPZ / NPY READER (targeted) -- from phase_analyser.m
% ======================================================================
function S = readNPZmembers(npz_path, want_names)
S = struct();
want = containers.Map();
for i = 1:numel(want_names), want(want_names{i}) = true; end
zf = java.util.zip.ZipFile(java.io.File(npz_path));
cleaner = onCleanup(@() zf.close()); %#ok<NASGU>
entries = zf.entries();
copier = com.mathworks.mlwidgets.io.InterruptibleStreamCopier.getInterruptibleStreamCopier();
while entries.hasMoreElements()
    entry = entries.nextElement();
    ename = char(entry.getName());
    if numel(ename) < 4 || ~strcmpi(ename(end-3:end), '.npy'), continue; end
    key = ename(1:end-4);
    if ~isKey(want, key), continue; end
    try
        is = zf.getInputStream(entry);
        baos = java.io.ByteArrayOutputStream();
        copier.copyStream(is, baos);
        is.close();
        bytes = typecast(baos.toByteArray(), 'uint8');
        val = parseNPY(bytes);
        if ~isempty(val) || ischar(val)
            S.(matlab.lang.makeValidName(key)) = val;
        end
    catch
    end
end
end

function val = parseNPY(bytes)
val = [];
bytes = bytes(:).';
if numel(bytes) < 10 || ~isequal(bytes(1:6), uint8([147 78 85 77 80 89]))
    return;
end
major = double(bytes(7));
if major >= 2
    hlen = double(typecast(uint8(bytes(9:12)), 'uint32'));  hstart = 13;
else
    hlen = double(typecast(uint8(bytes(9:10)), 'uint16'));  hstart = 11;
end
header = char(bytes(hstart:hstart + hlen - 1));
data_start = hstart + hlen;
descr = regexp(header, '''descr''\s*:\s*''([^'']*)''', 'tokens', 'once');
fortran = regexp(header, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
shape_tok = regexp(header, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
if isempty(descr), return; end
descr = descr{1};
fortran_order = ~isempty(fortran) && strcmp(fortran{1}, 'True');
if isempty(shape_tok), shape = [];
else, shape = str2double(regexp(shape_tok{1}, '-?\d+', 'match')); end
if any(descr(1) == '<>|='), bo = descr(1); rest = descr(2:end);
else, bo = '|'; rest = descr; end
kind = rest(1); isize = str2double(rest(2:end));
data = uint8(bytes(data_start:end));
switch kind
    case 'f'
        if isize == 8, cls = 'double'; elseif isize == 4, cls = 'single'; else, return; end
        vec = typecast(data, cls);
    case 'i', vec = typecast(data, sprintf('int%d', isize * 8));
    case 'u', vec = typecast(data, sprintf('uint%d', isize * 8));
    case 'b', vec = logical(data);
    case 'U'
        cps = typecast(data, 'uint32'); if bo == '>', cps = swapbytes(cps); end
        s = char(cps); s(s == 0) = []; val = s; return;
    case 'S'
        s = char(data); s(s == 0) = []; val = s; return;
    otherwise, return;
end
if bo == '>' && ~strcmp(kind, 'b'), vec = swapbytes(vec); end
if isempty(shape) || numel(shape) == 0
    val = vec(1);
elseif numel(shape) == 1
    val = reshape(vec, [shape 1]); val = val(:);
else
    if fortran_order, val = reshape(vec, shape);
    else, val = permute(reshape(vec, fliplr(shape)), numel(shape):-1:1); end
end
end
