function thermal_hologram_gui_export7_redo_FFT()
% Recompute one z plane from interferograms and display side-by-side XY
% amplitude and wrapped phase. Reconstruction matches HyperspectralAnalyzer.exe.
%
% Required fields: interferogram [z,x,y,scan], z_mm, wavelength_um, and
% interferogram_position_mm [z,scan]. Older files may use the auto-detected
% <mat-file-stem>_interferogram_positions_mm.tsv sidecar.

[mat_file, mat_path] = uigetfile('*.mat', 'Choose thermal hologram MAT file');
if isequal(mat_file, 0), return; end
mat_full_path = fullfile(mat_path, mat_file);
loaded = load(mat_full_path, 'thermal_hologram');
if ~isfield(loaded, 'thermal_hologram')
    error('The selected file does not contain thermal_hologram.');
end
H = loaded.thermal_hologram;
required_fields = {'interferogram', 'z_mm', 'wavelength_um'};
for field_index = 1:numel(required_fields)
    if ~isfield(H, required_fields{field_index})
        error('thermal_hologram is missing required field: %s', required_fields{field_index});
    end
end
if ~isnumeric(H.interferogram) || ndims(H.interferogram) ~= 4
    error('thermal_hologram.interferogram must be [z, x, y, scan].');
end
interferogram = H.interferogram;
z_mm = double(H.z_mm(:));
wavelength_um = double(H.wavelength_um(:));
n_z = size(interferogram, 1);
n_x = size(interferogram, 2);
n_y = size(interferogram, 3);
n_scan = size(interferogram, 4);
if numel(z_mm) ~= n_z
    error('Length of z_mm does not match the interferogram z dimension.');
end
if isempty(wavelength_um) || any(~isfinite(wavelength_um))
    error('wavelength_um must be nonempty and finite.');
end
[position_mm, position_source] = loadPositionAxes(H, mat_path, mat_file, z_mm, n_scan);

apod_width = 0.2;
expected_zpd_mm = 24.33;
search_mm = 0.1;
if isfield(H, 'metadata_json')
    try
        metadata_value = H.metadata_json;
        if iscell(metadata_value), metadata_value = metadata_value{1}; end
        if isstring(metadata_value), metadata_value = metadata_value(1); end
        metadata = jsondecode(char(metadata_value));
        if isfield(metadata, 'apod_width') && isfinite(metadata.apod_width)
            apod_width = double(metadata.apod_width);
        end
        if isfield(metadata, 'expected_zpd_mm') && isfinite(metadata.expected_zpd_mm)
            expected_zpd_mm = double(metadata.expected_zpd_mm);
        end
    catch
        warning('Could not parse metadata_json; using analyzer defaults.');
    end
end
clear loaded H;

[z_mm, z_order] = sort(z_mm, 'ascend');
if ~isequal(z_order(:), (1:n_z).')
    interferogram = interferogram(z_order, :, :, :);
    position_mm = position_mm(z_order, :);
end
wavelength_um = sort(wavelength_um, 'ascend');

script_dir = fileparts(mfilename('fullpath'));
calibration_file = fullfile(script_dir, 'Twins', 'ASRC calibration', 'parameters_cal.txt');
if ~isfile(calibration_file)
    error('Missing spectral calibration file: %s', calibration_file);
end
calibration = readmatrix(calibration_file, 'FileType', 'text', 'Delimiter', '\t');
if size(calibration, 1) ~= 2 && size(calibration, 2) == 2
    calibration = calibration.';
end
if size(calibration, 1) ~= 2
    error('parameters_cal.txt must contain two numeric rows.');
end
cal_wavelength_um = calibration(1, :);
cal_reciprocal = calibration(2, :);
valid_cal = isfinite(cal_wavelength_um) & isfinite(cal_reciprocal) & cal_wavelength_um ~= 0;
cal_inverse_wavelength = 1 ./ cal_wavelength_um(valid_cal);
cal_reciprocal = cal_reciprocal(valid_cal);
[cal_inverse_wavelength, cal_order] = sort(cal_inverse_wavelength, 'ascend');
cal_reciprocal = cal_reciprocal(cal_order);
[cal_inverse_wavelength, unique_index] = unique(cal_inverse_wavelength, 'stable');
cal_reciprocal = cal_reciprocal(unique_index);

state.z_index = max(1, round(n_z / 2));
state.wavelength_index = max(1, round(numel(wavelength_um) / 2));
cached_z_index = 0;
cached_signal = [];
cached_position_mm = [];
cached_center_index = [];
current_complex_map = [];
current_amplitude_map = [];
current_phase_map = [];
is_computing = false;
full_window_low = min(position_mm(:));
full_window_high = max(position_mm(:));

fig = uifigure('Name', 'Thermal Hologram FFT: XY Amplitude and Phase', ...
    'Color', 'w', 'Position', [60, 60, 1500, 850]);
main_grid = uigridlayout(fig, [3, 2]);
main_grid.RowHeight = {32, '1x', 190};
main_grid.ColumnWidth = {'1x', '1x'};
main_grid.Padding = [10, 10, 10, 10];
main_grid.RowSpacing = 8;
main_grid.ColumnSpacing = 8;
info_label = uilabel(main_grid);
info_label.Layout.Row = 1;
info_label.Layout.Column = [1, 2];
info_label.FontWeight = 'bold';
info_label.Text = sprintf('%s | interferogram [z,x,y,scan] = [%d,%d,%d,%d]', ...
    mat_file, n_z, n_x, n_y, n_scan);
ax_amplitude = uiaxes(main_grid);
ax_amplitude.Layout.Row = 2;
ax_amplitude.Layout.Column = 1;
ax_phase = uiaxes(main_grid);
ax_phase.Layout.Row = 2;
ax_phase.Layout.Column = 2;
control_panel = uipanel(main_grid, 'Title', 'Reconstruction and export');
control_panel.Layout.Row = 3;
control_panel.Layout.Column = [1, 2];
control_grid = uigridlayout(control_panel, [4, 8]);
control_grid.RowHeight = {28, 28, 32, '1x'};
control_grid.ColumnWidth = {105, '1x', '1x', 120, 105, '1x', '1x', 120};
control_grid.Padding = [8, 8, 8, 8];
control_grid.RowSpacing = 6;
control_grid.ColumnSpacing = 8;

z_label = uilabel(control_grid, 'Text', 'z position');
z_label.Layout.Row = 1; z_label.Layout.Column = 1;
z_slider = uislider(control_grid);
z_slider.Layout.Row = 1; z_slider.Layout.Column = [2, 3];
z_value_label = uilabel(control_grid);
z_value_label.Layout.Row = 1; z_value_label.Layout.Column = 4;
z_value_label.HorizontalAlignment = 'right';
wavelength_label = uilabel(control_grid, 'Text', 'Wavelength');
wavelength_label.Layout.Row = 1; wavelength_label.Layout.Column = 5;
wavelength_slider = uislider(control_grid);
wavelength_slider.Layout.Row = 1; wavelength_slider.Layout.Column = [6, 7];
wavelength_value_label = uilabel(control_grid);
wavelength_value_label.Layout.Row = 1; wavelength_value_label.Layout.Column = 8;
wavelength_value_label.HorizontalAlignment = 'right';

resolution_label = uilabel(control_grid, 'Text', 'XY resolution');
resolution_label.Layout.Row = 2; resolution_label.Layout.Column = 1;
resolution_field = uieditfield(control_grid, 'numeric');
resolution_field.Layout.Row = 2; resolution_field.Layout.Column = 2;
resolution_field.Value = 1.0; resolution_field.Limits = [eps, Inf];
resolution_field.ValueDisplayFormat = '%.6g';
resolution_units = uilabel(control_grid, 'Text', 'um/pixel');
resolution_units.Layout.Row = 2; resolution_units.Layout.Column = 3;
window_low_label = uilabel(control_grid, 'Text', 'FFT window low');
window_low_label.Layout.Row = 2; window_low_label.Layout.Column = 4;
window_low_field = uieditfield(control_grid, 'numeric');
window_low_field.Layout.Row = 2; window_low_field.Layout.Column = 5;
window_low_field.Value = full_window_low; window_low_field.ValueDisplayFormat = '%.6f';
window_high_label = uilabel(control_grid, 'Text', 'FFT window high');
window_high_label.Layout.Row = 2; window_high_label.Layout.Column = 6;
window_high_field = uieditfield(control_grid, 'numeric');
window_high_field.Layout.Row = 2; window_high_field.Layout.Column = 7;
window_high_field.Value = full_window_high; window_high_field.ValueDisplayFormat = '%.6f';
window_units = uilabel(control_grid, 'Text', 'mm');
window_units.Layout.Row = 2; window_units.Layout.Column = 8;

threshold_label = uilabel(control_grid, 'Text', 'Phase threshold');
threshold_label.Layout.Row = 3; threshold_label.Layout.Column = 1;
threshold_field = uieditfield(control_grid, 'numeric');
threshold_field.Layout.Row = 3; threshold_field.Layout.Column = 2;
threshold_field.Value = 5.0; threshold_field.Limits = [0, 100];
threshold_field.ValueDisplayFormat = '%.3g';
threshold_units = uilabel(control_grid, 'Text', '% of max amplitude');
threshold_units.Layout.Row = 3; threshold_units.Layout.Column = 3;
full_window_button = uibutton(control_grid, 'push', 'Text', 'Full FFT window');
full_window_button.Layout.Row = 3; full_window_button.Layout.Column = 4;
center_window_button = uibutton(control_grid, 'push', 'Text', 'Center +/-0.2 mm');
center_window_button.Layout.Row = 3; center_window_button.Layout.Column = 5;
copy_button = uibutton(control_grid, 'push', 'Text', 'Copy TXT');
copy_button.Layout.Row = 3; copy_button.Layout.Column = 6;
save_button = uibutton(control_grid, 'push', 'Text', 'Save TXT');
save_button.Layout.Row = 3; save_button.Layout.Column = 7;
status_label = uilabel(control_grid);
status_label.Layout.Row = 4; status_label.Layout.Column = [1, 8];
status_label.WordWrap = 'on'; status_label.Text = sprintf('Position axis: %s', position_source);

configureIndexSlider(z_slider, z_mm, '%.4f');
configureIndexSlider(wavelength_slider, wavelength_um, '%.6f');
z_slider.Value = state.z_index;
wavelength_slider.Value = state.wavelength_index;
x_um = getCenteredCoordinate(n_x, resolution_field.Value);
y_um = getCenteredCoordinate(n_y, resolution_field.Value);
initial_map = zeros(n_y, n_x);
h_amplitude = imagesc(ax_amplitude, [x_um(1), x_um(end)], [y_um(1), y_um(end)], initial_map);
h_phase = imagesc(ax_phase, [x_um(1), x_um(end)], [y_um(1), y_um(end)], initial_map);
axis(ax_amplitude, 'image'); axis(ax_phase, 'image');
ax_amplitude.YDir = 'normal'; ax_phase.YDir = 'normal';
ax_phase.Color = [0.75, 0.75, 0.75];
colormap(ax_amplitude, 'turbo'); colormap(ax_phase, 'hsv');
colorbar(ax_amplitude); colorbar(ax_phase); clim(ax_phase, [-pi, pi]);
xlabel(ax_amplitude, 'x (um)'); ylabel(ax_amplitude, 'y (um)');
xlabel(ax_phase, 'x (um)'); ylabel(ax_phase, 'y (um)');

z_slider.ValueChangedFcn = @(src, event) onZChanged(round(src.Value));
wavelength_slider.ValueChangedFcn = @(src, event) onWavelengthChanged(round(src.Value));
window_low_field.ValueChangedFcn = @(src, event) recomputeSelectedMap();
window_high_field.ValueChangedFcn = @(src, event) recomputeSelectedMap();
threshold_field.ValueChangedFcn = @(src, event) updateDisplayMaps();
resolution_field.ValueChangedFcn = @(src, event) updateSpatialAxes();
full_window_button.ButtonPushedFcn = @(src, event) setFullWindow();
center_window_button.ButtonPushedFcn = @(src, event) setCenterWindow();
copy_button.ButtonPushedFcn = @(src, event) copyCurrentData();
save_button.ButtonPushedFcn = @(src, event) saveCurrentData();
recomputeSelectedMap();

    function recomputeSelectedMap()
        if is_computing, return; end
        is_computing = true;
        fig.Pointer = 'watch';
        try
            prepareSelectedZPlane();
            low_mm = min(window_low_field.Value, window_high_field.Value);
            high_mm = max(window_low_field.Value, window_high_field.Value);
            window_low_field.Value = low_mm;
            window_high_field.Value = high_mm;
            in_window = cached_position_mm >= low_mm & cached_position_mm <= high_mm;
            if nnz(in_window) < 3
                error('The FFT window must contain at least three scan positions.');
            end

            sigma = abs(cached_position_mm(end) - cached_position_mm(1)) * apod_width;
            if sigma > 0
                gaussian = exp(-((cached_position_mm - ...
                    cached_position_mm(cached_center_index)).^2) ./ (2 * sigma^2));
            else
                gaussian = ones(n_scan, 1);
            end
            center_position = cached_position_mm(cached_center_index);
            if low_mm <= center_position && center_position <= high_mm
                apodization = gaussian .* double(in_window);
            else
                % Original analyzer uses a rectangular window for a tail-only region.
                apodization = double(in_window);
            end

            selected_wavelength = wavelength_um(state.wavelength_index);
            reciprocal_frequency = interp1(cal_inverse_wavelength, cal_reciprocal, ...
                1 / selected_wavelength, 'linear', 'extrap');
            position_step = diff(cached_position_mm);
            position_step = [position_step; position_step(end)];
            integration_kernel = position_step .* apodization .* ...
                exp(2i * pi * cached_position_mm * reciprocal_frequency);
            signal_flat = reshape(cached_signal, n_scan, []);
            spectrum_flat = integration_kernel.' * signal_flat;
            current_complex_map = reshape(spectrum_flat, n_x, n_y).';
            updateDisplayMaps();
        catch ME
            status_label.Text = sprintf('Reconstruction failed: %s', ME.message);
            uialert(fig, ME.message, 'Reconstruction error');
        end
        is_computing = false;
        if isvalid(fig), fig.Pointer = 'arrow'; end
    end

    function prepareSelectedZPlane()
        if cached_z_index == state.z_index && ~isempty(cached_signal), return; end
        status_label.Text = sprintf('Preparing z = %.4f mm: subtracting moving baseline...', ...
            z_mm(state.z_index));
        drawnow;
        raw = permute(interferogram(state.z_index, :, :, :), [4, 2, 3, 1]);
        raw = reshape(double(raw), n_scan, n_x, n_y);
        baseline_window = max(1, floor(n_scan / 5));
        if baseline_window == 1
            baseline = raw;
        else
            left_padding = floor(baseline_window / 2);
            right_padding = baseline_window - left_padding - 1;
            padded = cat(1, repmat(raw(1, :, :), left_padding, 1, 1), raw, ...
                repmat(raw(end, :, :), right_padding, 1, 1));
            averaging_kernel = reshape(ones(baseline_window, 1) / baseline_window, ...
                [baseline_window, 1, 1]);
            baseline = convn(padded, averaging_kernel, 'valid');
            clear padded;
        end
        cached_signal = raw - baseline;
        clear raw baseline;
        cached_position_mm = position_mm(state.z_index, :).';
        summed_interferogram = squeeze(sum(sum(cached_signal, 3), 2));
        cached_center_index = findCenterBurst( ...
            summed_interferogram, cached_position_mm, expected_zpd_mm, search_mm);
        cached_z_index = state.z_index;
    end

    function center_index = findCenterBurst(signal, positions, expected_zero, search_half_width)
        signal = double(signal(:));
        signal = signal - mean(signal);
        n = numel(signal);
        analytic_multiplier = zeros(n, 1);
        analytic_multiplier(1) = 1;
        if mod(n, 2) == 0
            analytic_multiplier(2:n/2) = 2;
            analytic_multiplier(n/2 + 1) = 1;
        else
            analytic_multiplier(2:(n + 1)/2) = 2;
        end
        envelope = abs(ifft(fft(signal) .* analytic_multiplier));
        search_mask = abs(positions - expected_zero) <= search_half_width;
        if ~any(search_mask), search_mask = true(n, 1); end
        candidate_indices = find(search_mask);
        [~, local_index] = max(envelope(search_mask));
        center_index = candidate_indices(local_index);
    end

    function updateDisplayMaps()
        if isempty(current_complex_map), return; end
        current_amplitude_map = abs(current_complex_map);
        current_phase_map = angle(current_complex_map);
        maximum_amplitude = max(current_amplitude_map(:));
        threshold_fraction = threshold_field.Value / 100;
        if isfinite(maximum_amplitude) && maximum_amplitude > 0
            current_phase_map(current_amplitude_map < ...
                threshold_fraction * maximum_amplitude) = NaN;
        end
        h_amplitude.CData = current_amplitude_map;
        h_phase.CData = current_phase_map;
        h_phase.AlphaData = ~isnan(current_phase_map);
        clim(ax_amplitude, 'auto');
        clim(ax_phase, [-pi, pi]);
        updateSpatialAxes();
        updateLabelsAndTitles();
    end

    function updateSpatialAxes()
        pixel_um = resolution_field.Value;
        if ~isfinite(pixel_um) || pixel_um <= 0, return; end
        x_coordinates = getCenteredCoordinate(n_x, pixel_um);
        y_coordinates = getCenteredCoordinate(n_y, pixel_um);
        h_amplitude.XData = [x_coordinates(1), x_coordinates(end)];
        h_amplitude.YData = [y_coordinates(1), y_coordinates(end)];
        h_phase.XData = [x_coordinates(1), x_coordinates(end)];
        h_phase.YData = [y_coordinates(1), y_coordinates(end)];
        ax_amplitude.XLim = [x_coordinates(1), x_coordinates(end)];
        ax_amplitude.YLim = [y_coordinates(1), y_coordinates(end)];
        ax_phase.XLim = [x_coordinates(1), x_coordinates(end)];
        ax_phase.YLim = [y_coordinates(1), y_coordinates(end)];
    end

    function updateLabelsAndTitles()
        selected_z = z_mm(state.z_index);
        selected_wavelength = wavelength_um(state.wavelength_index);
        center_position = cached_position_mm(cached_center_index);
        z_value_label.Text = sprintf('%d/%d, %.4f mm', state.z_index, n_z, selected_z);
        wavelength_value_label.Text = sprintf('%d/%d, %.6f um', ...
            state.wavelength_index, numel(wavelength_um), selected_wavelength);
        title(ax_amplitude, sprintf('XY amplitude | z = %.4f mm | wavelength = %.6f um', ...
            selected_z, selected_wavelength), 'Interpreter', 'none');
        title(ax_phase, sprintf('XY wrapped phase [-pi, pi] | threshold = %.3g%%', ...
            threshold_field.Value), 'Interpreter', 'none');
        status_label.Text = sprintf([ ...
            'Calibrated DFT complete | center burst %.6f mm | window %.6f to %.6f mm | ', ...
            'Gaussian width %.4g | phase below %.3g%% max amplitude is masked | axis: %s'], ...
            center_position, window_low_field.Value, window_high_field.Value, ...
            apod_width, threshold_field.Value, position_source);
    end

    function onZChanged(new_index)
        state.z_index = clampIndex(new_index, n_z);
        z_slider.Value = state.z_index;
        recomputeSelectedMap();
    end

    function onWavelengthChanged(new_index)
        state.wavelength_index = clampIndex(new_index, numel(wavelength_um));
        wavelength_slider.Value = state.wavelength_index;
        recomputeSelectedMap();
    end

    function setFullWindow()
        window_low_field.Value = min(position_mm(state.z_index, :));
        window_high_field.Value = max(position_mm(state.z_index, :));
        recomputeSelectedMap();
    end

    function setCenterWindow()
        try
            prepareSelectedZPlane();
            center_position = cached_position_mm(cached_center_index);
            window_low_field.Value = center_position - 0.2;
            window_high_field.Value = center_position + 0.2;
            recomputeSelectedMap();
        catch ME
            uialert(fig, ME.message, 'Window error');
        end
    end

    function [matrix, header] = makeExportMatrix()
        if isempty(current_amplitude_map) || isempty(current_phase_map)
            error('No reconstructed map is available to export.');
        end
        x_coordinates = getCenteredCoordinate(n_x, resolution_field.Value);
        y_coordinates = getCenteredCoordinate(n_y, resolution_field.Value);
        [X, Y] = meshgrid(x_coordinates, y_coordinates);
        matrix = [X(:), Y(:), current_amplitude_map(:), current_phase_map(:)];
        header = sprintf('x_um\ty_um\tamplitude\tphase_rad');
    end

    function copyCurrentData()
        try
            [matrix, header] = makeExportMatrix();
            clipboard('copy', matrixToTXT(matrix, header));
            status_label.Text = sprintf('Copied %d rows: x, y, amplitude, phase.', size(matrix, 1));
        catch ME
            uialert(fig, ME.message, 'Copy error');
        end
    end

    function saveCurrentData()
        try
            [matrix, header] = makeExportMatrix();
            [txt_file, txt_path] = uiputfile('*.txt', ...
                'Save XY amplitude and phase as TXT', fullfile(mat_path, makeDefaultFilename()));
            if isequal(txt_file, 0), return; end
            writeMatrixTXT(fullfile(txt_path, txt_file), matrix, header);
            status_label.Text = sprintf('Saved %d rows to %s', ...
                size(matrix, 1), fullfile(txt_path, txt_file));
        catch ME
            uialert(fig, ME.message, 'Save error');
        end
    end

    function filename = makeDefaultFilename()
        [~, base_name, ~] = fileparts(mat_file);
        z_tag = strrep(sprintf('%.4f', z_mm(state.z_index)), '.', 'p');
        wavelength_tag = strrep(sprintf('%.6f', wavelength_um(state.wavelength_index)), '.', 'p');
        filename = sprintf('%s_XY_amp_phase_z%smm_lambda%sum.txt', ...
            base_name, z_tag, wavelength_tag);
    end

    function text = matrixToTXT(matrix, header)
        body = sprintf('%.10g\t%.10g\t%.10g\t%.10g\n', matrix.');
        text = [header, newline, body];
    end

    function writeMatrixTXT(filename, matrix, header)
        fid = fopen(filename, 'w');
        if fid < 0, error('Could not open file for writing: %s', filename); end
        cleanup_file = onCleanup(@() fclose(fid));
        fprintf(fid, '%s', matrixToTXT(matrix, header));
        clear cleanup_file;
    end
end

function [position_mm, source_label] = loadPositionAxes(H, mat_path, mat_file, z_mm, n_scan)
n_z = numel(z_mm);
if isfield(H, 'interferogram_position_mm')
    position_mm = double(H.interferogram_position_mm);
    source_label = 'embedded interferogram_position_mm';
    if isfield(H, 'interferogram_position_is_calibrated')
        calibrated = logical(H.interferogram_position_is_calibrated(:));
        if numel(calibrated) ~= n_z || ~all(calibrated)
            error('The embedded interferogram position axis is not calibrated at every z.');
        end
    end
else
    [~, base_name, ~] = fileparts(mat_file);
    sidecar = fullfile(mat_path, [base_name, '_interferogram_positions_mm.tsv']);
    if ~isfile(sidecar)
        error(['The MAT file has an interferogram but no calibrated scan-position axis.\n', ...
            'Expected embedded field interferogram_position_mm or sidecar:\n%s'], sidecar);
    end
    sidecar_data = readmatrix(sidecar, 'FileType', 'text', ...
        'Delimiter', '\t', 'NumHeaderLines', 1);
    if size(sidecar_data, 2) ~= n_scan + 1
        error('The position sidecar must contain z_mm plus %d scan positions.', n_scan);
    end
    sidecar_z = sidecar_data(:, 1);
    sidecar_position = sidecar_data(:, 2:end);
    position_mm = zeros(n_z, n_scan);
    for z_index = 1:n_z
        [distance, match_index] = min(abs(sidecar_z - z_mm(z_index)));
        if distance > 1e-7
            error('No calibrated position row matches z = %.8g mm.', z_mm(z_index));
        end
        position_mm(z_index, :) = sidecar_position(match_index, :);
    end
    source_label = ['sidecar ', base_name, '_interferogram_positions_mm.tsv'];
end
if isequal(size(position_mm), [n_scan, n_z]), position_mm = position_mm.'; end
if ~isequal(size(position_mm), [n_z, n_scan])
    error('interferogram_position_mm must have shape [z, scan] = [%d, %d].', n_z, n_scan);
end
if any(~isfinite(position_mm(:)))
    error('The interferogram position axis contains nonfinite values.');
end
if any(any(diff(position_mm, 1, 2) <= 0))
    error('Each interferogram position axis must be strictly increasing.');
end
end

function coordinate = getCenteredCoordinate(n_pixels, pixel_um)
coordinate = ((1:n_pixels) - (n_pixels + 1) / 2) * pixel_um;
end

function index = clampIndex(index, count)
index = max(1, min(count, round(index)));
end

function configureIndexSlider(slider, values, format_string)
count = numel(values);
slider.Limits = [1, max(2, count)];
if count == 1
    slider.Enable = 'off';
    slider.MajorTicks = 1;
    slider.MajorTickLabels = {sprintf(format_string, values(1))};
else
    ticks = unique(round(linspace(1, count, min(6, count))));
    slider.MajorTicks = ticks;
    slider.MajorTickLabels = arrayfun(@(index) sprintf(format_string, values(index)), ...
        ticks, 'UniformOutput', false);
end
end
