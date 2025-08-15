%% DAS Beamformer Implementation
clc; clear; close all;

% Define parameters
N = 59;
f = 5000;

lambda = 343 / 5000;      % Wavelength in meters (0.0686 m)

dx = lambda/2; % Microphone spacing 
% dx = lambda;
mic_positions = (0:N-1) * dx;  

k = 2 * pi / lambda; % Wavenumber

%Single source
% theta_true = 10;

%multiple sources
theta_true = [15, 17]; % Angles of two sources in degrees 

theta_rad = deg2rad(theta_true); % Convert to radians
K = length(theta_true); % Number of sources

% Reshape vectors to align dimensions for broadcasting
mic_positions = mic_positions(:); % N x 1 column vector
theta_rad = theta_rad(:)'; % 1 x K row vector

% Compute phase shifts matrix
% phase_shifts N x K matrix
phase_shifts = -k * mic_positions * sin(theta_rad); % N x K matrix

% Compute received signals from all sources
% Each column corresponds to a source
received_signals = exp(1j * phase_shifts); % N x K matrix

% Sum the signals across sources for each microphone
received_signal = sum(received_signals, 2); % N x 1 vector

% Add noise
sigma_noise = 0.1;
noise = sqrt(sigma_noise/2) * (randn(N, 1) + 1j * randn(N,1));
received_signal = received_signal + noise;


% Define scanning angles
theta_scan = -90:0.1:90; % degrees
theta_scan_rad = deg2rad(theta_scan);  

% Initialize array to store beamforming output
beamforming_output = zeros(length(theta_scan), 1);

% Loop over scanning angles
for idx = 1:length(theta_scan_rad)
    % Compute steering vector for the scanning angle
    steering_vector = exp(-1j * k * mic_positions * sin(theta_scan_rad(idx)));
    
    % Compute beamformer output (power)
    beamforming_output(idx) = abs(steering_vector' * received_signal)^2;
end

% Normalize the beamforming output
beamforming_output = beamforming_output / max(beamforming_output);

% Find peaks in the beamforming output
[pks, locs] = findpeaks(beamforming_output, 'MinPeakHeight', 0.5, 'SortStr', 'descend', 'NPeaks', K);

% Estimated angles
theta_est = theta_scan(locs);

% Sort the estimated angles
theta_est = sort(theta_est);

% Plot the Beamforming Spectrum

figure;
plot(theta_scan, beamforming_output, 'LineWidth', 1.5);
xlabel('Angle (degrees)');
ylabel('Normalized Power');
title('Beamforming Output');
grid on;
hold on;
plot(theta_est, pks, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
legend('Beamforming Output', 'Estimated Angles');

% Output Results
fprintf('True Angles: %.2f degrees\n', theta_true);
fprintf('\n--- DAS Beamforming Results ---\n');
fprintf('Estimated Angles: %.2f degrees\n', theta_est);


%% MUSIC Algorithm Implementation

% Parameters
L = 100;                   % Number of snapshots
sigma_noise = 0.1;         % Noise power
K = length(theta_true);     % Number of sources

% Generate signal snapshots with random amplitudes
received_signals_snapshots = zeros(N, L);

for l = 1:L
    % Generate K random complex source signals for this snapshot
    source_amplitudes = (randn(K, 1) + 1j * randn(K, 1));  % K x 1
    
    % Simulate signal = A * s + noise
    signal = received_signals * source_amplitudes;        % N x 1
    noise = sqrt(sigma_noise/2) * (randn(N, 1) + 1j * randn(N, 1)); % N x 1
    
    received_signals_snapshots(:, l) = signal + noise;
end

% Estimate covariance matrix (N x N)
Rxx = (received_signals_snapshots * received_signals_snapshots') / L;

% Eigen-decomposition of covariance matrix
[Evec, Eval] = eig(Rxx);
[evals_sorted, idx] = sort(diag(Eval), 'descend');  % Sort eigenvalues
Evec = Evec(:, idx);  % Sort eigenvectors accordingly

% Separate signal and noise subspaces
En = Evec(:, K+1:end);  % Noise subspace (N x (N-K))

% MUSIC spectrum computation
music_spectrum = zeros(size(theta_scan));

for idx = 1:length(theta_scan_rad)
    a_theta = exp(-1j * k * mic_positions * sin(theta_scan_rad(idx)));  % N x 1
    music_spectrum(idx) = 1 / abs(a_theta' * (En * En') * a_theta);
end

% Normalize the MUSIC spectrum
music_spectrum = music_spectrum / max(music_spectrum);

% Find peaks in the MUSIC spectrum
[pks_music, locs_music] = findpeaks(music_spectrum, ...
    'MinPeakHeight', 0.5, ...
    'SortStr', 'descend', ...
    'NPeaks', K);

% Estimated angles (MUSIC)
theta_est_music = sort(theta_scan(locs_music));

% Plot the MUSIC Spectrum
figure;
plot(theta_scan, music_spectrum, 'LineWidth', 1.5);
xlabel('Angle (degrees)');
ylabel('Normalized Spectrum');
title('MUSIC Spectrum');
grid on;
hold on;
plot(theta_est_music, pks_music, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('MUSIC Spectrum', 'Estimated Angles');

% Print MUSIC results
fprintf('\n--- MUSIC Results ---\n');
fprintf('Estimated Angles: %.2f degrees\n', theta_est_music);
