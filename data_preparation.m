%--------------- Preparation
clear all; close all; clc;

% Description:
% This section of the script defines the input and output folders for the data preparation process.
% The input_folder specifies the location of the downsampled audio files used for testing.
% The output_folder specifies the location where the generated spectrogram images will be saved.
input_folder = 'E:\Revised_AI_Paper_CQT\Datbase\RALE_SOUNDSLIBARARY\DATA_Used_Downsampled\test\Wheeze';
output_folder = 'E:\Revised_AI_Paper_CQT\Datbase\RALE_SOUNDSLIBARARY\CQT_Images\test\Wheeze';


% Get all audio files in the input folder
audio_files = dir(fullfile(input_folder, '*.wav'));

% Parameters for VMD and CQT
alpha = 2000;  % Moderate bandwidth constraint
tau = 0;  % Noise-tolerance (no strict fidelity enforcement)
K = 3;  % 3 modes
DC = 0;  % No DC part imposed
init = 1;  % Initialize omegas uniformly
tol = 1e-7;
frequency_resolution = 2;
minimum_frequency = 55;
maximum_frequency = 4000;
time_resolution = 25;

% Loop through each audio file
for i = 1:length(audio_files)
    % Read the audio file
    [audio_signal, fs] = audioread(fullfile(input_folder, audio_files(i).name));
    audio_signal = audio_signal(:)'; % Ensure audio_signal is a row vector
    
    % Run VMD to get the third IMF
    try
        [u, ~, ~] = VMD(audio_signal, alpha, tau, K, DC, init, tol);
        x = u(3,:);

        % Compute the CQT kernel
        cqt_kernel = zaf.cqtkernel(fs, frequency_resolution, minimum_frequency, maximum_frequency);

        % Compute the (magnitude) CQT spectrogram using the kernel
        audio_spectrogram = zaf.cqtspectrogram(x', fs, time_resolution, cqt_kernel);

        % Display the CQT spectrogram in dB, seconds, and Hz
        xtick_step = 1;
        figure('Visible', 'off');
        zaf.cqtspecshow(audio_spectrogram, time_resolution, frequency_resolution, minimum_frequency, xtick_step);

        % Remove axis and colorbar for AlexNet input
        axis off;
        colorbar off;

        % Save the spectrogram image
        [~, name, ~] = fileparts(audio_files(i).name);
        output_file_name = fullfile(output_folder, [name, '.png']);
        saveas(gcf, output_file_name);
        close(gcf);
    catch ME
        fprintf('Error processing file %s: %s\n', audio_files(i).name, ME.message);
    end
end
