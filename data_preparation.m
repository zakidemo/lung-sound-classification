% Data Preparation Script
% This script preprocesses the lung sound signals and generates spectrogram images.

input_folder = 'data/train';
output_folder = 'data/train_processed';
process_audio_vmd(input_folder, output_folder, 3); % 3 VMD modes

input_folder = 'data/test';
output_folder = 'data/test_processed';
process_audio_vmd(input_folder, output_folder, 3); % 3 VMD modes
