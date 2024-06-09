# Pulmonary Disease Classification

This repository contains the code for the project titled "Leveraging CQT-VMD and Pre-trained AlexNet Architecture for Accurate Pulmonary Disease Classification from Lung Sound Signals".

## Files

1. **data_preparation.m**: This script preprocesses the lung sound signals and generates spectrogram images using CQT and VMD.
2. **VMD.m**: This script contains the implementation of the Variational Mode Decomposition (VMD) algorithm.
3. **alexnet_training.m**: This script trains an AlexNet model for pulmonary disease classification using the generated spectrogram images.

## Usage

1. **Data Preparation**: Run `data_preparation.m` to preprocess the lung sound signals and generate the spectrogram images.
2. **Training**: Run `alexnet_training.m` to train the AlexNet model using the preprocessed data.
3. **Evaluation**: The trained model will be evaluated and the test accuracy will be displayed.

## References

- K. Dragomiretskiy, D. Zosso, "Variational Mode Decomposition," IEEE Trans. on Signal Processing (in press).(http://dx.doi.org/10.1109/TSP.2013.2288675)
