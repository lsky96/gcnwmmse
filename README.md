# GCN-WMMSE
This library contains the code for the research paper "Coordinated Multicell MU-MIMO Beamforming Using Deep  WMMSE Algorithm Unrolling" [1](https://github.com/lsky96/gcnwmmse).

## Requirements
- Python 3.9
- PyTorch 1.10.0
- tensorboard 2.7.0
- numpy 1.21.4
- scipy 1.8.0
- matplotlib 3.5.0
- pandas 1.3.4
Newer versions should work as well.

## Description
### Example
An example script to perform a training run for a GCN-WMMSE network and a validation on a test set is provided in ```main.py```.
### Structure
- ```comm\trainer.py``` contains the class ```GeneralTrainer``` which abstracts model creation, training with checkpoints, model validation. See the docstring of the class constructor for possible arguments. Call the methods ```run_learning_upto_step(max_num_training_steps)``` to run training for ```max_num_training_steps``` and ```evaluate_on_test_set(test_data_path)``` to evaluate the model.

### DeepMIMO
The code supports the [DeepMIMO](https://deepmimo.net/) data set. The MATLAB scripts inside ```deepmimo\datagen```. The scripts require the DeepMIMO_Dataset_Generator class supplied in the DeepMIMO dataset.

## Usage
Please cite the paper "Schynol et al. - Coordinated Multicell MU-MIMO Beamforming Using Deep  WMMSE Algorithm Unrolling" [1](https://github.com/lsky96/gcnwmmse) if you apply this library in you own work.

## References
This paper provides extended PyTorch re-implementations for the algorithms and architectures of the following works:
- Shi et al. - 2011 - An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel
- Chowdhury et al. - 2020 - Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation
- Pellaco et al. - 2020 - Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems
- Hu et al. - 2021 - Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems
