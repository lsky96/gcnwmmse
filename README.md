# GCN-WMMSE
This library contains the code for the research paper [Coordinated Multicell MU-MIMO Beamforming Using Deep WMMSE Algorithm Unrolling](https://arxiv.org/abs/2202.10371).

## Requirements
- Python 3.9
- PyTorch 1.10.0 (newer version might lead to errors)
- tensorboard 2.7.0
- numpy 1.21.4
- scipy 1.8.0
- matplotlib 3.5.0
- pandas 1.3.4

## Description
#### Quickstart
An example script to perform a training run for a GCN-WMMSE network and a validation on a test set is provided in ```main.py```.
#### Structure
- ```main``` is an examplary script to use the library.
- ```comm\trainer``` contains the class ```GeneralTrainer``` which abstracts model creation, training with checkpoints, model validation. See the docstring of the class constructor for possible arguments. Call the methods ```run_learning_upto_step(max_num_training_steps)``` to run training for ```max_num_training_steps``` and ```evaluate_on_test_set(test_data_path)``` to evaluate the model.
- ```comm\gcnwmmse``` supplies the architecture of this work.
- ```comm\algorithm``` contains various implementations of the WMMSE algorithm and various helper functions to generate baseline data.
- ```comm\reference_wmmse_unrolls``` supplies extended PyTorch implementations of architectures reported in reference works.
- ```comm\architectures``` supplies ```get_model(model_name, *args, **kwargs)``` to return any architecture of this library.
- ```comm\channel``` supplies functions to generate wireless scenarios (```mimoifc_randcn```, ```mimoifc_triangle```, ```siso_adhoc_2d```, ```deepmimo```) and to compute the achievable downlink rate given some beamformers. 
- ```comm\lossfun``` supplies loss functions for training.
- ```comm\mathutil``` supplies various helper functions used throughout the rest of the library.
- ```comm\network``` supplies a general GCN implementation.
- ```datawriter``` contains classes to export data to CSV-format.
- ```data``` serves as run and data directory for the example script.

#### DeepMIMO
The code supports the [DeepMIMO](https://deepmimo.net/) data set. The script  ```deepmimo_channel_generation``` generates pickled channel matrix data that can be imported by ```comm\channels.deepmimo```. The DeepMIMO package must be installed for the script. After creation of the channel matrix data, use the ```deepmimo``` data type/channel type and supply the path to the channel data to generate a scenario batch that can be used for training or testing.

## Usage
Please cite the paper 
L. Schynol and M. Pesavento, "Coordinated Sum-Rate Maximization in Multicell MU-MIMO with Deep Unrolling," in IEEE Journal on Selected Areas in Communications, (doi: 10.1109/JSAC.2023.3242716)
if you apply the provided code in you own work. If you use the reference architectures, please cite the respective works as well.

## References
This paper provides extended PyTorch implementations for the algorithms and architectures of the following works:
- Shi et al. - 2011 - An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel
- Chowdhury et al. - 2020 - Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation
- Pellaco et al. - 2020 - Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems
- Hu et al. - 2021 - Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems
