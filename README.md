# MVO-identification

This repository contains the source code for the MICCAI 2024 paper:  
**"Coarse-Grained Mask Regularization for Microvascular Obstruction Identification from non-contrast Cardiac Magnetic Resonance"**.


## Environment Setup

This codebase is tested with the following environment:

- **CUDA**: 11.0
- **PyTorch**: 1.7.1
- **MMSegmentation**: 0.13.0
- **Python**: 3.8+

To ensure compatibility, it is recommended to use the above versions.

# Multi-GPU Training
To train a model with multiple GPUs (e.g., 4 GPUs), use the following command:

```
./tools/dist_train.sh local_configs/your_model_config.py 4 --work-dir "path/to/save/weights"
```
- Replace local_configs/your_model_config.py with the path to your model configuration file.
- Replace 4 with the number of GPUs you want to use.
- Replace "path/to/save/weights" with the directory where you want to save the model weights and training logs.

# Inference
To run inference on a single GPU:

```
./tools/dist_test.sh  local_configs/your_model_config.py "your_weight_path" 4 --out "path/to/save/weights" 
```

- Replace local_configs/your_model_config.py with the path to your model configuration file.
- Replace 4 with the number of GPUs you want to use.
- Replace "path/to/save/weights" with the directory where you want to save the model weights and training logs.