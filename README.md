# Threshold Annealing in Spiking Neural Networks
This repo contains the corresponding code from the paper *The Fine Line Between Dead Neurons and Sparsity in Binarized Spiking Neural Networks*. 

## Requirements
A working `Python` (≥3.6) interpreter and the `pip` package manager. All required libraries and packages can be installed using  `pip install -r requirements.txt`. To avoid potential package conflicts, the use of a `conda` environment is recommended. The following commands can be used to create and activate a separate `conda` environment, clone this repository, and to install all dependencies:

```
conda create -n snn-tha python=3.8
conda activate snn-tha
git clone XXX
cd snn-tha
pip install -r requirements.txt
```

## Code Execution
To execute code, `cd` into one of three dataset directories, and then run `python run.py`. 

## Hyperparameter Tuning
* In each directory, `conf.py` defines all configuration parameters and hyperparameters for each dataset. The default parameters in this repo are identical to those for the high precision case reported in the corresponding paper.
* To run 4-bit quantized networks, set `"num_bits" : 4"` in `conf.py`. For optimized parameters, follow the values reported in the paper (to be linked upon completion of the double blind peer review process.)
