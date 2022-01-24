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
To execute code, `cd` into one of four dataset directories, and then run `python run.py`. 

## Hyperparameter Tuning
* In each directory, `conf.py` defines all configuration parameters and hyperparameters for each dataset. The default parameters in this repo are identical to those for the high precision case reported in the corresponding paper.
* To run binarized networks, set `"binarize" : True"` in `conf.py`. For optimized parameters, follow the values reported in the paper (to be linked upon completion of the double blind peer review process.)


# Temporal Coding
Section 4 of the paper demonstrates the use of threshold annealing in a spike-timing task. A fully connected network of structure 100-1000-1 is used, where a Poisson spike train is pass at the input, and the output neuron is trained to spike at $t=75$ by linearly ramping up the membrane potential over time using a mean square error loss at each time step:

<img src="https://user-images.githubusercontent.com/40262130/150854145-d90d61ed-c41b-4aea-ad16-e077044f4f90.png" width="1200">

![temporal_code](https://user-images.githubusercontent.com/40262130/150854145-d90d61ed-c41b-4aea-ad16-e077044f4f90.png)

The animated versions of the above figures are provided below, and can be reproduced in the corresponding notebook.

## Animations

### High Precision Weights, Normalized Threshold <img src="https://render.githubusercontent.com/render/math?math=\theta=1">

This is the optimal baseline, showing that it is a reasonably straightforward task to achieve.

https://user-images.githubusercontent.com/40262130/150855093-4cdaa55b-7cad-4d5a-b5fa-9e482c6fe07e.mp4

### Binarized Weights, Normalized Threshold <img src="https://render.githubusercontent.com/render/math?math=\theta=1">
The results become significantly unstable when binarizing weights.

https://user-images.githubusercontent.com/40262130/150855727-9ccfcca2-8b48-48cc-b5df-0d17f367968c.mp4

A moving average over training iterations is used in an attempt to clean up the above plot, but the results remain senseless:

https://user-images.githubusercontent.com/40262130/150855822-02d9177c-e08f-48f4-8753-d5c937e49c00.mp4

### Binarized Weights, Large Threshold <img src="https://render.githubusercontent.com/render/math?math=\theta=50">
Increasing the threshold of all neurons provides a higher dynamic range state-space. But increasing the threshold too high leads to the dead neuron problem. The animation below shows how spiking activity has been suppressed; the flat membrane potential is purely a result of the bias.

https://user-images.githubusercontent.com/40262130/150856229-0a3ae7ce-5670-4545-b13c-06dd3ca992f3.mp4

### Binarized Weights, Threshold Annealing <img src="https://render.githubusercontent.com/render/math?math=\theta: 5 \rightarrow 50">
Now apply threshold annealing to use an evolving neuronal state-space to gradually lift spiking activity. This avoids the dead neuron problem in the large threshold case, and avoids the instability/memory leakage in the normalized threshold case.

https://user-images.githubusercontent.com/40262130/150856483-f53f2156-4348-46da-9c0f-5f05f31cf677.mp4

This now looks far more functional than all previous binarized cases. 
We can take a moving average to smooth out the impact of sudden reset dynamics. Although not as perfect as the high precision case, the binarized SNN continues to learn despite the excessively high final threshold.

https://user-images.githubusercontent.com/40262130/150856726-aedb1d08-fe61-4b32-a3aa-6dcc9c76311a.mp4

