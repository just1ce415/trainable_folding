# Trainable AlphaFold 2 ported to Pytorch

We release our version of AlphaFold 2 written using Pytorch library. 

The repository provides a training script `alphafold/train.py`, which contains a similar to 
AlphaFold 2 training procedure.

We also provide a script for protein folding `alphafold/inference.py`, which can use 
AlphaFold 2 original parameters as well as our trained parameters.

##Inference with AlphaFold 2 original parameters

Go to `download_af2_parameters/` and follow the steps. This will download the parameters and convert
them to `pth` format, which can be used with `alphafold/inference.py` to fold proteins from MSAs. 
Check `examples/inference_1ezi_with_af2_params/` for an example of how to use AF2 parameters with 
our code.

##Toy training example

Follow the steps in `examples/training_toy_example/` to try training on a small dataset.