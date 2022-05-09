The test folder includes `af2_params_to_pth.py` and `predict.py`.
- `af2_params_to_pth.py` is used for converting alphafold parameters to compatible parameters with our pytorch model.
The script is written based on [Openfold](https://github.com/aqlaboratory/openfold/).
It will produce a `.pth` file
- `predict.py` is used to predict a structure given input features and model parameters. The script will produce `test.pdb`.

How to use:
1. `python af2_params_to_pth.py params_model_3.npz`.
2. `python predict.py params_model_3.pth`.

Other files:
- `features.pkl`: input features processed from `1a0b.cif`
- `expected_prediction.pdb`: expected structure prediction. The `test.pdb` should be similar to this file.
- All alphafold parameter files can be downloaded [here](https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar)
