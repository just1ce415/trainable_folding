#Inference using AF2 generated features

##Usage
`python predict.py ../download_af2_parameters/params_model_3.pth`    

This will produce `test.pdb` which should be identical to `expected_prediction.pdb`

##Files
- `features.pkl`: Features produced by AlphaFold 2 from `1a0b.cif`
- `expected_prediction.pdb`: expected predicted structure.

