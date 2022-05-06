The test folder includes af2\_params\_to\_pth.py and predict.py.\
- af2\_params\_to\_pth.py is used for converting alphafold parameters to compatible parameters with our pytorch model.\
The script is written based on [Openfold](https://github.com/aqlaboratory/openfold/).\
It will produce a .pth file\
- predict.py is used to predict a structure given input features and model parameters. The script will produce 'test.pdb'.\
\
How to use:\
- python af2\_params\_to\_pth 'path\_to\_alphafold\_parameter\_file'.\
- python predict.py 'pth\_file'.\
Other files:\
- features.pkl: input features processed from 1a0b.cif\ 
- expected\_prediction.pdb: expected structure prediction. The test.pdb should be similar with this file.\
- All alphafold parameter files can be downloaded as instructed here: https://github.com/deepmind/alphafold
