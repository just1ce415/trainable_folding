import torch
import sys
import math

from alphadock import docker
from alphadock import config
from alphadock import utils
from alphadock import all_atom
from copy import deepcopy
import torchvision
import pickle


def pred_to_pdb(out_pdb, input_dict, out_dict):
    with open(out_pdb, 'w') as f:
        f.write(f'test.pred\n')
        serial = all_atom.atom14_to_pdb_stream(
            f,
            input_dict['target']['rec_aatype'][0].cpu(),
            out_dict['final_all_atom']['atom_pos_tensor'].detach().cpu(),
            chain='A',
            serial_start=1,
            resnum_start=1
        )


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(123456)
    with open('features.pkl', 'rb') as f:
        inputs = pickle.load(f)

    model = docker.DockerIteration(config.config['model'], config.config)
    model.load_state_dict(torch.load(sys.argv[1])['model_state_dict'])
    model.eval()
    num_recycles = config.config['model']['recycling_num_iter'] if config.config['model']['recycling_on'] else 1
    with torch.no_grad():
        for recycle_iter in range(num_recycles):
            output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)
    pred_to_pdb("prediction.pdb", inputs, output)

