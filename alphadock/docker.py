import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import sys
from copy import copy

from alphadock import modules
from alphadock import structure
from alphadock import all_atom
from alphadock import loss
from alphadock import utils


def flatten_input(input, output=[], path=''):
    '''
    Wrote this to use in hooks but it turned out to be too slow
    '''
    if isinstance(input, tuple) or isinstance(input, list):
        for i, x in enumerate(input):
            flatten_input(x, output, path + '.' + str(i))
    if isinstance(input, dict):
        for k, v in input.items():
            flatten_input(v, output, path + '.' + str(k))
    if isinstance(input, torch.Tensor):
        output += [(path, input)]
    return output


class DockerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InputEmbedder = modules.InputEmbedder(config['InputEmbedder'], global_config)
        self.Evoformer = nn.ModuleList([modules.EvoformerIteration(config['Evoformer']['EvoformerIteration'], global_config) for x in range(config['Evoformer']['num_iter'])]).to(config['Evoformer']['device'])
        self.EvoformerExtractSingleLig = nn.Linear(global_config['rep_1d']['num_c'], global_config['num_single_c']).to(config['Evoformer']['device'])
        self.EvoformerExtractSingleRec = nn.Linear(global_config['rep_1d']['num_c'], global_config['num_single_c']).to(config['Evoformer']['device'])
        self.StructureModule = structure.StructureModule(config['StructureModule'], global_config).to(config['StructureModule']['device'])

        self.config = config
        self.global_config = global_config

        for name, module in self.Evoformer.named_modules():
            module.man_name = name
            def nan_hook(self, input, output):
                if any([torch.any(torch.isnan(x)) for x in output]):
                    print(f'Module {self.man_name} generated nans')
                    print('Inputs contains nan: ', [torch.any(torch.isnan(x)) for x in input])
                    print('Output contains nan: ', [torch.any(torch.isnan(x)) for x in output])
                    print('Inputs were: ', input)
                    print('Outputs were: ', output)
                    sys.stdout.flush()
                    raise utils.GeneratedNans(f'Module {self.man_name} generated nans')
            module.register_forward_hook(nan_hook)

    def forward(self, input, recycling=None):
        x = self.InputEmbedder(input, recycling=recycling)
        #return {'loss_total': x['r1d'].sum()}

        #x = {k: v.to('cuda:1') for k, v in x.items()}
        x['r1d'], x['l1d'], x['pair'] = x['r1d'].to(self.config['Evoformer']['device']), x['l1d'].to(self.config['Evoformer']['device']), x['pair'].to(self.config['Evoformer']['device'])

        def checkpoint_fun(function):
            return lambda a, b, c: function(a.clone(), b.clone(), c.clone())

        for evo_i, evo_iter in enumerate(self.Evoformer):
            if self.config['Evoformer']['EvoformerIteration']['checkpoint']:
                x['r1d'], x['l1d'], x['pair'] = checkpoint(checkpoint_fun(evo_iter), x['r1d'], x['l1d'], x['pair'])
            else:
                x['r1d'], x['l1d'], x['pair'] = evo_iter(x['r1d'], x['l1d'], x['pair'])

        pair = x['pair']
        rec_single = self.EvoformerExtractSingleRec(x['r1d'])
        lig_single = self.EvoformerExtractSingleLig(x['l1d'][:, 0])

        input = {k: {k1: v1.to(self.config['StructureModule']['device']) for k1, v1 in v.items()} for k, v in input.items()}
        struct_out = self.StructureModule({
            'r1d': rec_single.to(self.config['StructureModule']['device']),
            'l1d': lig_single.to(self.config['StructureModule']['device']),
            'pair': pair.to(self.config['StructureModule']['device']),
            'rec_bb_affine': input['target']['rec_bb_affine'],
            'rec_bb_affine_mask': input['target']['rec_bb_affine_mask'],
            'rec_torsions': input['target']['rec_torsions_sin_cos'],
            'lig_starts': input['target']['lig_starts'],
            'lig_ends': input['target']['lig_ends'],
            'lig_init_coords': input['target']['lig_init_coords']
        })

        # rescale to angstroms
        struct_out['rec_T'][..., -3:] = struct_out['rec_T'][..., -3:] * self.global_config['position_scale']
        struct_out['lig_T'][..., -3:] = struct_out['lig_T'][..., -3:] * self.global_config['position_scale']

        assert struct_out['rec_T'].shape[0] == 1
        final_all_atom = all_atom.backbone_affine_and_torsions_to_all_atom(
            struct_out['rec_T'][0][-1].clone(),
            struct_out['rec_torsions'][0][-1],
            input['target']['rec_aatype'][0]
        )
        #print({k: v.shape for k, v in struct_out.items()})

        out_dict = {}
        out_dict['loss'] = loss.total_loss(input, struct_out, final_all_atom, self.global_config)
        out_dict['final_all_atom'] = final_all_atom
        out_dict['struct_out'] = struct_out

        # make recycling input
        cbeta_coords, cbeta_mask = all_atom.atom14_to_cbeta_coords(
            final_all_atom['atom_pos_tensor'],
            input['target']['rec_atom14_has_coords'][0],
            input['target']['rec_aatype'][0]
        )
        #for i in range(len(input['target']['rec_aatype'][0])):
        #    print(input['target']['rec_aatype'][0][i])
        #    print(final_all_atom['atom_pos_tensor'][i])
        #    print(cbeta_mask[i])
        out_dict['recycling_input'] = {
            'rec_1d_prev': x['r1d'],
            'lig_1d_prev': x['l1d'][:, 0],
            'rep_2d_prev': pair,
            'rec_cbeta_prev': cbeta_coords[None],
            'rec_mask_prev': cbeta_mask[None],
            'lig_coords_prev': struct_out['lig_T'][:, -1, :, -3:]
        }

        return out_dict


def example3():
    from config import config, DATA_DIR
    with torch.no_grad():
        model = DockerIteration(config, config).cuda()

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)

        from dataset import DockingDataset
        ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
        #print(ds[0])
        item = ds[0]

        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None].cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)

        #print(item['fragments']['rr_2d'])
        model(item)

        #print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in model(item).items()})


def example4():
    from config import config, DATA_DIR

    #with torch.autograd.set_detect_anomaly(True):
    model = DockerIteration(config, config) #.cuda()
    model.train()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num params:', pytorch_total_params)

    from dataset import DockingDataset
    ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
    item = ds[0]

    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)

    #with torch.cuda.amp.autocast():
    #with torch.autograd.set_detect_anomaly(True):
    out = model(item)
    loss = out['loss_total']
    loss.backward()

    #print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in model(item).items()})


def example_profiler():
    from config import config, DATA_DIR

    model = DockerIteration(config, config) #.cuda()
    model.train()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num params:', pytorch_total_params)

    from dataset import DockingDataset
    ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
    item = ds[0]

    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #    out = model(item)
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == '__main__':
    tmp = {
        'a': {
            'b': [torch.zeros(3), torch.ones(2), {'g': torch.tensor(3)}],
            'c': torch.ones(5)
        }
    }
    output = []
    print(flatten_input(tmp, output))
    print(output)
