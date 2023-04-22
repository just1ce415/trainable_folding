import numpy as np


def crop_feature(features, crop_size):
    # - 1 because we will always add the last element
    seq_len = features['seq_length'] - 1
    crop_size = crop_size - 1
    crop_size = min(seq_len, crop_size)
    # + 1 because np.random.randint doesn't include highest value
    start_crop = np.random.randint(0, seq_len - crop_size + 1)
    feat_skip = {'seq_length', 'resolution', 'num_alignments', 'assembly_num_chains', 'num_templates',
                 'cluster_bias_mask', 'id', 'seed', 'is_val', 'crop_size'}
    feat_1 = {
        'aatype', 'residue_index', 'all_atom_positions',
        'all_atom_mask', 'asym_id', 'sym_id', 'entity_id',
        'deletion_mean', 'entity_mask', 'seq_mask', 'loss_mask'
    }
    for k in features.keys():
        if k not in feat_skip:
            if k in feat_1:
                # before: (seq_len, ...), after: (crop_size + 1, ...)
                features[k] = np.concatenate((
                    features[k][start_crop: start_crop + crop_size],
                    features[k][-1:]
                ), axis=0)
            else:
                # before: (n_msa, seq_len, ...), after: (n_msa, crop_size + 1, ...)
                features[k] = np.concatenate((
                    features[k][:, start_crop: start_crop + crop_size],
                    features[k][:, -1:]
                ), axis=1)
    features['seq_length'] = crop_size + 1
    return features
