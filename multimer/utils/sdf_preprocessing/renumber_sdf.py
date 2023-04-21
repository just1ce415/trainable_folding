"""
Written By: Omeir Khan

This script will renumber atoms, so it is numbered consistently.
For now, I made this to only work with symmetric benzene rings
(i.e. Xc1ccccc1, where X is a non-carbon atom)
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--sdf_file', '-f', help='A file containing coordinates of the fragment to renumnber')
parser.add_argument('--root_elem', '-a',
                    help='The element name of the non-carbon atom in the fragment (i.e. Cl, Br, N, etc..)')
parser.add_argument('--outfile', '-o', help='The name of the renumbered .sdf file (default = new.sdf)',
                    default='new.sdf')

args = parser.parse_args()


def read_sdf(sdf_f):
    coord_l = []
    bond_l = []
    with open(sdf_f) as f:
        for line in f:
            len_l = len(line.split())
            print(len_l, line)
            if len_l == 16:
                coord_l.append(line)
            if len_l == 7:
                bond_l.append(line)

    data_dict = {}

    for i, coord in enumerate(coord_l):
        idx = str(i + 1)
        data_dict[idx] = {'coord': coord, 'bonded_ats': []}

    for idx in data_dict:
        for bond in bond_l:
            bond_data = bond.split()
            bond_order = bond_data[2]
            if idx in bond_data[:2]:  # If the atom is in the bond
                for at in bond_data[:2]:
                    if at != idx:  # Select the other atom
                        data_dict[idx]['bonded_ats'].append((at, bond_order))

    return data_dict


# Renumber atoms in the file, starting from the root element
def renumber_file(sdf_dict, root_elem):
    for idx in sdf_dict:
        elem = sdf_dict[idx]['coord'].split()[3]
        print(elem, elem == root_elem)
        if elem == root_elem:
            root_idx = idx
    try:
        print(f'Atom with index {root_idx} selected as the root atom!')
    except:
        raise ValueError(f'No atom with element {root_elem} found in the .sdf file!')

    # Create a new dictionary with atomic coordinates renumbered
    num_ats = len(sdf_dict)
    curr_idx = 1
    prev_idx = root_idx
    new_sdf_dict = {}
    new_sdf_dict[curr_idx] = {'coord': sdf_dict[root_idx]['coord'], 'bonded_ats': []}

    processed_at_l = [root_idx]

    # Renumber atoms based on bonds from the root atom.
    # Renumbering is done by going in a loop around the benzene ring,
    # starting from the carbon atom bound to the root atom
    at_mapping = {}
    at_mapping[prev_idx] = curr_idx
    print(prev_idx, curr_idx, new_sdf_dict[curr_idx])

    while len(processed_at_l) < num_ats:
        print(f'Last at {prev_idx}:', len(processed_at_l), processed_at_l)
        for at in sdf_dict[prev_idx]['bonded_ats']:
            if at[0] not in processed_at_l:
                print(f'\tPrev {prev_idx}\tNew: {at[0]}', sdf_dict[prev_idx]['bonded_ats'])
                bto_at_idx = at[0]
                curr_idx += 1
                new_sdf_dict[curr_idx] = {'coord': sdf_dict[bto_at_idx]['coord'], 'bonded_ats': []}
                prev_idx = bto_at_idx
                processed_at_l.append(prev_idx)
                print('\t\t', prev_idx, curr_idx, new_sdf_dict[curr_idx])
                at_mapping[prev_idx] = curr_idx
                break

    print(f'Last at {prev_idx}:', len(processed_at_l), processed_at_l)
    print(at_mapping)

    # Recover bond information, and renumber bound atoms
    for idx in sdf_dict:
        new_idx = at_mapping[idx]
        for bond in sdf_dict[idx]['bonded_ats']:
            new_bond = (at_mapping[bond[0]], bond[1])
            new_sdf_dict[new_idx]['bonded_ats'].append(new_bond)

    return new_sdf_dict


def write_output(sdf_dict, outfile='new.sdf'):
    # Create strings for coordinates and bonds
    bond_str_l = []
    coord_l = []
    read_sets = []
    for i in range(len(sdf_dict)):
        idx = i + 1
        coord_l.append(sdf_dict[idx]['coord'])
        for bond in sdf_dict[idx]['bonded_ats']:
            bond_set = set([idx, bond[0]])
            if bond_set in read_sets:
                continue
            else:
                read_sets.append(bond_set)
                bond_str = f'  {idx}  {bond[0]}  {bond[1]}  0  0  0  0\n'
                bond_str_l.append(bond_str)
            print('\t', bond_set, bond_str)

    # Write the output
    outlines = ['FRAG\n', '  Renumber          3D                             0\n', '\n']
    outlines.append(f'  {len(coord_l)}  {len(bond_str_l)}  0  0  0  0  0  0  0  0999 V2000\n')
    for l in coord_l:
        outlines.append(l)
    for l in bond_str_l:
        outlines.append(l)
    outlines.append('$$$$\n')

    with open(outfile, 'w') as fo:
        for l in outlines:
            fo.write(l)


def main():
    sdf_data = read_sdf(args.sdf_file)

    for idx in sdf_data:
        print(idx, 'bonds:', sdf_data[idx]['bonded_ats'], 'coords:', sdf_data[idx]['coord'])

    new_sdf_data = renumber_file(sdf_data, args.root_elem)
    print(new_sdf_data)
    for idx in new_sdf_data:
        print(idx, 'bonds:', new_sdf_data[idx]['bonded_ats'], 'coords:', new_sdf_data[idx]['coord'])

    write_output(new_sdf_data, outfile=args.outfile)


if __name__ == '__main__':
    main()
