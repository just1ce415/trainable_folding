import prody
import numpy as np

from alphadock.residue_constants import known_atom_indices, relative_coords


def get_virtual_point(res, scale, axis):
    coords = res.getCoords()
    coords = coords - coords[1, :]
    d = coords[0] - coords[axis]
    d_norm = np.linalg.norm(d)
    d_unit = d / d_norm
    d_scaled = d_unit * scale
    coords[0] = coords[axis] + d_scaled
    return coords


def get_normalised_rmsd(true, pred, scale, axis):
    true_centered = get_virtual_point(true, scale, axis)
    pred_centered = get_virtual_point(pred, scale, axis)
    return np.sqrt(((true_centered - pred_centered) ** 2).sum(axis=1).mean())


def reconstruct_residue(intput_pdb_file, output_pdb_file):
    # Load the structure from the PDB file
    structure = prody.parsePDB(intput_pdb_file)

    # Get the NEW residue
    residue = structure.select('resname NEW')

    # Get the known coordinates from the residue based on the indices
    known_coords = residue.getCoords()

    # Get relative coords:
    atom_types, coordinates = list(zip(*relative_coords))
    coordinates = np.array(coordinates)

    # Calculate the transformation matrix from the known coordinates
    affine_matrix = prody.calcTransformation(coordinates[known_atom_indices], known_coords)

    # Transform the entire residue using the calculated matrix
    full_residue = np.matmul(coordinates, affine_matrix.getRotation().T) + affine_matrix.getTranslation()

    # Create a new residue with the reconstructed coordinates and atom types
    new_residue = prody.AtomGroup('REC')
    new_residue.setCoords(full_residue)
    new_residue.setNames(atom_types)
    new_residue.setResnames(['REC'] * 20)
    structure = structure + new_residue

    prody.writePDB(output_pdb_file, structure)