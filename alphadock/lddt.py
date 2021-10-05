# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
####################################################################
# THE FILE WAS MODIFIED TO USE PYTORCH INSTEAD OF THE ORIGINAL JAX #
####################################################################

"""lDDT protein distance score."""
import torch


def lddt(predicted_points_a,
         predicted_points_b,
         true_points_a,
         true_points_b,
         true_points_a_mask,
         true_points_b_mask,
         cutoff=15.,
         exclude_self=False,
         reduce_axis=1,
         per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
      cutoff: Maximum distance for a pair of points to be included
      per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.

    Returns:
      An (approximate, see above) lDDT score in the range 0-1.
    """

    assert len(predicted_points_a.shape) == 3
    assert len(predicted_points_b.shape) == 3
    assert predicted_points_a.shape[-1] == 3
    assert predicted_points_b.shape[-1] == 3
    assert true_points_a_mask.shape[-1] == 1
    assert true_points_b_mask.shape[-1] == 1
    assert len(true_points_a_mask.shape) == 3
    assert len(true_points_b_mask.shape) == 3

    # Compute true and predicted distance matrices.
    dmat_true = torch.sqrt(1e-10 + torch.sum(torch.square(true_points_a[:, :, None] - true_points_b[:, None, :]), dim=-1))

    dmat_predicted = torch.sqrt(1e-10 + torch.sum(torch.square(predicted_points_a[:, :, None] - predicted_points_b[:, None, :]), dim=-1))

    dists_to_score = (dmat_true < cutoff) * true_points_a_mask * torch.transpose(true_points_b_mask, 2, 1)

    if exclude_self:
        assert dmat_true.shape[1] == dmat_true.shape[2]
        dists_to_score *= (1. - torch.eye(dmat_true.shape[1], dtype=dmat_true.dtype, device=dmat_true.device))

    # Shift unscored distances to be far away.
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).to(predicted_points_a.dtype) +
                    (dist_l1 < 1.0).to(predicted_points_a.dtype) +
                    (dist_l1 < 2.0).to(predicted_points_a.dtype) +
                    (dist_l1 < 4.0).to(predicted_points_a.dtype))

    # Normalize over the appropriate axes.
    reduce_axes = (1 + reduce_axis,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + torch.sum(dists_to_score, dim=reduce_axes))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=reduce_axes))
    return score
