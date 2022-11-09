import torch
from alphadock import quat_affine


quat = torch.tensor([-2., 5., -1., 4.])

rotation = torch.tensor([
    [0.26087, 0.130435, 0.956522],
    [-0.565217, -0.782609, 0.26087],
    [0.782609, -0.608696, -0.130435]])

translation = torch.tensor([1., -3., 4.])
point = torch.tensor([0.7, 3.2, -2.9])

a = quat_affine.QuatAffine(quat, translation, unstack_inputs=True)
print(a)
true_new_point = torch.matmul(rotation, point[:, None])[:, 0] + translation
print(true_new_point)
print(a.apply_to_point(point))
print(quat_affine.rot_to_quat(rotation, unstack_inputs=True))
print(quat_affine.quat_to_rot(quat / torch.linalg.norm(quat, axis=-1, keepdims=True)))
print(rotation)
