


def batched_gather(params, indices, axis=0, batch_dims=0):
    """Implements a JAX equivalent of `tf.gather` with `axis` and `batch_dims`."""
    take_fn = lambda p, i: jnp.take(p, i, axis=axis)
    for _ in range(batch_dims):
        take_fn = jax.vmap(take_fn)
    return take_fn(params, indices)


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.Iterable), (
        'axis needs to be either an iterable, integer or "None"')

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return (jnp.sum(mask * value, axis=axis) /
            (jnp.sum(mask, axis=axis) * broadcast_factor + eps))