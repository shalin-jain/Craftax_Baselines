import jax
import jax.numpy as jnp

def sq_norm(g):
    return jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda _g: jnp.sum(jnp.square(_g)), g)
    )

def dot(g_1, g_2):
    """Compute dot product of g_1 and g_2"""
    return jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda g1, g2: jnp.sum(g1 * g2), g_1, g_2)
    )

def clip_rel(g_1, g_2):
    """Clip magnitude of g_1 s.t. it is no greater than g_2"""
    norm_sq_1 = sq_norm(g_1)
    norm_1 = jnp.sqrt(norm_sq_1 + 1e-8)
    norm_sq_2 = sq_norm(g_2)
    norm_2 = jnp.sqrt(norm_sq_2 + 1e-8)
    clipping_scale = jnp.minimum(1.0, norm_2 / norm_1)
    return jax.tree.map(
        lambda g: g * clipping_scale, g_1
    )

def project(g_1, g_2):
    """Projects g_1 on g_2"""
    dot_p = dot(g_1, g_2)
    norm_sq_2 = sq_norm(g_2)
    return jax.tree.map(
        lambda g2: g2 * (dot_p / (norm_sq_2 + 1e-8)), g_2
    )

def project_out(g_1, g_2):
    """Projects g_1 out of g_2"""
    projection = project(g_1, g_2)
    return jax.tree.map(lambda g1, proj: g1 - proj, g_1, projection)

def ocg(g_i, g_e, config):
    """Orthogonalize the intrinsic gradient to the extrinsic one and combines them."""
    # dot = g_i \cdot g_e
    dot = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda gi, ge: jnp.sum(gi * ge), g_i, g_e)
    )

    # project out the component of g_i that aligns with g_e
    # only when the dot product is negative (gradients are conflicting).
    # if COND_OCG is False, we always project.
    condition_to_project = jnp.logical_or(not config["COND_OCG"], dot < 0)

    g_i = jax.lax.cond(
        condition_to_project,
        project_out,         # If True, project g_i
        lambda gi, ge: gi,   # If False, use g_i as is
        g_i, g_e
    )

    # clip final_g_i magnitude
    if config["CLIP_OCG"]:
        g_i = clip_rel(g_i, g_e)

    return g_i, g_e
