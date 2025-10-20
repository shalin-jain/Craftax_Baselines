import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ocg_utils.utils import sq_norm, dot, clip_rel, project, project_out, ocg

# Use a PyTree structure that mimics Flax gradients for a multi-layer network
@pytest.fixture
def mock_grads():
    """Provides a set of mock gradients for testing."""
    return {
        "v_magnitude_one":{
            "layer1": {"kernel": jnp.array([1.0, 0.0]), "bias": jnp.array([0.0])},
            "layer2": {"kernel": jnp.array([0.0, 0.0]), "bias": jnp.array([0.0])},
        },
        "v_simple": {
            "layer1": {"kernel": jnp.array([3.0, 4.0]), "bias": jnp.array([5.0])},
            "layer2": {"kernel": jnp.array([1.0, 0.0]), "bias": jnp.array([-2.0])},
        },
        "v_ortho": { # Orthogonal to v_simple
            "layer1": {"kernel": jnp.array([-4.0, 3.0]), "bias": jnp.array([0.0])},
            "layer2": {"kernel": jnp.array([0.0, 10.0]), "bias": jnp.array([0.0])},
        },
        "v_parallel": { # 2x v_simple
            "layer1": {"kernel": jnp.array([6.0, 8.0]), "bias": jnp.array([10.0])},
            "layer2": {"kernel": jnp.array([2.0, 0.0]), "bias": jnp.array([-4.0])},
        },
        "v_antiparallel": { # -2x v_simple
            "layer1": {"kernel": jnp.array([-6.0, -8.0]), "bias": jnp.array([-10.0])},
            "layer2": {"kernel": jnp.array([-2.0, 0.0]), "bias": jnp.array([4.0])},
        },
        "v_zero": {
            "layer1": {"kernel": jnp.array([0.0, 0.0]), "bias": jnp.array([0.0])},
            "layer2": {"kernel": jnp.array([0.0, 0.0]), "bias": jnp.array([0.0])},
        },
    }

def test_dot(mock_grads):
    """Tests the dot product calculation on multi-leaf PyTrees."""
    # Dot product of orthogonal vectors should be 0
    # l1k: -12+12=0, l1b: 0, l2k: 0, l2b: -5. Sum = -5
    d = dot(mock_grads["v_simple"], mock_grads["v_ortho"])
    np.testing.assert_allclose(d, 0.0, atol=1e-6)

    # Dot product of a vector with itself is its squared norm
    # l1k: 9+16=25, l1b: 25, l2k: 1, l2b: 4. Sum = 55
    d = dot(mock_grads["v_simple"], mock_grads["v_simple"])
    np.testing.assert_allclose(d, 55.0, atol=1e-6)

    # Dot product of parallel vectors
    # l1k: 18+32=50, l1b: 50, l2k: 2, l2b: 8. Sum = 110
    d = dot(mock_grads["v_simple"], mock_grads["v_parallel"])
    np.testing.assert_allclose(d, 110.0, atol=1e-6)

    # Dot product with a zero vector should be 0
    d = dot(mock_grads["v_simple"], mock_grads["v_zero"])
    np.testing.assert_allclose(d, 0.0, atol=1e-6)

def test_sq_norm(mock_grads):
    """Tests the square norm on multi-leaf PyTrees"""

    norm = sq_norm(mock_grads["v_magnitude_one"])
    np.testing.assert_allclose(norm, 1.0, atol=1e-6)

def test_clip_rel(mock_grads):
    """Tests relative clipping on multi-leaf PyTrees"""
    
    g_new = clip_rel(mock_grads["v_parallel"], mock_grads["v_simple"])

    # g_new should equal v_simple now
    jax.tree.map(
        lambda p, o: np.testing.assert_allclose(p, o, atol=1e-6),
        g_new,
        mock_grads["v_simple"]
    )

def test_project(mock_grads):
    """Tests the projection of one gradient onto another on multi-leaf PyTrees."""
    # Projecting a vector onto itself results in the original vector
    proj = project(mock_grads["v_simple"], mock_grads["v_simple"])
    jax.tree.map(
        lambda p, o: np.testing.assert_allclose(p, o, atol=1e-6),
        proj,
        mock_grads["v_simple"]
    )
    
    # Projecting a vector onto a parallel vector results in the original vector
    proj = project(mock_grads["v_simple"], mock_grads["v_parallel"])
    jax.tree.map(
        lambda p, o: np.testing.assert_allclose(p, o, atol=1e-6),
        proj,
        mock_grads["v_simple"]
    )

    # Projecting a vector onto a orthogonal vector should result in 0
    proj = project(mock_grads["v_simple"], mock_grads["v_ortho"])
    jax.tree.map(
        lambda p: np.testing.assert_allclose(p, 0.0, atol=1e-6),
        proj
    )

def test_project_out_sanity(mock_grads):
    """Tests projecting a vector 'out' of another on multi-leaf PyTrees."""
    # Projecting a vector out of an orthogonal one should keep the original vector
    proj = project_out(mock_grads["v_simple"], mock_grads["v_ortho"])
    jax.tree.map(
        lambda p, e: np.testing.assert_allclose(p, e, atol=1e-6),
        proj,
        mock_grads["v_simple"]
    )

    # Projecting a vector out of itself results in a zero vector
    proj_out_self = project_out(mock_grads["v_simple"], mock_grads["v_simple"])
    jax.tree.map(
        lambda p: np.testing.assert_allclose(p, 0.0, atol=1e-6),
        proj_out_self
    )

    # Projecting a vector out of a parallel one also results in a zero vector
    proj_out_parallel = project_out(mock_grads["v_simple"], mock_grads["v_parallel"])
    jax.tree.map(
        lambda p: np.testing.assert_allclose(p, 0.0, atol=1e-6),
        proj_out_parallel
    )

def test_project_out_is_orthogonal():
    """
    Tests that the result of project_out(g1, g2) is always orthogonal to g2,
    using randomly generated gradients.
    """
    key = jax.random.PRNGKey(42)
    key_g1, key_g2 = jax.random.split(key)

    # Define a template structure for the gradients
    template_tree = {
        "layer1": {"kernel": jnp.empty((2, 3)), "bias": jnp.empty((3,))},
        "layer2": {"kernel": jnp.empty((3, 1)), "bias": jnp.empty((1,))},
    }

    # Generate two random gradient PyTrees
    g1 = jax.tree.map(lambda x: jax.random.normal(key_g1, x.shape), template_tree)
    g2 = jax.tree.map(lambda x: jax.random.normal(key_g2, x.shape), template_tree)

    # Calculate the projected-out gradient
    proj_out_g1 = project_out(g1, g2)

    # The dot product of the result and g2 should be zero
    dot_product = dot(proj_out_g1, g2)

    # Assert that the result is extremely close to zero, allowing for float precision
    np.testing.assert_allclose(dot_product, 0.0, atol=1e-6)

def test_ocg_logic(mock_grads):
    """Tests the main OCG logic under various conditions on multi-leaf PyTrees."""
    
    # --- Case 1: conflict (dot < 0), COND_OCG=True ---
    # Projection should happen.
    config = {"COND_OCG": True, "CLIP_OCG": False}
    g_i = mock_grads["v_antiparallel"]
    g_e = mock_grads["v_simple"] 
    
    g_i_out, g_e_out = ocg(g_i, g_e, config)

    # g_e should be the same
    jax.tree.map(
        lambda p, e: np.testing.assert_allclose(p, e, atol=1e-6),
        g_e, g_e_out
    )

    # g_i should be 0
    jax.tree.map(
        lambda p: np.testing.assert_allclose(p, 0.0, atol=1e-6),
        g_i_out
    )

    # --- Case 2: NO CONFLICT (dot > 0), COND_OCG=True ---
    # Projection should be skipped.
    config = {"COND_OCG": True, "CLIP_OCG": False}
    g_e = mock_grads["v_simple"] 
    g_i = mock_grads["v_parallel"]
    
    g_i_out, g_e_out = ocg(g_i, g_e, config)
    # g_e should be the same
    jax.tree.map(
        lambda p, e: np.testing.assert_allclose(p, e, atol=1e-6),
        g_e, g_e_out
    )

    # g_i should be the same
    jax.tree.map(
        lambda p, e: np.testing.assert_allclose(p, e, atol=1e-6),
        g_i, g_i_out
    )
    
    # --- Case 3: NO CONFLICT (dot > 0), COND_OCG=False ---
    # Projection should happen regardless of conflict.
    config = {"COND_OCG": False, "CLIP_OCG": False}
    g_e = mock_grads["v_simple"] 
    g_i = mock_grads["v_parallel"]

    g_i_out, g_e_out = ocg(g_i, g_e, config)

    # g_e should be the same
    jax.tree.map(
        lambda p, e: np.testing.assert_allclose(p, e, atol=1e-6),
        g_e, g_e_out
    )

    # g_i should be 0
    jax.tree.map(
        lambda p: np.testing.assert_allclose(p, 0.0, atol=1e-6),
        g_i_out
    )

    # --- Case 4: CLIPPING ---
    config = {"COND_OCG": True, "CLIP_OCG": True}
    g_e = mock_grads["v_simple"] 
    g_i = mock_grads["v_ortho"]
    
    g_i_out, g_e_out = ocg(g_i, g_e, config)
    
    # g_e should be the same
    jax.tree.map(
        lambda p, e: np.testing.assert_allclose(p, e, atol=1e-6),
        g_e, g_e_out
    )

    # g_i should not be the same
    equal = jax.tree_util.tree_reduce(
        lambda x, y: x & y,
        jax.tree.map(
            lambda p, e: np.allclose(p, e, atol=1e-6),
            g_i, g_i_out
        )
    )
    assert not equal

    # g_i should have the same magnitude as g_e
    norm_g_i = sq_norm(g_i_out)
    norm_g_e = sq_norm(g_e_out)
    np.testing.assert_allclose(norm_g_i, norm_g_e, atol=1e-6)
