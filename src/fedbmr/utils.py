from multipledispatch import dispatch

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array

from numpyro.distributions.continuous import (
    Gamma,
    Normal,
)

def log_factor(p, params, x):
    vals = jtu.tree_map(lambda T, η: T * η, sufficient_statistics(p, x), params)
    return jnp.sum(jnp.stack(jtu.tree_flatten(vals)[0]), 0)

def sufficient_statistics(p, x):
    r"""
    Return the sufficient statistics of the exponential distribution.
    """
    raise NotImplementedError

################################################################################
# Implementations
################################################################################

@dispatch(Normal, float)
def sufficient_statistics(p, x):
    return dict(eta1=x, eta2=x ** 2)

@dispatch(Normal, Array)
def sufficient_statistics(p, x):
    return dict(eta1=x, eta2=x ** 2)

@dispatch(Gamma, float)
def sufficient_statistics(p, x):
    return dict(eta1=jnp.log(x), eta2=x)

@dispatch(Gamma, Array)
def sufficient_statistics(p, x):
    return dict(eta1=jnp.log(x), eta2=x)