from multipledispatch import dispatch

from jax import lax
import jax.numpy as jnp

from numpyro.distributions.continuous import (
    Gamma,
    Beta,
    Normal,
    LogNormal
)

from numpyro.distributions.distribution import (
    ExpandedDistribution,
    Independent,
    MaskedDistribution,
)

def natural(p):
    r"""
    Return the natural parameters of the exponential distribution.
    """
    raise NotImplementedError

def canonical(p):
    r"""
    Retrun the canoncial parameters of the exponential distribution.
    """
    raise NotImplementedError


################################################################################
# Implementations
################################################################################

@dispatch(ExpandedDistribution)
def natural(p):
    params = natural(p.base_dist)
    shape = lax.broadcast_shapes(p.batch_shape)
    return jnp.broadcast_to(params, shape)

@dispatch(Independent)
def natural(p):
    params = natural(p.base_dist)
    shape = lax.broadcast_shapes(p.batch_shape)
    return jnp.broadcast_to(params, shape)

@dispatch(MaskedDistribution)
def natural(p):
    params = natural(p.base_dist)
    shape = lax.broadcast_shapes(p.batch_shape)
    return jnp.broadcast_to(params, shape)

@dispatch(Normal)
def natural(p):
    mu = p.loc
    sigma_sqr = p.scale ** 2
    return dict(eta1=mu / sigma_sqr, eta2=- 1 / 2 * sigma_sqr)

@dispatch(Normal, dict)
def canonical(p, params):
    p.loc = - params['eta1'] / (2 * params['eta2'])
    p.scale = jnp.sqrt( - 1 / (2 * params['eta2']))
    return p

@dispatch(LogNormal)
def natural(p):
    mu = p.loc
    sigma_sqr = p.scale ** 2
    return dict(eta1=mu / sigma_sqr, eta2=- 1 / 2 * sigma_sqr)

@dispatch(LogNormal, dict)
def canonical(p, params):
    p.loc = - params['eta1'] / (2 * params['eta2'])
    p.scale = jnp.sqrt( - 1 / (2 * params['eta2']))
    return p

@dispatch(Gamma)
def natural(p):
    alpha, beta = p.concentration, p.rate
    return dict(eta1=alpha - 1, eta2=- beta)

@dispatch(Gamma, dict)
def canonical(p, params):
    alpha = params['eta1'] + 1
    beta = - params['eta2']
    
    p.concentration = alpha
    p.rate = beta
    return p

@dispatch(Beta)
def natural(p):
    return dict(eta1=p.concentration1, eta2=p.concentration0)

@dispatch(Beta, dict)
def canonical(p, params):
    p.concentration1 = params['eta1']
    p.concentration0 = params['eta2']
    return p