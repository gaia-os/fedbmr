import json
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jax import Array
from multipledispatch import dispatch
from typing import Dict

from numpyro.distributions.continuous import (
    Gamma,
    Normal,
)

def encode_parameters(params: Dict) -> Dict:
    data = jtu.tree_map( lambda x: [[float(v) for v in x.reshape(-1)], list(x.shape)], params)
    return json.dumps( data ).encode('utf-8')

def decode_parameters(params: bytes) -> Dict:
    params = json.loads(params.decode())
    for var_name in params:
        for param_name in params[var_name]:
            x = jnp.array(params[var_name][param_name][0])
            shape = tuple(params[var_name][param_name][1])
            params[var_name][param_name] = x.reshape(*shape)
    
    return params

def log_factor(p, params, x):
    vals = jtu.tree_map(lambda T, η: T * η, sufficient_statistics(p, x), params)
    return jnp.sum(jnp.stack(jtu.tree_flatten(vals)[0]), 0)


###############################################################################

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