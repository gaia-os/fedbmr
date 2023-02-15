import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

from typing import Callable, Dict, Optional
from optax import adabelief
from numpyro.infer import SVI, TraceGraph_ELBO, Predictive
from numpyro.distributions import Distribution
from numpyro.optim import optax_to_numpyro
from functools import partial
from equinox import Module

Array = jnp.ndarray

class NaturalExponentialFamily(object):
    dist: Distribution  # numpyro distribution
    params: Optional[Dict]  # dictionary containing natural parameters of that distribution

    def __init__(self, dist, params=None):
        self.dist = dist
        self.params = params


class GenerativeModel(object):

    global_prior: Dict
    global_model: Callable
    local_model: Callable
    likelihood: Callable

    def __init__(self, global_prior: Dict, global_model: Callable, local_model: Callable, likelihood: Callable) -> None:
        self.global_prior = global_prior
        self.global_model = global_model
        self.local_model = local_model
        self.likelihood = likelihood

    def __call__(self, *args, **kwargs) -> None:
        global_model = partial(self.global_model, self.global_prior)
        global_output = global_model(*args, **kwargs)
        local_output = self.local_model(global_output)
        self.likelihood(local_output)


class Posterior(object):
    global_posterior: Callable
    local_posterior: Callable
    local_solution: Optional[Dict]

    def __init__(self, global_posterior: Callable, local_posterior: Callable) -> None:
        self.global_posterior = global_posterior
        self.local_posterior = local_posterior

    def __call__(self, *args, **kwargs) -> None:
        global_output = self.global_posterior(*args, **kwargs)
        self.local_posterior(global_output)


class InfFed(Module):
    '''Federeated inference.
    '''
    generative_model: Callable
    posterior: Callable
    optimizer: Callable
    approx_likelihood_params: Optional[Dict]

    def __init__(
        self, 
        generative_model: GenerativeModel, 
        posterior: Posterior,
        optimizer: Callable = adabelief
    ) -> None:

        self.generative_model = generative_model
        self.posterior = posterior
        self.optimizer = optimizer
        self.__init_likelihood_params()

    def __init_likelihood_params(self):

        def set_to_zero(p):
            return jtu.tree_map(lambda x: 0., p.params)

        prior = self.generative_model.global_prior
        self.approx_likelihood_params = jtu.tree_map(set_to_zero, prior)

    def process_messages(self, messages):
        prior = self.generative_model.global_prior

        def add_nat_params(p, params):
            p.params = jtu.tree_map(lambda x, y: x + y, p.params, params)
            return p

        def remove_likelihood_params(p, params):
            p.params = jtu.tree_map(lambda x, y: x - y, p.params, params)
            return p

        for m in messages:
            prior = jtu.tree_map(add_nat_params, prior, m)

        self.generative_model.global_prior = prior

    def send_message(self):

        def remove_global_prior(glb, lcl):
            return jtu.tree_map(lambda x, y: y - x, glb.params, lcl.params)

        # remove global prior from the local solution
        tmp = jtu.tree_map(
            remove_global_prior, 
            self.generative_model.global_prior, 
            self.posterior.local_solution
        )

        # add previous estimates of the local parameters
        previous_params = self.approx_likelihood_params
        self.approx_likelihood_params = jtu.tree_map(lambda x, y: x + y, tmp, previous_params)

        return self.approx_likelihood_params
    
    def make_local_generative_model(self):
        def remove_likelihood_params(p, params):
            p.params = jtu.tree_map(lambda x, y: x - y, p.params, params)
            return p

        local_global_pior = jtu.tree_map(
            remove_likelihood_params, 
            self.generative_model.global_prior,
            self.approx_likelihood_params
        )

        global_model = self.generative_model.global_model
        local_model = self.generative_model.local_model
        likelihood = self.generative_model.likelihood

        return GenerativeModel(local_global_pior, global_model, local_model, likelihood)

    def inference(
            self,
            rng_key, 
            data, 
            num_steps=10_000, 
            num_particles=10,
            opt_kwargs={'learning_rate': 1e-3}
        ):
        
        optimizer = optax_to_numpyro(self.optimizer(**opt_kwargs))
        model = self.make_local_generative_model()
        guide = self.posterior
            
        loss = TraceGraph_ELBO(num_particles=num_particles)
        rng_key, _rng_key = jr.split(rng_key)

        svi = SVI(model, guide, optimizer, loss)

        results = svi.run(_rng_key, num_steps, progress_bar=False, **data)
        return results


class GossipFed(InfFed):
    '''Gossip-based p2p federated variational inference'''
    pass


class BMRFed(InfFed):
    '''Bayesian model reduction for federeated variational inference.
    '''
    global_params: Dict=None

    def local_model(self, gm: GenerativeModel):

        local_params = self.local_prior(params=self.global_params)
        self.likelihood(local_params)
    
    def local_guide(self, guide):
        guide(params=self.global_params)

    def inference(self, rng_key, guide, num_steps=1, *, progress_bar=True):
        optim = optax_to_numpyro(adabelief(learning_rate=1e-3))
        self.model = self.local_model
        self.guide = partial(self.local_guide, guide)
        svi = SVI(self.model, self.guide, optim, TraceGraph_ELBO(num_particles=10))
        result = svi.run(rng_key, num_steps=num_steps, progress_bar=progress_bar)

        return result

    
