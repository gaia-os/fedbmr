import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

from collections import defaultdict
from typing import Callable, Dict, Optional
from optax import adabelief
from numpyro.infer import SVI, TraceGraph_ELBO
from numpyro.distributions import Distribution
from numpyro.optim import optax_to_numpyro
from functools import partial
from fedbmr.params import canonical, natural

Array = jnp.ndarray

class NaturalExponentialFamily(object):
    dist: Distribution  # numpyro distribution
    divide: Callable  # function used for removing contribution of local approximate likelihood
    natural_params: Dict  # dictionary containing natural parameters of that distribution
    approx_params: Dict  # dictionary containing natural paramters of the approximate likelihood

    def __init__(self, dist: Distribution) -> None:
        self.dist = dist
        self.natural_params = natural(dist)
        self.approx_params = jtu.tree_map(lambda x: 0., self.natural_params)

    def set_approx_params(self, params=Dict) -> None:
        self.approx_params = params


class GlobalPrior(object):
    prior_specs: Dict
    canonical_params: Dict
    approx_likelihood_params: Optional[Dict]=None

    def __init__(
        self, 
        prior_specs, 
        canonical_params,
        approx_likelihood_params=None 
    ):
        self.prior_specs = prior_specs
        self.canonical_params = canonical_params
        self.approx_likelihood_params = approx_likelihood_params
        
    def create_prior(self, func, params):
        return NaturalExponentialFamily(func(**params))
       
    def __call__(self):
        prior = jtu.tree_map(
            self.create_prior, 
            self.prior_specs, 
            self.canonical_params
        )

        if self.approx_likelihood_params is None:
            self.approx_likelihood_params = jtu.tree_map(
                lambda p: p.approx_params, 
                prior
            )
        else:
            def set_approx_params(p, params):
                p.set_approx_params(params)
                return p

            prior = jtu.tree_map(
                set_approx_params,
                prior,
                self.approx_likelihood_params
            )
        
        return prior


class GenerativeModel(object):
    global_prior: Dict
    global_model: Callable
    local_model: Callable

    def __init__(self, global_prior: Dict, global_model: Callable, local_model: Callable) -> None:
        self.global_prior = global_prior
        self.global_model = global_model
        self.local_model = local_model

    def __call__(self, *args, **kwargs) -> None:
        global_model = partial(self.global_model, self.global_prior)
        global_output = global_model(*args, **kwargs)
        self.local_model(global_output)


class Posterior(object):
    global_posterior: Callable
    local_posterior: Callable

    def __init__(self, global_posterior: Callable, local_posterior: Callable) -> None:
        self.global_posterior = global_posterior
        self.local_posterior = local_posterior

    def __call__(self, *args, **kwargs) -> None:
        global_output = self.global_posterior(*args, **kwargs)
        self.local_posterior(global_output)


class InfFed(object):
    '''Federeated inference.
    '''
    global_prior: GlobalPrior
    local_global_estimate: GlobalPrior
    generative_model: GenerativeModel
    posterior: Posterior
    optimizer: Callable

    def __init__(
        self, 
        global_prior: GlobalPrior,
        generative_model: GenerativeModel, 
        posterior: Posterior,
        optimizer: Callable = adabelief,
    ) -> None:

        self.global_prior = global_prior
        self.generative_model = generative_model
        self.posterior = posterior
        self.optimizer = optimizer
        
    def recieve_prior(self, global_prior):

        canonical_params = jtu.tree_map(
            lambda p: canonical(p.dist, p.natural_params), 
            global_prior)
        
        self.global_prior = GlobalPrior(
            self.global_prior.prior_specs, 
            canonical_params,
            self.global_prior.approx_likelihood_params)

    def process_messages(self, messages):
        # messages contain natural parameters of approximate likelihoods from other clients
        prior = self.global_prior()
        M = len(messages)

        def add_nat_params(p, params):
            p.natural_params = jtu.tree_map(lambda x, y: x + y/M, p.natural_params, params)
            return p

        for msg in messages:
            prior = jtu.tree_map(add_nat_params, prior, msg)
        
        canonical_params = jtu.tree_map(lambda p: canonical(p.dist, p.natural_params), prior)
        prior_specs = self.global_prior.prior_specs
        approx_params = self.global_prior.approx_likelihood_params
        gp = GlobalPrior(prior_specs, canonical_params, approx_likelihood_params=approx_params)
        self.global_prior = gp

    def send_message(self):

        def update_approximate_likelihood_params(glb, lcl, approx_params):
            return jtu.tree_map(
                lambda x, y, z: x - y + z, 
                lcl.natural_params, 
                glb.natural_params, 
                approx_params
            )

        # update approximate likelihood params
        approx_params = jtu.tree_map(
            update_approximate_likelihood_params, 
            self.global_prior(), 
            self.local_global_estimate(),
            self.global_prior.approx_likelihood_params
        )
        self.global_prior.approx_likelihood_params = approx_params
        return approx_params
    
    def send_posterior(self):

        def update_approximate_likelihood_params(glb, lcl, approx_params):
            return jtu.tree_map(
                lambda x, y, z: x - y + z, 
                lcl.natural_params, 
                glb.natural_params, 
                approx_params
            )

        # update approximate likelihood params
        approx_params = jtu.tree_map(
            update_approximate_likelihood_params, 
            self.global_prior(), 
            self.local_global_estimate(),
            self.global_prior.approx_likelihood_params
        )
        self.global_prior.approx_likelihood_params = approx_params
        
        return self.local_global_estimate()

    def make_local_generative_model(self):
        local_global_prior = self.global_prior()

        self.generative_model.global_prior = local_global_prior

        return self.generative_model

    def make_local_global_esimate(self, params):
        post_params = defaultdict(lambda: defaultdict(lambda: {}))
        for key, value in params.items():
            s1, s2 = key.split('.')
            post_params[s1][s2] = value
        
        prior_specs = self.global_prior.prior_specs
        return GlobalPrior(prior_specs, canonical_params=dict(post_params))
    
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

        self.results = svi.run(_rng_key, num_steps, progress_bar=False, **data)

        self.local_global_estimate = self.make_local_global_esimate(self.results.params)


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

    
