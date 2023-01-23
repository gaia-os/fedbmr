from typing import Callable, Dict
from optax import adabelief
from numpyro.infer import SVI, TraceGraph_ELBO
from numpyro.optim import optax_to_numpyro
from functools import partial

class GenerativeModel(object):

    global_prior: Callable
    local_prior: Callable
    likelihood: Callable

    def __init__(self, global_prior, local_prior, likelihood) -> None:
        self.global_prior = global_prior
        self.local_prior = local_prior
        self.likelihood = likelihood

    def __call__(self):
        global_params = self.global_prior()
        local_params = self.local_prior(params=global_params)
        self.likelihood(local_params)


class InfFed(GenerativeModel):
    '''Federeated inference.
    '''
    def process_messages(self, messages):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError
    
    def guide(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError


class GossipFed(InfFed):
    '''Gossip-based p2p federated variational inference'''
    pass


class BMRFed(InfFed):
    '''Bayesian model reduction for federeated variational inference.
    '''
    global_params: Dict=None

    def local_model(self):
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

    
