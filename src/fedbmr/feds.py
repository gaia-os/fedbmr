from typing import Callable

class GenerativeModel(object):

    global_prior: Callable
    local_prior: Callable
    likelihood: Callable

    def model(self):
        global_params = self.global_prior()
        local_params = self.local_prior(global_params)
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
    pass
    
