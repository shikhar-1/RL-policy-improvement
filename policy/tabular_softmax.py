import numpy as np
from .skeleton import Policy
from typing import Union

class TabularSoftmax(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int):

        self._theta = np.zeros((numStates, numActions))

        self._nStates = numStates
        self._nActions = numActions

    @property
    def parameters(self)->np.ndarray:

        return self._theta.flatten()

    @parameters.setter
    def parameters(self, p:np.ndarray):

        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:

        if action == None:
          return self.getActionProbabilities(state)
        else:
          return self.getActionProbabilities(state)[action]


    def samplAction(self, state:int)->int:
        """
        Samples an action to take given the state provided.

        output:
            action -- the sampled action
        """

        return np.random.choice(np.arange(self._nActions), p = self.getActionProbabilities(state))


    def getActionProbabilities(self, state:int)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided.

        output:
            distribution -- a 1D numpy array representing a probability
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in
                            the state provided.
        """
        a = self._theta[state]
        s = np.sum(np.exp(a))

        return np.exp(a)/s
