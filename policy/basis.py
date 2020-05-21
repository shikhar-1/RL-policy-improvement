import numpy as np
from .skeleton import Policy
from .Fourier_Basis import Fourier

class Basis(Policy):
    def __init__(self):
        self._theta = None
        self._sigma = 1
        self._fb = Fourier(1,1,1)

    @property
    def parameters(self):
        return self._theta.flatten()

    @parameters.setter
    def parameters(self, p : np.ndarray):
        self._theta =  p.reshape(2,-1).T
        # print(self._theta)
    def __call__(self, state, action = None):
        if action == None:
            return self.getActionProbabilities(state)
        else:
            return self.getActionProbabilities(state)[action]

    def getExpansion(self, state):
        return self._fb.basify(state)
        # a,b,c,d = state
        # return (1,a,b,c,d,a*b,a*c,a*d,b*c,b*d,c*d,a*b*c,a*b*d,a*c*d,b*c*d,a*b*c*d,a*a*b,a*a*c,a*a*d,b*b*c,b*b*d,c*c*d,a*b*b,a*c*c,a*d*d,b*c*c,b*d*d,c*d*d)


    # def samplAction(self,state):
    #     return np.random.choice([0,1], p = self.getActionProbabilities(state))

    def getActionProbabilities(self, state):
        x = np.sum(np.exp(self._sigma * (self.getExpansion(state)).dot(self._theta)))
        y = np.exp(self._sigma*(self.getExpansion(state).dot(self._theta)))
        return y/x
