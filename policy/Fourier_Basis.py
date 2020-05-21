import numpy as np
import math

class Fourier:

    def __init__(self, state_dim, i_order, d_ord):

        self._state_dim = state_dim
        self.__d_ord = d_ord
        self._d_terms = pow((d_ord + 1), state_dim)
        self._i_terms = i_order * state_dim
        self.o_terms = min(i_order, d_ord) * state_dim

        self._n_terms = self._i_terms + self._d_terms - self.o_terms

        self._coeff = [None] * self._n_terms

        self._cnt = np.zeros((self._state_dim))
        
        tc = 0
        while tc < self._d_terms:
            self._coeff[tc] = np.copy(self._cnt)
            i = 0
            while i < len(self._cnt):
                self._cnt[i] += 1
                if self._cnt[i] <= self.__d_ord:
                    break
                self._cnt[i] = 0
                i = i + 1
            tc = tc + 1

        i = 0
        while i < self._state_dim:
            j = d_ord + 1
            while j <= i_order:
                self._coeff[tc] = np.zeros(self._state_dim)
                self._coeff[tc][i] = float(j)
                tc = tc + 1
                j = j + 1
            i = i + 1


    def basify(self, x):
        result = np.zeros(self._n_terms)
        i = 0
        while i < self._n_terms:
            res = math.cos(math.pi * np.dot(self._coeff[i], x))
            result[i] = math.cos(math.pi * np.dot(self._coeff[i], x))
            i = i + 1
        return result

    def getNumOutputs(self):
        return self._n_terms