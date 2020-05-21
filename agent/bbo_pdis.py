import numpy as np
# from .bbo_agent import BBOAgent
import cma
from typing import Callable
from agent.ga_pdis import GA
# import mlrose

class BBO_PDIS():


    def __init__(self, sigma:float, evaluationFunction:Callable, evalGA:Callable, dim_theta, num_rand_inits, num_behv_inits, behv_params, min_req_pop):
        self._Sigma = sigma * np.identity((4))
        self._dim_theta = dim_theta
        self._evaluationFunction = evaluationFunction
        self._evalGA = evalGA
        self._num_rand_inits = num_rand_inits
        self._behv_params = behv_params
        self._num_behv_inits = num_behv_inits
        # self._ga_iters = ga_iters
        self._min_req_pop = min_req_pop
        pass

    def train(self)->np.ndarray:

        # Initialization of Thetas

        print("INITIALIZING THETAS")
        init_thetas = []

        for k in range(self._num_rand_inits):
            init_thetas.append(np.random.rand(self._dim_theta*2))

        init_thetas.append(self._behv_params)

        for k in range(self._num_behv_inits - 1):
            init_thetas.append(np.random.multivariate_normal(self._behv_params, self._Sigma))

        # Running CMA-ES

        print("RUNNING CMA-ES NOW")
        x_opt = []
        c = 0
        # opts = cma.CMAOptions()
        # opts.set('maxiter', 5)
        top_returns = []
        for tht in init_thetas:
            print(c)
            a,b = cma.fmin2(self._evaluationFunction, tht, 1, options = {'maxiter' : 4})
            x_opt.append(a)
            top_returns.append(b.result[1])
            c= c+1


        # Running GA

        print("RUNNING GA NOW")
        ga = GA(populationSize= len(x_opt), evaluationFunction=self._evalGA, initPopulation=x_opt, numElite= (self._num_rand_inits + self._num_behv_inits),min_req_pop = self._min_req_pop, top_returns = top_returns )
        ga.train()
        x_opt = ga._population
        return x_opt
        pass
