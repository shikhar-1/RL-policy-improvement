import numpy as np
# from .bbo_agent import BBOAgent

from typing import Callable


class GA():

    def __init__(self, populationSize: int, evaluationFunction: Callable,
                 initPopulation, numElite, min_req_pop, top_returns):
        self._name = "Genetic_Algorithm"
        self._population = initPopulation
        self._iniPop = self._population
        self._populationSize = populationSize
        self._truncIndex = int(populationSize / 2)
        self._numElite = numElite
        self._evaluationFunction = evaluationFunction
        self._alpha = 2.5
        self._J = float("-inf")
        self._min_req_pop = min_req_pop
        self._top_returns = top_returns

        pass

    @property
    def name(self) -> str:
        return self._name

    def _mutate(self, parent: np.ndarray) -> np.ndarray:

        return parent + self._alpha * np.random.normal(0, 1, len(parent))
        pass

    def train(self) -> np.ndarray:

        popul = []

        for k in range(0, self._populationSize):
            ret, flag, _ = self._evaluationFunction(self._population[k])
            popul.append((self._population[k], ret, flag))

        popul.sort(key=lambda x: x[1], reverse=True)
        thts = [x[0] for x in popul]
        thts_pass = [x[0] for x in popul if x[2]==1]

        children = []

        print("GA while loop started")

        while (len(children)+len(thts_pass)) < self._min_req_pop:
            chld = self._mutate(thts[np.random.choice(np.arange(len(thts)))])
            rt, flg, pd_list = self._evaluationFunction(chld)
            if flg ==1 and rt>min([-1*x for x in self._top_returns]):
                children.append(chld)
            if (len(children)+len(thts_pass))%10 == 0:
                print ((len(children)+len(thts_pass)))
        print("GA while loop finished")

        self._population = np.array(thts_pass + children)


        return 1


    def reset(self) -> None:
        self._population = self._iniPop
        pass
