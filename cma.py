import cma
import numpy as np


def objective_function(x):
    return 3*x - 5*x + np.log(x)

xopt = cma.fmin(cma.ff.rosen, 8 * [0], 0.5)
