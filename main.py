import numpy as np
import math
from policy.basis import Basis
from agent.bbo_pdis import BBO_PDIS
from numpy import genfromtxt
from scipy import stats
from data_prep_eval import DataEval


#---------------------------
# INITIALIZING BBO PARAMETERS
#---------------------------


# For randomization, using multivariate normal dist
sigma = 1
# Policy dimensions
dim_theta = 2                # Issue with variable fixed in agents -> BBO_PDIS
# No. of random inits required
num_rand_inits = 10
# No. of inits related to behaviour policy required
num_behv_inits = 10
# behaviour policy
behv_params = np.array([0.01, -0.01, 1, 1])
# Minimum no. of required params before running the on the TEST set
min_req_pop = 30
# Alpha value
alpha = 0.05
#
prev_best_ret = 10
# Init behaviour policy
p_b = Basis()
p_b.parameters = behv_params


#---------------------------
# LOADING DATA NOW
#---------------------------

data = DataEval(dataPath = 'data_mdp.csv', gamma = 1, p_b = p_b)
epi_cand_list = data.epi_his_candidate
epi_safe_list = data.epi_his_safety



#---------------------------
# RUNNING THE ALGORITHM TO FIND BETTER POLICIES
#---------------------------

bbo = BBO_PDIS(sigma, data.calc_pdis, data.evalGA, dim_theta, num_rand_inits, num_behv_inits, behv_params, min_req_pop)
final_thetas = bbo.train()

print("OPTIMIZATION DONE")
print("RUNNING SAFETY TEST NOW")

result_list = []
sum_f = 0
print("----------")
print(len(final_thetas), "len_final_thetas")
print("----------")
for tht in final_thetas:
    t, f = data.safety_test(tht)
    result_list.append((t,f))
    sum_f = sum_f + f
succ_rate = sum_f*100/len(final_thetas)
pol_passed = sum_f
print("Policy Results:")
for k in result_list:
    if k[1] == 1:
        print(k[0])
print("Success Rate: ", succ_rate)
print("Number of Policies Passed:", pol_passed)

