import csv
import math
from policy.basis import Basis
from agent.bbo_pdis import BBO_PDIS
from scipy import stats
import numpy as np


class DataEval():
    def __init__(self, dataPath, gamma, p_b):
        self.full_data = self.read_data(dataPath)
        self.epi_his_safety = self.get_safety_data(self.full_data)
        self.epi_his_candidate = self.get_candidate_data(self.full_data)
        self.gamma = gamma
        self.p_b = p_b

    def read_data(self, dataPath):
        """
        Read data from the csv.
        """
        data = []
        with open(dataPath) as csvData:
            csvReader = csv.reader(csvData)
            for row in csvReader:
                data.append(row)
        return data


    def get_candidate_data(self, data):
        """
        Extract the candidate data from the full dataset.
        """
        history_cand  = data[5:int(len(data)*0.6)]
        epi_cand_list = []
        for h in history_cand:
            ret_list = []
            act_list = []
            st_list = []
            for k in range(len(h)):
                if k % 3 == 2 and math.isnan(float(h[k]))==0:
                    ret_list.append(float(h[k]))
                if k % 3 == 1 and math.isnan(float(h[k]))==0:
                    act_list.append(int(h[k]))
                if k % 3 == 0 and math.isnan(float(h[k]))==0:
                    st_list.append(float(h[k]))
            epi_cand_list.append((st_list,act_list,ret_list))
        return epi_cand_list


    def get_safety_data(self, data):
        """
        Extract the safety data from the full dataset.
        """
        history_safety = data[int(len(data) * 0.6):-1]
        epi_safe_list = []
        for h in history_safety:
            ret_list = []
            act_list = []
            st_list = []
            for k in range(len(h)):
                if k % 3 == 2 and math.isnan(float(h[k])) == 0:
                    ret_list.append(float(h[k]))
                if k % 3 == 1 and math.isnan(float(h[k])) == 0:
                    act_list.append(int(h[k]))
                if k % 3 == 0 and math.isnan(float(h[k])) == 0:
                    st_list.append(float(h[k]))
            epi_safe_list.append((st_list,act_list,ret_list))
        return epi_safe_list


    # ---------------------------
    # DEFINING EVALUATION FUNCTIONS
    # ---------------------------

    def calc_pdis(self, theta):
        """
        Calculate and return the PDIS value for a given policy.

        Used by the optimization functions (CMA here) as the objective function.
        """
        p_e = Basis()
        p_e.parameters = theta

        pdis_list = []
        for i in range(len(self.epi_his_candidate)):
            pdis_h = 0
            p_ratio = 1
            for t in range(len(self.epi_his_candidate[i][0])):
                p_ratio = p_ratio * (p_e.getActionProbabilities(self.epi_his_candidate[i][0][t])[self.epi_his_candidate[i][1][t]] /
                                     self.p_b.getActionProbabilities(self.epi_his_candidate[i][0][t])[self.epi_his_candidate[i][1][t]])
                pdis_h = pdis_h + p_ratio * (self.gamma ** t) * self.epi_his_candidate[i][2][t]
            pdis_list.append(pdis_h)

        return -1 * np.mean(pdis_list)

    def evalGA(self, theta):
        """
        Function used by the GA layer to decide on new policies.
        """
        p_e = Basis()
        p_e.parameters = theta

        pdis_list = []
        for i in range(len(self.epi_his_candidate)):
            pdis_h = 0
            p_ratio = 1
            for t in range(len(self.epi_his_candidate[i][0])):
                p_ratio = p_ratio * (p_e.getActionProbabilities(self.epi_his_candidate[i][0][t])[self.epi_his_candidate[i][1][t]] /
                                     self.p_b.getActionProbabilities(self.epi_his_candidate[i][0][t])[self.epi_his_candidate[i][1][t]])
                pdis_h = pdis_h + p_ratio * (self.gamma ** t) * self.epi_his_candidate[i][2][t]
            pdis_list.append(pdis_h)
        ds = len(history_safety)
        if (np.mean(pdis_list) - (1 / np.sqrt(ds)) * np.std(pdis_list) * stats.t.ppf(1 - alpha,
                                                                                     df=ds - 1)) < prev_best_ret:
            accept_flag = 0
        else:
            accept_flag = 1

        return (np.mean(pdis_list), accept_flag, pdis_list)

    def safety_test(self, theta):
        """
        Function to run the safety test on a particular policy
        """
        p_e = Basis()
        p_e.parameters = theta

        pdis_list = []
        # print(len(self.epi_his_safety),"len")
        for i in range(len(self.epi_his_safety)):
            pdis_h = 0
            p_ratio = 1
            for t in range(len(self.epi_his_safety[i][0])):
                p_ratio = p_ratio * (p_e.getActionProbabilities(self.epi_his_safety[i][0][t])[self.epi_his_safety[i][1][t]] /
                                     self.p_b.getActionProbabilities(self.epi_his_safety[i][0][t])[self.epi_his_safety[i][1][t]])
                pdis_h = pdis_h + p_ratio * (self.gamma ** t) * self.epi_his_safety[i][2][t]
            pdis_list.append(pdis_h)
        ds = len(self.epi_his_safety)
        # print(pdis_list)
        if (np.mean(pdis_list) - (1 / np.sqrt(ds)) * np.std(pdis_list) * stats.t.ppf(1 - alpha,
                                                                                     df=ds - 1)) < prev_best_ret:
            accept_flag = 0
        else:
            accept_flag = 1

        return (theta, accept_flag)
