import numpy as np
import gym
from typing import List, Callable, Tuple
from math import exp


class Param:
    def __init__(self, threshold: Tuple[float, float],id,set_param_fn):
        self.threshold = threshold
        self.id = id
        self.value = np.random.uniform(*threshold)
        self.set_param_fn = set_param_fn

    def get_param(self):
        return self.value

    def set_param(self, value,  params):
        self.value = self.set_param_fn(self.threshold,value,params)

    def get_threshold(self):
        return self.threshold

def set_param1(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        p3_low,p3_high = params[3].get_threshold()
        params[3].set_param(p3_low + (p3_high - p3_low) * (value / (high - low)),params)

        return value

def set_param2(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param3(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param4(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param5(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param6(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param7(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value


class KPI:
    def __init__(self, updatefn: Callable):
        self.updatefn = updatefn
        self.value = 0.0

    def update_kpi(self, prev_params: List[float], prev_kpis: List[float]):
        self.value = self.updatefn(prev_params, prev_kpis)
        return self.value

    def get_kpi(self):
        return self.value



def update_Kpi1(prev_params, prev_kpis):
    P1 = prev_params[0]
    P2 = max(prev_params[1], 1e-3)
    return 0.5 * exp(-(P1 + 1) ** 2 / (2 * (P2 ** 2)))


def update_Kpi2(prev_params, prev_kpis):
    P1 = prev_params[0]
    P3 = max(prev_params[2], 1e-3)
    return exp(-(P1 - 1) ** 2 / (2 * (P3 ** 2)))


def update_Kpi3(prev_params, prev_kpis):
    P4 = prev_params[3]
    P5 = max(prev_params[4], 1e-3)
    K1 = prev_kpis[0]
    return exp(-(P4) ** 2 / (2 * (P5 ** 2)))


def update_Kpi4(prev_params, prev_kpis):
    P7 = prev_params[6]
    P6 = max(prev_params[5], 1e-3)
    K2 = prev_kpis[1]
    return exp(-(P7) ** 2 / (2 * (P6 ** 2)))



class ORANEnvironment(gym.Env):

    def __init__(self, num_bins=10, max_steps=50):

        super().__init__()

        setParamFns = [set_param1,set_param2,set_param3,set_param4,set_param5,set_param6,set_param7]

        # ----- Parameters -----
        self.params = [
            Param((0, 3),i,setParamFns[i]) for i in range(7)
        ]


        # ----- KPIs -----
        self.kpis = [
            KPI(update_Kpi1),
            KPI(update_Kpi2),
            KPI(update_Kpi3),
            KPI(update_Kpi4),
        ]

                # ---- TRUE CAUSAL GRAPH (11 x 11) ----
        # node order:
        # 0-6  = param0..param6
        # 7    = kpi0
        # 8    = kpi1
        # 9    = kpi2
        # 10   = kpi3

        self.true_adj_matrix = np.zeros((11, 11), dtype=np.float32)

        # P1,P2 → K1
        self.true_adj_matrix[7, 0] = 1
        self.true_adj_matrix[7, 1] = 1

        # P1,P3 → K2
        self.true_adj_matrix[8, 0] = 1
        self.true_adj_matrix[8, 2] = 1

        # P4,P5→ K3
        self.true_adj_matrix[9, 3] = 1
        self.true_adj_matrix[9, 4] = 1

        # P7,P6→ K4
        self.true_adj_matrix[10, 6] = 1
        self.true_adj_matrix[10, 5] = 1

        #P1 -> P4
        self.true_adj_matrix[3, 0] = 1

        self.num_params = len(self.params)
        self.num_kpis = len(self.kpis)

        # ----- Discrete Action Space -----
        # action = param_index * num_bins + bin_index
        self.num_bins = num_bins
        self.action_dim = self.num_params * self.num_bins
        self.action_spec = None   # <-- ADD THIS LINE

        self.max_steps = max_steps
        self.cur_step = 0

        self.prev_params = None
        self.prev_kpis = None

        self.reset()

    def get_save_information(self):
        return {
            "true_graph": self.true_adj_matrix
        }

    def reset(self):

        self.cur_step = 0

        for p in self.params:
            low, high = p.get_threshold()
            p.set_param(np.random.uniform(low, high),self.params)

        self.prev_params = [p.get_param() for p in self.params]
        self.prev_kpis = [0.0 for _ in self.kpis]

        return self._get_state()
    

    def reward(self, prev_kpis, new_kpis):
        # Reward: maximize all KPIs
        return sum(new_kpis)


    def step(self, action: int):

        param_id = action // self.num_bins
        bin_id = action % self.num_bins

        low, high = self.params[param_id].get_threshold()
        value = low + (high - low) * (bin_id / (self.num_bins - 1))

        self.params[param_id].set_param(value,self.params)
        new_params = [p.get_param() for p in self.params]

        # Update KPIs
        new_kpis = []
        for kpi in self.kpis:
            val = kpi.update_kpi(self.prev_params, self.prev_kpis)
            new_kpis.append(val)

        reward = self.reward(self.prev_kpis, new_kpis)

        # Update internal state
        self.prev_params = new_params
        self.prev_kpis = new_kpis

        self.cur_step += 1
        done = self.cur_step >= self.max_steps

        info = {"success": False}

        return self._get_state(), reward, done, info


    def _get_state(self):
        state = {}
        for i, val in enumerate(self.prev_params):
            state[f"param{i}"] = np.array([val], dtype=np.float32)

        for i, val in enumerate(self.prev_kpis):
            state[f"kpi{i}"] = np.array([val], dtype=np.float32)

        return state

    def observation_spec(self):
        return self._get_state()

    def observation_dims(self):
        dims = {}
        for i in range(self.num_params):
            dims[f"param{i}"] = np.array([1])
        for i in range(self.num_kpis):
            dims[f"kpi{i}"] = np.array([1])
        return dims