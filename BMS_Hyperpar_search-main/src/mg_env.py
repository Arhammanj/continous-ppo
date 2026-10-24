#!/usr/bin/env python3
#
#  tcl_env.py
#  TCL environment for RL algorithms
#
# Author: Ehtisham Asghar

import random
import numpy as np
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
# Run only if Kernel is dying due to matplotlib plt command
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Toggle verbose per-step prints
VERBOSE = False
import gym
# Trying out if this works for others. from gym import spaces had some issues
import gym.spaces as spaces
import threading
import math
from openpyxl import load_workbook
# Default parameters for
# From Taha's code
# days range
DEFAULT_DAY0 = 0
DEFAULT_DAYN = 365    # full year of data
DEFAULT_DAY0_TRAIN = 0
DEFAULT_DAYN_TRAIN = 300
DAY0_EVAL = 300
DAYN_EVAL = 365
# PV power generated in the microgrid
# DEFAULT_POWER_GENERATED = np.genfromtxt("Nim-PV.csv", delimiter=',', skip_header=0, usecols=[-1])

# Grid ToU prices
# DEFAULT_TOU_PRICE = np.genfromtxt("tou_price.csv", delimiter=',', skip_header=0, usecols=[-1])

# data link for pc simulation
# DEFAULT_TOU_PRICE = np.genfromtxt("c:\\RL-Git-Projects\\BMS-DQN\\datasets\\Prices.csv", delimiter=';', skip_header=1, usecols=[-1])

def _load_default_tou_price():
    price_path = Path(__file__).with_name("Prices.csv")
    if price_path.exists():
        try:
            prices = np.genfromtxt(
                price_path,
                delimiter=";",
                skip_header=1,
                usecols=1,
                dtype=np.float32,
            )
            prices = np.atleast_1d(prices)
            prices = prices[np.isfinite(prices)]
            if prices.size > 0:
                if float(np.max(prices)) > 5.0:
                    prices = prices / 100.0
                return prices.astype(np.float32)
        except Exception:
            pass

    # Synthetic fallback used when the CSV is unavailable or unreadable.
    return np.array([
        0.12, 0.11, 0.10, 0.09, 0.09, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.26,
        0.28, 0.30, 0.31, 0.32, 0.31, 0.29, 0.27, 0.24, 0.20, 0.17, 0.14, 0.12,
        0.12, 0.11, 0.10, 0.09, 0.09, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.26,
        0.28, 0.30, 0.31, 0.32, 0.31, 0.29, 0.27, 0.24, 0.20, 0.17, 0.14, 0.12,
    ], dtype=np.float32)


DEFAULT_TOU_PRICE = _load_default_tou_price()

# DEFAULT_PEN_UNB = 50000

# Length of one episode (24 hrs)
DEFAULT_ITERATIONS = 24

# Loads params
# DEFAULT_BASE_LOAD = np.genfromtxt("Average_Load.csv", delimiter=',', skip_header=0, usecols=[-1])

# Battery characteristics (kwh)
DEFAULT_BAT_CAPACITY=400.0
DEFAULT_MAX_SOC = 0.8       # max soc 0.8
DEFAULT_MIN_SOC = 0.2       # min soc 0.2
DEFAULT_EFFICIENCY = 1       # charge/discharge efficiency = 1
DEFAULT_DEGRADATION = 0.001      # battery degradation coefficent = 0
DEFAULT_PEN_SOC = 500        # SOC penalty set to 50000
DEFAULT_SOC_TARGET = 0.6
DEFAULT_BOUNDARY_SOC_LOW = 0.2
DEFAULT_BOUNDARY_SOC_HIGH = 0.8
DEFAULT_SOC_DEADBAND = 0.05
DEFAULT_ALPHA_SOC_STABILITY = 2.5
DEFAULT_BETA_BOUNDARY_SAFETY = 2.0
DEFAULT_GAMMA_ACTION_SMOOTHNESS = 0.4
DEFAULT_DELTA_BATTERY_HEALTH = 0.1
DEFAULT_ETA_ECONOMIC = 0.4
DEFAULT_TERMINAL_SOC_WEIGHT = 3.0
DEFAULT_ACTION_STEP_NORM = 80.0
DEFAULT_REWARD_MODE = "simple"
DEFAULT_PRICE_AWARE_WEIGHT = 0.6
DEFAULT_PRICE_AWARE_SMOOTHNESS_WEIGHT = 0.1
DEFAULT_PRICE_AWARE_SOC_WEIGHT = 0.3
DEFAULT_BETA_ENERGY_USE = 0.2
DEFAULT_LAMBDA_SOC_VIOLATION = 2.0

# Dg parameters
# DEFAULT_A = 0.0027
# DEFAULT_B = 0.02

MAX_R = 100

# Rendering lists
# LOADS_RENDER = []
BATTERY_RENDER = []
BATTERY_CH_DCH_RANDER = []
PRICE_RENDER = []
ENERGY_GRID_RENDER = []
# ENERGY_R_UNBALANCE_RENDER = []
GRID_PRICES_BUY_RENDER = []
GRID_PRICES_SELL_RENDER = []
ENERGY_GENERATED_RENDER = []
# TOTAL_CONSUMPTION_RENDER=[]
# DG_GEN_RENDER = []
OPERATION_COST = []
OP_CO_woDC = []

#ACTIONS = [[i, j] for i in range(0, 220, 20) for j in range(-80, 120, 40)]
# ACTIONS = [[i, j] for i in range(0, 220, 20) for j in range(-80, 100, 20)]
#ACTIONS = [[i, j] for i in range(0, 220, 20) for j in range(-80, 85, 5)]
#ACTIONS = [[i, j] for i in range(0, 205, 5) for j in range(-80, 85, 5)]

ACTIONS = [i for i in range(-80, 90, 10)]    # [-80, -70, ..., 70, 80]

# ACTIONS = [i for i in range(-80, 90, 10)]

# class Battery():
#     '''simulate a simple battery here'''
#     def __init__(self,capacity, max_soc, min_soc, efficiency, degradation, penalty_soc):
#         self.capacity = capacity # max capacity 200 kwh
#         self.max_soc = max_soc # max soc 0.8
#         self.min_soc = min_soc # 0.2
#         self.efficiency = efficiency # charge and discharge efficiency 1.0
#         self.degradation = degradation # degradation cofficient set to 0
#         self.pen_soc = penalty_soc

#     def step(self,battery_action):
#         p_bat = battery_action
#         #print(f'SOC Before: {self.current_capacity}')
#         #print(f'Bat Action: {p_bat}') 
#         #print(f'max soc: {self.max_soc}')
#         #print(f'min soc: {self.min_soc}')
#         #print(f'cap: {self.capacity}')
#         updated_capacity = (self.current_capacity+(p_bat/self.capacity))
#         #print(f'Updated capacity: {updated_capacity}')
#         #print(f'current cap: {self.current_capacity}')
#         self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge -> positive, if discharge-> negative
#         self.current_capacity=updated_capacity# update capacity to current codition 
#         #print(f'Power charg/disch: {self.energy_change}')
#         #print('---****')

#     def _get_cost(self,energy):# calculate the cost depends on the energy change
#         cost = (energy**2)*self.degradation
#         return cost
    
#     @property
#     def SOC(self):
#         return self.current_capacity
    
#     def reset(self):
#         #self.current_capacity=round(np.random.uniform(0.2,0.8), 1)  # initial capcity rounded to 1 to get 0.2, 0.3,...
#         self.current_capacity=0.3      # set to 30%

class Battery():
    '''simulate a simple battery here'''
    def __init__(self,capacity, max_soc, min_soc, efficiency, degradation):
        self.capacity=capacity
        self.max_soc=max_soc
        # self.initial_capacity=parameters['initial_capacity']
        self.min_soc=min_soc # 0.2
        self.degradation=degradation # degradation cost 1.2
        self.efficiency=efficiency
    def step(self,action_battery):
        energy=action_battery
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge, positive, if discharge, negative
        self.current_capacity=updated_capacity# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        cost=energy**2*self.degradation
        return cost  
    @property
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity=np.random.uniform(0.2,0.8)
        # print(f'Battery initial soc: {self.current_capacity}')
        return self.current_capacity


class Grid:
    def __init__(self, tou_price):
        # Price the grid charges to buy energy from it
        self.buy_prices = np.array(tou_price, dtype=np.float32)
        # Price the grid pays when we sell energy (lower than buy price)
        self.sell_prices = np.array(tou_price, dtype=np.float32) * 0.7
        self.exchange_ability = 200    # 200 kW
        self.time = 0
    
    # to link self.timestep (which is current time_step) to self.time
    def set_time(self, time):
        self.time = time

    def _get_buy_cost(self, p_ex):
        # Cost to buy p_ex energy from the grid at current time
        idx = self.time % len(self.buy_prices)
        return (self.buy_prices[idx]) * p_ex

    def _get_sell_benefit(self, p_ex):
        # Benefit gained by selling p_ex energy to the grid at current time
        idx = self.time % len(self.sell_prices)
        return (self.sell_prices[idx]) * p_ex

    # Backwards-compatible alias (returns buy cost)
    def _get_cost(self, p_ex):
        return self._get_buy_cost(p_ex)

# class DG:
#     def __init__(self, a_factor, b_factor):
#         self.a_factor=a_factor
#         self.b_factor=b_factor
#         self.last_step_output=None 

#     def step(self,dg_action):
#         self.current_output=dg_action

#     def _get_cost(self, output):
#         cost = ((self.a_factor*output**2)+(self.b_factor*output))
#         # print(cost)
#         return cost 
    
#     def reset(self):
#         self.current_output = 0
  

# class Generation:
#     def __init__(self, generation):
#         self.power = generation


#     def current_generation(self, time):
#         # We consider that we have 2 sources of power a constant source and a variable source
#         return self.power[time]


# class Load:
#     def __init__(self, base_load):
#         self.base_load = base_load

#     def current_load(self, time):
#         return self.base_load[time]

class MicroGridEnv(gym.Env):
    def __init__(self,**kwargs):

        # Get number of iterations and TCLs from the
        # parameters (we have to define it through kwargs because
        # of how Gym works...)
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.day0 = kwargs.get("day0", DEFAULT_DAY0)            # Day no. 0
        self.dayn = kwargs.get("dayn", DEFAULT_DAYN)    # Day no. depends on how much data model is being trained 
        self.tou_price = kwargs.get("tou_price", DEFAULT_TOU_PRICE)
        if len(self.tou_price) == 0:
            raise ValueError("tou_price must contain at least one value")
        self.price_series_len = len(self.tou_price)

        # Battery parameters
        self.bat_capacity = kwargs.get("battery_capacity", DEFAULT_BAT_CAPACITY)
        self.soc_max = kwargs.get("max_soc", DEFAULT_MAX_SOC)
        self.soc_min = kwargs.get("min_soc", DEFAULT_MIN_SOC)
        self.eff = kwargs.get("eff", DEFAULT_EFFICIENCY)
        self.deg = kwargs.get("deg", DEFAULT_DEGRADATION)
        self.soc_penalty = kwargs.get("soc_penalty", DEFAULT_PEN_SOC)
        self.soc_target = np.clip(kwargs.get("soc_target", DEFAULT_SOC_TARGET), self.soc_min, self.soc_max)
        self.boundary_soc_low = kwargs.get("boundary_soc_low", DEFAULT_BOUNDARY_SOC_LOW)
        self.boundary_soc_high = kwargs.get("boundary_soc_high", DEFAULT_BOUNDARY_SOC_HIGH)
        self.soc_deadband = kwargs.get("soc_deadband", DEFAULT_SOC_DEADBAND)
        self.alpha_soc_stability = kwargs.get("alpha_soc_stability", DEFAULT_ALPHA_SOC_STABILITY)
        self.beta_boundary_safety = kwargs.get("beta_boundary_safety", DEFAULT_BETA_BOUNDARY_SAFETY)
        self.gamma_action_smoothness = kwargs.get("gamma_action_smoothness", DEFAULT_GAMMA_ACTION_SMOOTHNESS)
        self.delta_battery_health = kwargs.get("delta_battery_health", DEFAULT_DELTA_BATTERY_HEALTH)
        self.eta_economic = kwargs.get("eta_economic", DEFAULT_ETA_ECONOMIC)
        self.terminal_soc_weight = kwargs.get("terminal_soc_weight", DEFAULT_TERMINAL_SOC_WEIGHT)
        self.action_step_norm = kwargs.get("action_step_norm", DEFAULT_ACTION_STEP_NORM)
        self.reward_mode = kwargs.get("reward_mode", DEFAULT_REWARD_MODE)
        self.price_aware_weight = kwargs.get("price_aware_weight", DEFAULT_PRICE_AWARE_WEIGHT)
        self.price_aware_smoothness_weight = kwargs.get(
            "price_aware_smoothness_weight", DEFAULT_PRICE_AWARE_SMOOTHNESS_WEIGHT
        )
        self.price_aware_soc_weight = kwargs.get("price_aware_soc_weight", DEFAULT_PRICE_AWARE_SOC_WEIGHT)
        self.beta_energy_use = kwargs.get("beta_energy_use", DEFAULT_BETA_ENERGY_USE)
        self.lambda_soc_violation = kwargs.get("lambda_soc_violation", DEFAULT_LAMBDA_SOC_VIOLATION)

        # Penalty coefficient power imbalance
        # self.penalty_coefficient = kwargs.get("unb_penalty", DEFAULT_PEN_UNB)   #control soft penalty constraint
        
        # DG parameters
        # self.a = kwargs.get("a", DEFAULT_A)
        # self.b = kwargs.get("b", DEFAULT_B)

        # The current day: initally set to day no. 0
        self.day = self.day0

        # Setting first timestep of each time period
        self.time_step = 0
        self.prev_battery_action = 0.0
        self.prev_soc = DEFAULT_SOC_TARGET

        # self.generation = Generation(kwargs.get("generation_data", DEFAULT_POWER_GENERATED))
        self.grid = Grid(tou_price=self.tou_price)
        self.battery = Battery(capacity=self.bat_capacity, max_soc=self.soc_max, min_soc=self.soc_min, \
                               efficiency=self.eff, degradation=self.deg)
        # self.dg = DG(a_factor=self.a, b_factor=self.b)

        # self.load = Load(kwargs.get('load_data', DEFAULT_BASE_LOAD))

        #self.action_space_sep = spaces.Box(low=0, high=1, dtype=np.float32,
                                       #shape=(13,))

        # Total 189 combination of actions = [(0, -80), (0, -60)...(10, -80),.. (20, 80)]
        #self.action_space = spaces.Discrete(55)
        self.action_space = spaces.Discrete(len(ACTIONS))
        # self.action_space = spaces.Discrete(17)
        #self.action_space = spaces.Discrete(363)
        #self.action_space = spaces.Discrete(1353)

        # Observations: loads + battery soc + res generation + price + time-step of day (any array of shape = (5,))
        # self.observation_space = spaces.Box(low=-600, high=600, dtype=np.float32,
        #                                     shape=(5,))
        
        # Observations here are time_step of the day, SOC, Energy price for t and next t+23 (an array of shape=(26,))
        # self.observation_space = spaces.Box(low=0, high=1, dtype=np.float32,
        #                                     shape=(3,))
        
        # # Observations here are time_step of the day, SOC, Energy price for t and next t+23 (an array of shape=(26,))
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.float32,
                                            shape=(26,))


    def _price_at(self, absolute_step):
        """Return wrapped price at absolute step index."""
        return self.grid.sell_prices[absolute_step % self.price_series_len]


    def _price_window(self, start_step, horizon=24):
        """Return fixed-length wrapped future price window."""
        indices = (np.arange(start_step, start_step + horizon) % self.price_series_len).astype(np.int64)
        return self.grid.sell_prices[indices]


    def _simple_reward(self, sell_benefit, buy_cost, battery_cost):
        soc_t = float(self.battery.current_capacity)

        # 1. Profit (normalized to [-1, +1])
        profit = float(sell_benefit) - float(buy_cost)
        profit_norm = max(-1.0, min(1.0, profit / 50.0))

        # 2. Battery penalty
        battery_penalty = min(0.2, (float(battery_cost) / 5.0) * 0.3)

        # 3. SOC penalty
        if soc_t < self.boundary_soc_low:
            soc_penalty = min(1.0, (self.boundary_soc_low - soc_t) / max(1e-6, self.boundary_soc_low))
        elif soc_t > self.boundary_soc_high:
            soc_penalty = min(1.0, (soc_t - self.boundary_soc_high) / max(1e-6, (1.0 - self.boundary_soc_high)))
        else:
            soc_penalty = 0.0

        # 4. Shaping bonus
        soc_mid = (self.boundary_soc_low + self.boundary_soc_high) / 2.0
        soc_range = max(1e-6, (self.boundary_soc_high - self.boundary_soc_low) / 2.0)
        soc_distance = abs(soc_t - soc_mid) / soc_range
        shaping_bonus = 0.2 * (1.0 - min(1.0, soc_distance))

        # 5. Price alignment
        price_t = float(self.grid.sell_prices[self.grid.time % len(self.grid.sell_prices)])
        max_price = max(1e-6, float(np.max(np.abs(self.grid.sell_prices))))
        p_norm = price_t / max_price

        energy_change = float(self.battery.energy_change)
        max_energy = 80.0

        if energy_change > 0:
            alignment_bonus = 0.15 * (1.0 - p_norm) * (energy_change / max_energy)
        else:
            alignment_bonus = 0.15 * p_norm * (abs(energy_change) / max_energy)

        alignment_bonus = max(-0.15, min(0.15, alignment_bonus))

        # 6. Final reward
        reward = (
            profit_norm
            - battery_penalty
            - (0.5 * soc_penalty)
            + shaping_bonus
            + alignment_bonus
        )
        if 'VERBOSE' in globals() and VERBOSE:
            print(
                f"profit_norm={profit_norm:.3f} | battery_pen={battery_penalty:.3f} | soc_pen={soc_penalty:.3f} | soc_t={soc_t:.3f} | bonus={shaping_bonus:.3f} | align={alignment_bonus:.3f}"
            )

        return float(reward)


    def _simple_similar_reward(self, sell_benefit, buy_cost, battery_cost):
        return 0.95 * sell_benefit - 1.05 * buy_cost - 0.01 * battery_cost


    def _similar_factors_reward(self, sell_benefit, buy_cost, battery_cost, battery_action):
        # Same factors as the earlier shaped reward, but with normalized action terms.
        scale = max(1e-6, float(self.action_step_norm))
        a_t = float(battery_action) / scale
        a_prev = float(self.prev_battery_action) / scale
        soc_t = float(self.battery.current_capacity)
        soc_prev = float(self.prev_soc)

        economic_term = 0.4 * (sell_benefit - buy_cost)

        action_penalty = (
            0.7 * ((a_t - a_prev) ** 2)
            + 0.2 * (a_t ** 2)
            + 1.5 * (1.0 if (a_t * a_prev) < 0 else 0.0)
            + 0.5 * (max(0.0, 0.05 - abs(a_t)) ** 2)
        )

        soc_center = soc_t - 0.6
        soc_penalty = (
            1.2 * ((soc_center ** 2) + 0.5 * (soc_center ** 4))
            + 1.5 * ((max(0.0, 0.2 - soc_t) ** 2) + (max(0.0, soc_t - 0.8) ** 2))
        )

        soc_smoothness_penalty = 0.2 * ((soc_t - soc_prev) ** 2)
        reward = economic_term - action_penalty - soc_penalty - soc_smoothness_penalty
        operation_cost = (-1.0 * economic_term) + action_penalty + soc_penalty + soc_smoothness_penalty
        return reward, operation_cost


    def _price_aware_reward(self, sell_benefit, buy_cost, battery_cost, battery_action):
        # Baseline arbitrage objective + battery wear.
        base_term = sell_benefit - buy_cost - 0.01 * battery_cost

        # Explicit charge/discharge alignment with current grid price.
        price_t = float(self.grid.sell_prices[self.grid.time])
        max_price = max(1e-6, float(np.max(self.grid.sell_prices)))
        p_norm = price_t / max_price

        max_action = max(1e-6, float(np.max(np.abs(ACTIONS))))
        a_norm = float(battery_action) / max_action
        charge_mag = max(0.0, a_norm)
        discharge_mag = max(0.0, -a_norm)

        # Reward charging when cheap and discharging when expensive.
        price_action_term = (
            charge_mag * (1.0 - p_norm)
            + discharge_mag * p_norm
        )

        # Keep control smooth.
        prev_a_norm = float(self.prev_battery_action) / max_action
        smoothness_penalty = (a_norm - prev_a_norm) ** 2

        # Soft SOC tracking around nominal target.
        soc_t = float(self.battery.current_capacity)
        soc_penalty = (soc_t - 0.6) ** 2 + (max(0.0, 0.2 - soc_t) ** 2) + (max(0.0, soc_t - 0.8) ** 2)

        reward = (
            base_term
            + self.price_aware_weight * price_action_term
            - self.price_aware_smoothness_weight * smoothness_penalty
            - self.price_aware_soc_weight * soc_penalty
        )
        operation_cost = -reward
        return reward, operation_cost


    def _profit_energy_soc_reward(self, sell_benefit, buy_cost, battery_cost):
        # Requested reward: normalized profit - battery wear - energy-use penalty - SOC boundary violation.
        energy_used = abs(float(self.battery.energy_change))
        energy_max = max(1e-6, float(np.max(np.abs(ACTIONS))))
        max_price = max(1e-6, float(np.max(self.grid.sell_prices)))
        profit_max = max(1e-6, max_price * energy_max)
        battery_max = max(1e-6, self.battery.degradation * (energy_max ** 2))

        soc_t = float(self.battery.current_capacity)
        soc_violation = (max(0.0, 0.2 - soc_t) ** 2) + (max(0.0, soc_t - 0.8) ** 2)

        reward = (
            ((sell_benefit - buy_cost) / profit_max)
            - 0.01 * (battery_cost / battery_max)
            - self.beta_energy_use * (energy_used / energy_max)
            - self.lambda_soc_violation * soc_violation
        )
        operation_cost = -reward
        return reward, operation_cost



    def _build_state(self):
        """
        Return current state representation as one vector.
        Returns:
            state: 1D state vector, Loads, current battery soc, current power generation,
                 current price and current time (hour) of day
        """
        # current price + time of day (hour)
        # Scaling between 0 and 1
        # We need to standardize the generation and the price
        #print(f'Time_step in build state: {self.time_step}')
        # current_load = self.load.current_load(self.day*self.iterations + self.time_step) 
        # current_load = (current_load - min(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                 / (max(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations])\
        #                 - min(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations]))
        
        # current_load = np.float32(current_load/max(self.load.base_load))

        #print(f'max load: {max(self.load.base_load)}')
        # current_load = (current_load - np.average(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                 / np.std(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations])

        #print(f'STEPS: {self.day*self.iterations+(self.time_step % 24)}')
        # current_generation = self.generation.current_generation(self.day*self.iterations + self.time_step)
        # current_generation = (current_generation-
        #                        np.average(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                       /np.std(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations])

        # current_generation = np.float32(current_generation/max(self.generation.power))

        # current_generation = (current_generation-
        #                        min(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                       /(max(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations])\
        #                         - min(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations]))

        # Time steps we are not giving 'self.day*self.iterations+self.time_step' 
        # because right now we are only using same 24 hrs price data for whole training
        # change it when you will be trianing on years training data 
        
        # price_grid_sell = self.grid.sell_prices[self.time_step % 24]
        # price_grid_sell = (price_grid_sell -
        #          np.average(self.grid.sell_prices[0:self.iterations])) \
        #         / np.std(self.grid.sell_prices[0: self.iterations])

        # price_grid_sell = (price_grid_sell -
        #         min(self.grid.sell_prices[0:self.iterations])) \
        #         / (max(self.grid.sell_prices[0: self.iterations])\
        #            - min(self.grid.sell_prices[0: self.iterations]))

        # price_grid_sell = self.grid.sell_prices[self.day*self.iterations+self.time_step]
        # price_grid_sell = np.float32(price_grid_sell/max(self.grid.sell_prices))

        absolute_step = self.day * self.iterations + self.time_step
        price_grid_sell = self._price_window(absolute_step, horizon=24)
        price_grid_sell = np.float32(price_grid_sell/max(self.grid.sell_prices))
        #print(f'max price: {max(self.grid.sell_prices)}')
        
        time_step = np.float32((self.time_step)/(self.iterations-1))

        # Soc is also normalized to 1 by dividing max_soc=0.8
        soc = np.float32(self.battery.SOC/(self.battery.max_soc))

        # state = np.array([current_load, time_step, soc, current_generation,
        #                          price_grid_sell ])

        # state = np.array([time_step, soc, price_grid_sell])
        state = np.concatenate((np.array([time_step, soc]), price_grid_sell))
        #rint(f'States: {state}')
        return state

    def _build_info(self):
        """
        Return dictionary of misc. infos to be given per state.
        Here this means providing forecasts of future
        prices and temperatures (next 24h)
        """
        # temp_forecast = np.array(self.temperatures[self.time_step + 1:self.time_step + self.iterations+1])
        # return {"temperature_forecast": temp_forecast,
        #         "forecast_times": np.arange(0, self.iterations)}


    def _plot_price_and_charge_discharge(self):
        """Plot hourly grid price against battery charge/discharge power."""
        if len(GRID_PRICES_SELL_RENDER) == 0 or len(BATTERY_CH_DCH_RANDER) == 0:
            return

        n = min(len(GRID_PRICES_SELL_RENDER), len(BATTERY_CH_DCH_RANDER))
        hours = np.arange(n)
        prices = np.array(GRID_PRICES_SELL_RENDER[:n], dtype=np.float32)
        ch_dch = np.array(BATTERY_CH_DCH_RANDER[:n], dtype=np.float32)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(hours, prices, color="dodgerblue", linewidth=2)
        axes[0].set_ylabel("Price")
        axes[0].set_title("Grid Price")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        charge = np.clip(ch_dch, 0, None)
        discharge = np.clip(ch_dch, None, 0)
        axes[1].bar(hours, charge, color="seagreen", label="Charge")
        axes[1].bar(hours, discharge, color="tomato", label="Discharge")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Hour")
        axes[1].set_ylabel("Power")
        axes[1].set_title("Battery Charge/Discharge")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(frameon=False)

        plt.tight_layout()
        plt.show()


    def step(self, action):
        """
        Arguments:
            action: A list.

        Returns:
            state: Current state
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (maximum number of iterations)
            info: None (not used here)
        """
        if type(action) is not list:
            action = ACTIONS[action]

        # print(f'Actions{self.time_step}: {action}')
        #print(f'Time_step in step: {self.time_step}')
        # Time steps we are not giving 'self.day*self.iterations+self.time_step' 
        # because right now we are only using same 24 hrs price data for whole training
        # change it when you will be trianing on years training data 
        # self.grid.set_time(self.time_step % 24)
        absolute_step = self.day * self.iterations + self.time_step
        self.grid.set_time(absolute_step) # setting current time step for grid functions
        reward = 0

        # print(f'Day no. with time step in step function: {self.day*self.iterations+self.time_step}, day: {self.day}, timestep: {self.time_step}')

        # assigning action indeces to corresponding components
        # 0 -> DG and 1 -> Battery
        #print(f'Before Battery SOC: {self.battery.SOC}')
        # dg_action = action[0]
        # battery_action = action[1]
        battery_action = action
        #print(f'Battery action: {battery_action}')
       
        # DG action executed
        # self.dg.step(dg_action)

        # Battery action executed
        self.battery.step(battery_action)
        #print(f'Power charg/disch: {self.battery.energy_change}')
        #print(f'After Battery SOC: {self.battery.SOC}')

        # Get the energy generated by the DER without normalized form
        # der_generation = self.generation.current_generation(self.day*self.iterations + self.time_step)
        
        # Total current of all the DGs
        # total_current_output = der_generation + self.dg.current_output - self.battery.energy_change
        total_current_output = self.battery.energy_change     # Here the only generation or demand -> Ch/Dis
        # print(f'Total Gen: {total_current_output}')    # +ive/-ive means Ch & Dis 
        self.total_current_output = total_current_output
    
        # Total current load without normalized forms
        # total_loads = self.load.current_load(self.day*self.iterations + self.time_step)
        #total_loads = self.load.current_load(self.time_step%24)
        # print("Total loads",total_loads)

        # Computing unbalance
        # unbalance = total_current_output - total_loads
        # print(f'Unbalance: {unbalance}')

        # excess_penalty=0
        # deficient_penalty=0
        excess_penalty = 0
        deficient_penalty = 0
        sell_benefit=0
        buy_cost=0
        # self.excess=0
        # self.shedding=0

        # logic here is: if unbalance >0 then it is production excess, so the excessed output should sold to power grid to get benefits 
        # if unbalance >= 0:# it is now in excess condition
        #     if unbalance <= self.grid.exchange_ability:
        #         sell_benefit = self.grid._get_cost(unbalance)   #sell money to grid is little [0.029,0.1]
        #     else:
        #         sell_benefit = self.grid._get_cost(self.grid.exchange_ability)
        #         #real unbalance that even grid could not meet 
        #         self.excess = unbalance - self.grid.exchange_ability
        #         #excess_penalty=self.excess*self.penalty_coefficient
        #         #excess_penalty=(self.excess**2)*self.penalty_coefficient
        #         excess_penalty = self.penalty_coefficient
        # else:# unbalance <0, its load shedding model, in this case, deficient penalty is used 
        #     if abs(unbalance)<=self.grid.exchange_ability:
        #         buy_cost=self.grid._get_cost(abs(unbalance))
        #     else:
        #         buy_cost=self.grid._get_cost(self.grid.exchange_ability)
        #         self.shedding=abs(unbalance)-self.grid.exchange_ability
        #         #deficient_penalty=self.shedding*self.penalty_coefficient
        #         #deficient_penalty=(self.shedding**2)*self.penalty_coefficient
        #         deficient_penalty = self.penalty_coefficient


        if total_current_output >= 0:  # charging mode (+ive energy change) so importing power from grid
            buy_cost = self.grid._get_buy_cost(total_current_output)
        else:  # discharging mode <0 so selling power to grid
            sell_benefit = self.grid._get_sell_benefit(abs(total_current_output))

        battery_cost=self.battery._get_cost(self.battery.energy_change) # we set it as 0 this time 
        if VERBOSE:
            print(f"sell:    {sell_benefit}")
            print(f"buy:     {buy_cost}")
            print(f"battery: {battery_cost}")
        # dg_cost = self.dg._get_cost(self.dg.current_output)
    
        # SOC constraints
        # if (self.battery.current_capacity < self.battery.min_soc or self.battery.current_capacity > self.battery.max_soc):
        #     soc_penalty = self.battery.pen_soc

        # if self.battery.current_capacity < self.battery.min_soc:
        #     soc_penalty = (self.battery.pen_soc)*(abs(self.battery.current_capacity-self.battery.min_soc))
        # else:
        #     soc_penalty = (self.battery.pen_soc)*(abs(self.battery.current_capacity-self.battery.max_soc))

        #print(f'soc penalties: {soc_penalty} and {self.battery.pen_soc}')

        # reward = -0.01*(battery_cost + dg_cost - sell_benefit + buy_cost) - (excess_penalty + deficient_penalty)
        # reward =  -1*(excess_penalty + deficient_penalty + soc_penalty)
        # reward = -0.01*(battery_cost + dg_cost - sell_benefit + buy_cost)
        # reward =  -1*(battery_cost + dg_cost - sell_benefit + buy_cost) - (excess_penalty + deficient_penalty + soc_penalty)
        # reward =  -0.01*(battery_cost - sell_benefit + buy_cost) - (soc_penalty)
        # reward =  -0.01*(battery_cost - sell_benefit + buy_cost)

        # Reward selection based on configured mode.
        if self.reward_mode == "similar_factors":
            reward, operation_cost = self._similar_factors_reward(
                sell_benefit=sell_benefit,
                buy_cost=buy_cost,
                battery_cost=battery_cost,
                battery_action=battery_action,
            )
        elif self.reward_mode == "simple_similar":
            reward = self._simple_similar_reward(
                sell_benefit=sell_benefit,
                buy_cost=buy_cost,
                battery_cost=battery_cost,
            )
            operation_cost = 1.05 * buy_cost - 0.95 * sell_benefit + 0.01 * battery_cost
        elif self.reward_mode == "price_aware":
            reward, operation_cost = self._price_aware_reward(
                sell_benefit=sell_benefit,
                buy_cost=buy_cost,
                battery_cost=battery_cost,
                battery_action=battery_action,
            )
        elif self.reward_mode == "profit_energy_soc":
            reward, operation_cost = self._profit_energy_soc_reward(
                sell_benefit=sell_benefit,
                buy_cost=buy_cost,
                battery_cost=battery_cost,
            )
        else:
            reward = self._simple_reward(
                sell_benefit=sell_benefit,
                buy_cost=buy_cost,
                battery_cost=battery_cost,
            )
            soc_t = float(self.battery.current_capacity)
            soc_penalty = self.soc_penalty * (
                (max(0.0, self.boundary_soc_low - soc_t) ** 2)
                + (max(0.0, soc_t - self.boundary_soc_high) ** 2)
            )
            operation_cost = buy_cost - sell_benefit + 0.01 * battery_cost + soc_penalty

        self.prev_battery_action = float(battery_action)
        self.prev_soc = float(self.battery.current_capacity)
        
        # Operational cost without penalty
        # self.operation_cost = battery_cost+dg_cost+buy_cost-sell_benefit

        ## some modification which must be reconsider in future problems

        # self.operation_cost = battery_cost+buy_cost-sell_benefit

        self.operation_cost = operation_cost
        self.op_co_woDC = buy_cost-sell_benefit

        
        '''
        - Unbalance within the main grid limits
        - Multiply by -1 to get +ive P_grid for import and -ive for export
        - Real unbalance that gets penalised beyond main grid limits
        '''
        # self.unbalance = (-1.0)*unbalance
        #print(f'Self Unbalance->grid: {self.unbalance}')         
        # self.real_unbalance = self.shedding + self.excess  

        #print(unbalance)

        # Proceed to next timestep.
        self.time_step += 1
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state()


        terminal = self.time_step == self.iterations
        # if terminal:

        #     # # reward if battery is charged
        #     # reward += abs(reward * self.battery.SOC / 2)
        info = self._build_info()
        return state, reward , terminal, info

    def reset(self,day=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if day == None:
            self.day= random.randint(self.day0,self.dayn-1)
        else:
            self.day = day
        # print("Day:", self.day)
        self.time_step = 0
        self.unbalance = 0
        self.prev_battery_action = 0.0

        # Reset the battery to a random state
        self.in_SOC = self.battery.reset()
        self.prev_soc = float(self.in_SOC)
        
        # Reset DG to 0
        # self.dg.reset()

        return self._build_state()

    def reset_all(self,day=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if day == None:
            # self.day = random.randint(self.day0, self.dayn-1)
            self.day= self.day0
        else:
            self.day = day
        # print("Day:", self.day)
        self.time_step = 0
        self.prev_battery_action = 0.0
        in_SOC = self.battery.reset()
        self.prev_soc = float(in_SOC)
        self.high_price = 0
        self.loads.clear()
        self.loads = [self._create_load(*self._create_load_parameters()) for _ in range(self.num_loads)]

        return self._build_state()

    def render(self, mode = None):
        # LOADS_RENDER.append(self.load.current_load(self.day*self.iterations + self.time_step-1))
        BATTERY_RENDER.append(self.battery.SOC)
        BATTERY_CH_DCH_RANDER.append(self.battery.energy_change)
        # ENERGY_GENERATED_RENDER.append(self.generation.current_generation(self.day*self.iterations+self.time_step-1))
        # ENERGY_GRID_RENDER.append(self.unbalance)
        # ENERGY_R_UNBALANCE_RENDER.append(self.real_unbalance)
        # DG_GEN_RENDER.append(self.dg.current_output)

        # print(f'Grid price in render function: {self.grid.sell_prices[self.time_step-1]}, time_step: {self.time_step}')
        # GRID_PRICES_SELL_RENDER.append(self.grid.sell_prices[self.time_step-1])

        # print(f'Grid price in render function: {self.grid.sell_prices[self.day*self.iterations+self.time_step-1]}, time_step: {self.day*self.iterations+self.time_step-1}')
        
        GRID_PRICES_SELL_RENDER.append(self._price_at(self.day*self.iterations+self.time_step-1))
        OPERATION_COST.append(self.operation_cost)
        OP_CO_woDC.append(self.op_co_woDC)

        if self.time_step==self.iterations:
            
            if mode == 'Eval':
                print(f'Initial SOC[0]: {self.in_SOC}')
                print(f'Bat Ch/Dch: {BATTERY_CH_DCH_RANDER}')
                print(f'Bat SOC: {BATTERY_RENDER}')
                # print(f'DG output: {DG_GEN_RENDER}')
                #fig=plt.figure()
                print(f'Total cost: {np.sum(np.array(OPERATION_COST))}')
                print(f'Total cost without degradation cost: {np.sum(np.array(OP_CO_woDC))}')
                self._plot_price_and_charge_discharge()


                #     # Define additional data
                #     additional_data = {
                #         'Parameter': ['Initial SOC', 'Total Cost', 'Total Cost without Degradation Cost'],
                #         'Value': [self.in_SOC, np.sum(np.array(OPERATION_COST)), np.sum(np.array(OP_CO_woDC))]  # Directly using the single value
                #     }

                #     # Convert BATTERY_CH_DCH_RANDER and BATTERY_RENDER to DataFrame
                #     df_ch_dch = pd.DataFrame({'Hour': np.arange(len(BATTERY_CH_DCH_RANDER)), 'Battery Ch/Dch': BATTERY_CH_DCH_RANDER})
                #     df_soc = pd.DataFrame({'Hour': np.arange(len(BATTERY_RENDER)), 'Battery SOC': BATTERY_RENDER})

                #     # Construct the filename
                #     filename = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\'
                #                 f'Res_LR{0.01}_'
                #                 f'SED{10000}_'
                #                 f'DF{0.99}_for_paper.xlsx')

                # # Check if file exists and create or append accordingly
                # try:
                #     with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                #         # Define sheet names
                #         sheet_name_params = 'Parameters'
                #         sheet_name_ch_dch = 'Battery_Ch_Dch'
                #         sheet_name_soc = 'Battery_SOC'
                #         # Write data to Excel
                #         pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name_params, index=False)
                #         df_ch_dch.to_excel(writer, sheet_name=sheet_name_ch_dch, index=False)
                #         df_soc.to_excel(writer, sheet_name=sheet_name_soc, index=False)
                #     print(f"Data successfully saved to {filename} in sheets: {sheet_name_params}, {sheet_name_ch_dch}, {sheet_name_soc}")
                # except Exception as e:
                #     print(f"An error occurred while saving data: {e}")



                # Define additional data
                # additional_data = {
                #     'Parameter': ['Initial SOC', 'Total Cost', 'Total Cost without Degradation Cost'],
                #     'Value': [self.in_SOC, np.sum(np.array(OPERATION_COST)), np.sum(np.array(OP_CO_woDC))]  # Directly using the single value
                # }

                # Convert BATTERY_CH_DCH_RANDER and BATTERY_RENDER to DataFrame
                # df_ch_dch = pd.DataFrame({'Hour': np.arange(len(BATTERY_CH_DCH_RANDER)), 'Battery Ch/Dch': BATTERY_CH_DCH_RANDER})
                # df_soc = pd.DataFrame({'Hour': np.arange(len(BATTERY_RENDER)), 'Battery SOC': BATTERY_RENDER})

                # Construct the filename
                # un comment this file name only

                # filename = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\'
                #             f'Res_LR{0.01}_'
                #             f'SED{10000}_'
                #             f'DF{1}.xlsx')

                # # Append additional data
                # with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                #     pd.DataFrame(additional_data).to_excel(writer, sheet_name='Parameters', index=False)
                #     df_ch_dch.to_excel(writer, sheet_name='Battery_Ch_Dch', index=False)
                #     df_soc.to_excel(writer, sheet_name='Battery_SOC', index=False)

                #     # Append additional data
                #     with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                #         pd.DataFrame(additional_data).to_excel(writer, sheet_name='Parameters', index=False)
                #         df_ch_dch.to_excel(writer, sheet_name='Battery_Ch_Dch', index=False)
                #         df_soc.to_excel(writer, sheet_name='Battery_SOC', index=False)


                '''
                Evaluation plots
                '''

                # # Create subplots
                # fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

                # # Set font properties globally
                # plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})

                # # Plot Energy Price
                # axs[0].grid(True, linestyle='-', color='lightgrey', zorder=1, which='both')
                # axs[0].bar(np.arange(0, 24), GRID_PRICES_SELL_RENDER, color='dodgerblue', zorder=3)  # Bars in the foreground
                # axs[0].set_title("Energy Price")
                # axs[0].set_ylabel("Energy Price €")

                # # Plot ESS SOC
                # axs[1].grid(True, linestyle='-', color='lightgrey', zorder=1, which='both')
                # axs[1].bar(np.arange(0, 24), BATTERY_RENDER, color='mediumseagreen', zorder=3)  # Bars in the foreground
                # axs[1].set_title("BSS SOC")
                # axs[1].set_ylabel("Energy (kWh)")

                # charge = np.maximum(BATTERY_CH_DCH_RANDER, 0)  # Only positive values for charging
                # discharge = np.minimum(BATTERY_CH_DCH_RANDER, 0)  # Only negative values for discharging

                # # Plot charge and discharge with different colors
                # axs[2].grid(True, linestyle='-', color='lightgrey', zorder=1, which='both')
                # axs[2].bar(np.arange(0, 24), charge, color='green', label='Charge', zorder=3)
                # axs[2].bar(np.arange(0, 24), discharge, color='red', label='Discharge', zorder=3)

                # # Title, labels, and legend
                # axs[2].set_title("BSS Charge/Discharge")
                # axs[2].set_ylabel("Power (kW)")
                # axs[2].legend(loc='upper right')
                # axs[2].legend(framealpha=0)

                # # Add x-label to the bottom subplot
                # axs[2].set_xlabel("Time (h)")

                # # Construct the filename
                # filename1 = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\Optuna_best_hyper-params.png')

                # Construct the filename
                # un comment this filename only 

                # filename1 = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\'
                #             f'FigSOC_LR{0.01}_'
                #             f'SED{10000}_'
                #             f'DF{1}.png')

                # # Adjust layout
                # plt.savefig(filename1, dpi=300, bbox_inches='tight')

                # # Adjust layout
                # plt.savefig(filename1, dpi=600, bbox_inches='tight')

                # plt.tight_layout()
                # plt.show()


            # LOADS_RENDER.clear()
            BATTERY_RENDER.clear()
            GRID_PRICES_SELL_RENDER.clear()
            # ENERGY_R_UNBALANCE_RENDER.clear()
            # ENERGY_GRID_RENDER.clear()
            # ENERGY_GENERATED_RENDER.clear()
            BATTERY_CH_DCH_RANDER.clear()
            # DG_GEN_RENDER.clear()
            OPERATION_COST.clear()
            OP_CO_woDC.clear()

        # return abs(self.real_unbalance), self.operation_cost
        return self.operation_cost
    
    '''
    Evaluation Plots for 1 complete day of scheduling
    # '''
    # def plot_ren(self, ENERGY_GENERATED_RENDER, ENERGY_GRID_RENDER, DG_GEN_RENDER, 
    #              BATTERY_CH_DCH_RANDER, ENERGY_R_UNBALANCE_RENDER, LOADS_RENDER):

    #     data = np.array([ENERGY_GENERATED_RENDER, ENERGY_GRID_RENDER, DG_GEN_RENDER, \
    #                      BATTERY_CH_DCH_RANDER, ENERGY_R_UNBALANCE_RENDER])

    #     data_shape = np.shape(data)

    #     # Take negative and positive data apart and cumulate
    #     def get_cumulated_array(data, **kwargs):
    #         cum = data.clip(**kwargs)
    #         cum = np.cumsum(cum, axis=0)
    #         d = np.zeros(np.shape(data))
    #         d[1:] = cum[:-1]
    #         return d  

    #     cumulated_data = get_cumulated_array(data, min=0)
    #     cumulated_data_neg = get_cumulated_array(data, max=0)

    #     # Re-merge negative and positive data.
    #     row_mask = (data<0)
    #     cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    #     data_stack = cumulated_data

    #     cols = ["g", "y", "b", "c", "m"]

    #     ax = plt.subplot(3,1, 3)
    #     ax.set_facecolor("silver")
    #     #ax.set_xlabel("Time (h)")
    #     ax.yaxis.grid(True)

    #     for i in np.arange(0, data_shape[0]):
    #         ax.bar(np.arange(data_shape[1]), data[i], bottom=data_stack[i], color=cols[i],)
        
    #     plt.plot(np.arange(data_shape[1]), np.array(LOADS_RENDER), color = 'r')
    #     plt.legend(['Load', 'RES', 'Imp/Exp', 'Dg_gen', 'Ch/Dch', 'P_unb'])
    #     plt.xlabel("Time (h)")
    #     plt.show()


    def close(self):
        """
        Nothing to be done here, but has to be defined
        """
        return

    def seedy(self, s):
        """
        Set the random seed for consistent experiments
        """
        random.seed(s)
        np.random.seed(s)


if __name__ == '__main__':
    # Batch runner: run `total_episodes` in batches of `batch_size` and print summaries.
    total_episodes = 200
    batch_size = 50

    env = MicroGridEnv()
    env.seedy(1)

    episode_rewards = []
    log_lines = []
    for ep in range(1, total_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        while True:
            action = env.action_space.sample()
            if VERBOSE:
                print(action)
            state, reward, terminal, _ = env.step(action)
            env.render()
            ep_reward += float(reward)
            if terminal:
                break

        episode_rewards.append(ep_reward)
        log_lines.append(f"Episode {ep}: total_reward={ep_reward}\n")

        # Batch summary
        if ep % batch_size == 0:
            batch_start = ep - batch_size + 1
            batch_rewards = episode_rewards[batch_start - 1 : ep]
            batch_total = float(sum(batch_rewards))
            batch_avg = float(np.mean(batch_rewards))
            summary = f"Batch {batch_start}-{ep}: total={batch_total:.6f}, avg={batch_avg:.6f}"
            print(summary)
            log_lines.append(summary + "\n")

    # Save log
    with open("run_200_batch50.log", "w") as f:
        f.writelines(log_lines)

    print("Run complete. Log written to run_200_batch50.log")

