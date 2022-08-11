# Author: Anselme Ndikumana
# Interpreter: python3.8.10
################################################################################
# Needed packages
import random
import time
import copy
import itertools
from itertools import zip_longest
from matplotlib import cm
import seaborn as sns
import cvxpy as cp
import networkx as nx
import numpy.random
from gym import Env
from gym.spaces import Discrete, Box
from math import isnan
import pandas as pd
import os
import pdb
import math
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate, make_interp_spline
from scipy.interpolate import make_interp_spline, BSpline
from MM1V7 import Simulation
import ray
from itertools import zip_longest
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete, Box
import ray
from typing import Deque, Union
import torch
import yaml
import torch
import warnings
import random
from communication import wireless_network, fronthaul_network
warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
################################################################################
# Get auction values generated from auction.py and distribute resources to the O-DUs. This can be considered as part
# of closed loop 2


def distributing_resource():
    # Get auction values generated from auction.py and distribute resources to the O-DUs. This can be considered as part
    # of closed loop 2
    auction_values = pd.read_csv('dataset/RB_allocation_via_auction.csv', index_col=0)
    RBS_all = auction_values["RBS"].values
    RB_allocation_via_auction = RBS_all[:-1]
    Number_MIMO_layers = 4
    print("RB_allocation_via_auction", RB_allocation_via_auction)
    channel_bandwidth = 100  # Channel bandwidth: 100 MHZ
    Num_O_RU = int(np.random.uniform(low=6, high=12))
    Num_vO_DUs = 3
    InP_max_RBs = RBS_all[-1]
    print("InP_max_RBs", InP_max_RBs)
    ##########################
    # Distribute RBs to O_DUs equally
    b_d = InP_max_RBs // Num_vO_DUs
    RB_vO_DUs = np.empty(Num_vO_DUs)
    RB_vO_DUs.fill(b_d)
    RB_vO_DUs_copy = copy.deepcopy(RB_vO_DUs)
    print("RB_vO_DUs", RB_vO_DUs)
    # round-robin policy to allocate slice/services and resource block to vO_DUs cyclically
    RAN_vDUs = {}
    for i in range(Num_vO_DUs):
        RAN_vDUs.setdefault(i, []).append(0)  # Initialization
    # print(RAN_vDUs)
    RBvODUs = []
    i = 0
    for j in range(0, len(RB_allocation_via_auction)):
        RAN_vDUs[i].append(RB_allocation_via_auction[j])
        RB_vO_DUs[i] = RB_vO_DUs[i] - RB_allocation_via_auction[j]
        k1 = j + 1
        if k1 < len(RB_allocation_via_auction):
            if RB_vO_DUs[i] <= RB_allocation_via_auction[k1]:
                i = i + 1
        RBvODUs.append(RB_allocation_via_auction[j])
    print("Total_RBs_Winnner", RBvODUs)
    services = {}
    for i in range(0, len(RBvODUs)):
        services["Service{0}".format(i)] = RBvODUs[i]
    print("Services", services)

    # We use dictionary where keys are vO_DUs and slice with associated slices are the values
    # print("RBs_tenants_to_vDUs", RAN_vDUs)  #
    y_decision_d_k_c = []
    for i in range(0, Num_vO_DUs):
        vdu_i = RAN_vDUs[i]
        vdu_i = list(vdu_i)
        for j in range(0, len(vdu_i)):
            y_decision_d_k_c.append(1)
    i = 0
    # Remove zero resources allocation
    for k, v in RAN_vDUs.items():
        # print("v",v)
        if v[i] == 0 and RAN_vDUs[k] == 0:
            del RAN_vDUs[k]
        elif v[i] == 0:
            v.remove(v[i])
        else:
            i = i + 1
    # Ignore the RAN_vDUs which does not have resource to manage
    empty_keys = [k for k, v in RAN_vDUs.items() if not v]
    for k in empty_keys:
        del RAN_vDUs[k]
    print("Final RBs_tenants_to_vDUs", RAN_vDUs)  #
    return RAN_vDUs, services, RB_vO_DUs, InP_max_RBs, RB_vO_DUs_copy


RAN_vDUs, services, RB_vO_DUs,  InP_max_RBs, RB_vO_DUs_copy = distributing_resource()
# CQI: channel quality indicator
CQI_Table = [0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266,
             6.9141, 7.4063]


def getList(p_dict):
    return p_dict.keys()


# Get routing information generated using veroviz in driving.py

time_between_ORUs_GroundVehicle = pd.read_csv('dataset/time_between_ORUs_GroundVehicle.csv',
                                              index_col=0)
time_between_ORUs_FlyingVehicle = pd.read_csv('dataset/time_between_ORUs_FlyingVehicle.csv',
                                              index_col=0)
# print(distance_between_ORUs_GroundVehicle)
time_path_groundV = time_between_ORUs_GroundVehicle.to_numpy()
time_path_flyingV = time_between_ORUs_FlyingVehicle.to_numpy()
time_path_groundV = nx.from_numpy_array(time_path_groundV)
time_path_flyingV = nx.from_numpy_array(time_path_flyingV)
# print("time_path_flyingV", time_path_flyingV.edges(data=True))
# print("time_path_groundV", time_path_groundV.edges(data=True))
# print(list(nx.all_simple_paths(time_path_groundV, source=0, target=5)))
# print(list(nx.all_simple_paths(time_path_flyingV, source=0, target=5)))

time_flyingv = []
for i in range(0, 5):
    f = time_path_flyingV[i][i + 1]["weight"]
    time_flyingv.append(f)
time_groundv = []
for i in range(0, 5):
    f = time_path_groundV[i][i + 1]["weight"]
    time_groundv.append(f)

distance_between_ORUs_GroundVehicle = pd.read_csv('dataset/distance_between_ORUs_GroundVehicle.csv', index_col=0)
distance_between_ORUs_FlyingVehicle = pd.read_csv('dataset/distance_between_ORUs_FlyingVehicle.csv', index_col=0)
# print(distance_between_ORUs_GroundVehicle)
distance_path_groundV = distance_between_ORUs_GroundVehicle.to_numpy()
distance_path_flyingV = distance_between_ORUs_FlyingVehicle.to_numpy()
distance_path_groundV = nx.from_numpy_array(distance_path_groundV)
distance_path_flyingV = nx.from_numpy_array(distance_path_flyingV)
# print("distance_path_flyingV", distance_path_flyingV.edges(data=True))
# print("distance_path_groundV", distance_path_groundV.edges(data=True))
# print(list(nx.all_simple_paths(distance_path_groundV, source=0, target=5)))
# print(list(nx.all_simple_paths(distance_path_flyingV, source=0, target=5)))

distance_flyingv = []
for i in range(0, 5):
    f = distance_path_flyingV[i][i + 1]["weight"]
    distance_flyingv.append(f)

distance_groundv = []
for i in range(0, 5):
    f = distance_path_groundV[i][i + 1]["weight"]
    distance_groundv.append(f)
# print(" distance_flyingv", distance_flyingv)
# print(" distance_groundv.", distance_groundv)
# print(" time_groundv", time_groundv)
# print(" time_flyingv", time_flyingv)
df_routing = pd.DataFrame()
df_routing['d_flying_v'] = np.around(distance_flyingv, decimals=2)
df_routing['t_flying_v'] = np.around(time_flyingv, decimals=2)
df_routing['d_gr_v'] = np.around(distance_groundv, decimals=2)
df_routing['t_gr_v'] = np.around(time_groundv, decimals=2)
with open('mytable.tex', 'w') as tf:
    tf.write(df_routing.to_latex())


Num_aerial_vehicle = 3
Num_ground_vehicle = int(np.random.uniform(low=10, high=35))
vehicles = Num_aerial_vehicle + Num_ground_vehicle
# We assume that vehicle 1 to 3 are flying vehicles and the remaining are the ground vehicles
services_only = list(getList(services))
# We choose the service of available tenants
# 0-5QI-86 advanced driving, 1-5QI-85 remote driving, 1-5QI-84 intelligent transportation
# 1-5QI-84 vehicle message for delay critical GBR
 # 1-5QI-6 video for Non-GBR

delay_budget = [5, 5, 30, 10, 300, 30, 20]  # in terms of milisecond
delay_budget = zip(services_only, delay_budget)
delay_budget = dict(delay_budget)  # Dictionary of services as key and corresponding delay as value
# print("delay_budget", delay_budget)
delay_budget_list = list(delay_budget.values())
Utilization_vODU1 = []
Utilization_vODU2 = []
Utilization_vODU3 = []
orchestration_parameter_service0 = []
orchestration_parameter_service1 = []
orchestration_parameter_service2 = []
orchestration_parameter_service3 = []
orchestration_parameter_service4 = []
orchestration_parameter_service5 = []
orchestration_parameter_service6 = []

all_vehicle_service0 = []
all_vehicle_service1 = []
all_vehicle_service2 = []
all_vehicle_service3 = []
all_vehicle_service4 = []
all_vehicle_service5 = []
all_vehicle_service6 = []

Vehicle_vdu1 = []
Vehicle_vdu2 = []
Vehicle_vdu3 = []
requirement_satisfaction_parameter_s0 = 0
requirement_satisfaction_parameter_s1 = 0
requirement_satisfaction_parameter_s2 = 0
requirement_satisfaction_parameter_s3 = 0
requirement_satisfaction_parameter_s4 = 0
requirement_satisfaction_parameter_s5 = 0
requirement_satisfaction_parameter_s6 = 0

requirement_satisfaction_parameter_s0_array = []
requirement_satisfaction_parameter_s1_array = []
requirement_satisfaction_parameter_s2_array = []
requirement_satisfaction_parameter_s3_array = []
requirement_satisfaction_parameter_s4_array = []
requirement_satisfaction_parameter_s5_array = []
requirement_satisfaction_parameter_s6_array = []

fronthaul_capacity = fronthaul_network()
maximize_utilization_satifisfaction = []
delay_budget_fulfillment_service0 = []
delay_budget_fulfillment_service1 = []
delay_budget_fulfillment_service2 = []
delay_budget_fulfillment_service3 = []
delay_budget_fulfillment_service4 = []
delay_budget_fulfillment_service5 = []
delay_budget_fulfillment_service6 = []
penality_fronthaul = 0.002
penality_violating_rb = 0.01
count_call = 0
Reward_loo11 = []
loop1_calling = []
n_iteration = 200


def ClosedLoop1(RAN_vDUs):
    for n in range(n_iteration):
        l = 0
        # We choose the services of available tenants
        vehicles_services = {}  # Dictionary of vehicles as key and value as services
        for j in range(0, vehicles):
            vehicles_services.setdefault(l, []).append(np.random.choice(services_only))
            vehicles_services.setdefault(l, []).append(np.random.choice(services_only))
            l = l + 1
        packet_size = np.random.uniform(low=1e3, high=1e4)  # in terms of kilobytes

        # Map demands of vehicles to RAN_vDUs
        m = 0
        rb_vehicle_service0 = 0
        rb_vehicle_service1 = 0
        rb_vehicle_service2 = 0
        rb_vehicle_service3 = 0
        rb_vehicle_service4 = 0
        rb_vehicle_service5 = 0
        rb_vehicle_service6 = 0
        vehicle_service0 = 0
        vehicle_service1 = 0
        vehicle_service2 = 0
        vehicle_service3 = 0
        vehicle_service4 = 0
        vehicle_service5 = 0
        vehicle_service6 = 0
        RB_service0 = 0
        RB_service1 = 0
        RB_service2 = 0
        RB_service3 = 0
        RB_service4 = 0
        RB_service5 = 0
        RB_service6 = 0
        distance_vehicle_ORU = 0

        for k, v in vehicles_services.items():
            if v[m] == 'Service0':
                vehicle_service0 = vehicle_service0 + 1
                RB_service0 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[0]
                else:
                    distance_vehicle_ORU = distance_groundv[0]
            elif v[m] == 'Service1':
                vehicle_service1 = vehicle_service1 + 1
                RB_service1 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[0]
                else:
                    distance_vehicle_ORU = distance_groundv[0]
            elif v[m] == 'Service2':
                vehicle_service2 = vehicle_service2 + 1
                RB_service2 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[1]
                else:
                    distance_vehicle_ORU = distance_groundv[1]
            elif v[m] == 'Service3':
                vehicle_service3 = vehicle_service3 + 1
                RB_service3 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[2]
                else:
                    distance_vehicle_ORU = distance_groundv[2]
            elif v[m] == 'Service4':
                vehicle_service4 = vehicle_service4 + 1
                RB_service4 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[3]
                else:
                    distance_vehicle_ORU = distance_groundv[3]
            elif v[m] == 'Service5':
                vehicle_service5 = vehicle_service5 + 1
                RB_service5 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[4]
                else:
                    distance_vehicle_ORU = distance_groundv[4]
            elif v[m] == 'Service6':
                vehicle_service6 = vehicle_service6 + 1
                RB_service6 = int(services.get(v[m]))
                if k <= 3:
                    distance_vehicle_ORU = distance_flyingv[4]
                else:
                    distance_vehicle_ORU = distance_groundv[4]
            else:
                m + 1
        all_vehicle_service0.append(vehicle_service0)
        all_vehicle_service1.append(vehicle_service1)
        all_vehicle_service2.append(vehicle_service2)
        all_vehicle_service3.append(vehicle_service3)
        all_vehicle_service4.append(vehicle_service4)
        all_vehicle_service5.append(vehicle_service5)
        all_vehicle_service6.append(vehicle_service6)
        # Map services with vODUs that manage the slices, where each service corresponds to one service
        # that needs to be used by vehicles
        # Each vODU can manage up to 5 services for testing, if slice has not assigned resource, it means it does not exist
        # We assume the buffer sizes and buffer_threshold for the services are the same

        Buffer_size = 100
        buffer_threshold = 75

        for k, v in RAN_vDUs.items():
            v = list(v)
            ##################################################################################
            # Service or slice 0#
            if (RB_service0 != 0.0) and (RB_service0 in v):
                g0 = v.index(RB_service0)
                rb_vehicle_service0 = v[g0] / vehicle_service0
                rb_vehicle_service0 = int(rb_vehicle_service0)
                MCS_spectrum_efficiency_service0 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)

                throughput_service0, achievable_data_rate_service0 = \
                    wireless_network(vehicle_service0, rb_vehicle_service0, MCS_spectrum_efficiency_service0,
                                     distance_vehicle_ORU)
                event_calendar_service0 = []
                clock_time0 = []
                arrival_times0 = []
                service_times0 = []
                departures0 = []
                number_vehicles0 = []
                # Call the queue model for the service 0
                s0 = Simulation(vehicle_service0)
                s0.advance_time()
                a_service0 = s0.clock
                b_service0 = s0.t_arrival
                c_service0 = s0.tmp_time
                d_service0 = s0.t_depart
                e_service0 = s0.num_in_system
                s0.vehicles = vehicle_service1
                s0.vehicles = vehicle_service1
                clock_time0.append(a_service0)
                arrival_times0.append(b_service0)
                service_times0.append(c_service0)
                departures0.append(d_service0)
                number_vehicles0.append(vehicle_service0)
                event_list_service0 = {"Iteration": "I_" + str(n), "Event_time": a_service0, "Arrival_time": b_service0,
                                       "Service_time": c_service0,
                                       "Departure_time": d_service0, "number_vehicles": vehicle_service0}
                event_calendar_service0.append(event_list_service0)
                queuing_delay_service0 = 1 / abs((b_service0 - d_service0))
                delay_service0 = (packet_size / (achievable_data_rate_service0)) + queuing_delay_service0 + \
                                 (packet_size / fronthaul_network())
                # print("delay_service0", packet_size / (achievable_data_rate_service0))
                if delay_service0 <= delay_budget_list[0]:
                    delay_budget_fulfillment_service0.append(1)
                else:
                    delay_budget_fulfillment_service0.append(0)
                np_packet_s0 = np.random.uniform(low=7, high=40)
                Expected_packet_service0 = np_packet_s0 * vehicle_service0
                queue_status_parameter_S0 = max(abs(Buffer_size - Expected_packet_service0), buffer_threshold)

                requirement_satisfaction_parameter_s0 = sum(delay_budget_fulfillment_service0) / \
                                                        len(delay_budget_fulfillment_service0)
                # print("delay_budget_fulfillment_service0) ", delay_budget_fulfillment_service0)

                requirement_satisfaction_parameter_s0_array.append(requirement_satisfaction_parameter_s0)
                if queue_status_parameter_S0 == buffer_threshold:
                    orchestration_parameter_service0.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S0 > buffer_threshold:
                    orchestration_parameter_service0.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S0 == Buffer_size:
                    orchestration_parameter_service0.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service0.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation
            reward_service0 = sum(orchestration_parameter_service0) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001) * vehicle_service0)) + \
                              penality_violating_rb * (RB_service0 - rb_vehicle_service0)

            ##################################################################################
            # Service or slice 1#

            if (RB_service1 != 0.0) and (RB_service1 in v):
                g1 = v.index(RB_service1)
                rb_vehicle_service1 = v[g1] / vehicle_service1
                rb_vehicle_service1 = int(rb_vehicle_service1)
                MCS_spectrum_efficiency_service1 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)

                throughput_service1, achievable_data_rate_service1 = \
                    wireless_network(vehicle_service1, rb_vehicle_service1, MCS_spectrum_efficiency_service1,
                                     distance_vehicle_ORU)
                event_calendar_service1 = []
                clock_time1 = []
                arrival_times1 = []
                service_times1 = []
                departures1 = []
                number_vehicles1 = []
                # Call the queue model for the service 1
                s1 = Simulation(vehicle_service1)
                s1.advance_time()
                a_service1 = s1.clock
                b_service1 = s1.t_arrival
                c_service1 = s1.tmp_time
                d_service1 = s1.t_depart
                e_service1 = s1.num_in_system
                s1.vehicles = vehicle_service1
                s1.vehicles = vehicle_service1
                clock_time1.append(a_service1)
                arrival_times1.append(b_service1)
                service_times1.append(c_service1)
                departures1.append(d_service1)
                number_vehicles1.append(vehicle_service1)
                event_list_service1 = {"Iteration": "I_" + str(n), "Event_time": a_service1, "Arrival_time": b_service1,
                                       "Service_time": c_service1,
                                       "Departure_time": d_service1, "number_vehicles": vehicle_service1}
                event_calendar_service1.append(event_list_service1)
                queuing_delay_service1 = 1 / abs((b_service1 - d_service1))
                delay_service1 = (packet_size / (1 + achievable_data_rate_service1)) + queuing_delay_service1 + \
                                 (packet_size / fronthaul_network())

                if delay_service1 <= delay_budget_list[1]:
                    delay_budget_fulfillment_service1.append(1)
                else:
                    delay_budget_fulfillment_service1.append(0)
                np_packet_s1 = np.random.uniform(low=7, high=40)
                Expected_packet_service1 = np_packet_s1 * vehicle_service1
                queue_status_parameter_S1 = max(abs(Buffer_size - Expected_packet_service1), buffer_threshold)

                requirement_satisfaction_parameter_s1 = (sum(delay_budget_fulfillment_service1)) / \
                                                        len(delay_budget_fulfillment_service1)
                requirement_satisfaction_parameter_s1_array.append(requirement_satisfaction_parameter_s1)

                if queue_status_parameter_S1 == buffer_threshold:
                    orchestration_parameter_service1.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S1 > buffer_threshold:
                    orchestration_parameter_service1.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S1 == Buffer_size:
                    orchestration_parameter_service1.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service1.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation
            reward_service1 = sum(orchestration_parameter_service1) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001)* vehicle_service1)) + \
                              penality_violating_rb * (RB_service1 - rb_vehicle_service1)
            ###################################################################################
            # Service or slice 2#
            if (RB_service2 != 0.0) and (RB_service2 in v):
                a = v.index(RB_service2)
                rb_vehicle_service2 = v[a] / vehicle_service2
                rb_vehicle_service2 = int(rb_vehicle_service2)
                MCS_spectrum_efficiency_service2 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)
                throughput_service2, achievable_data_rate_service2 = \
                    wireless_network(vehicle_service2, rb_vehicle_service2, MCS_spectrum_efficiency_service2,
                                     distance_vehicle_ORU)
                clock_time2 = []
                arrival_times2 = []
                service_times2 = []
                departures2 = []
                number_vehicles2 = []
                event_calendar_service2 = []

                # Call the queue model for the service 2
                s2 = Simulation(vehicle_service2)
                s2.advance_time()
                a_service2 = s2.clock
                b_service2 = s2.t_arrival
                c_service2 = s2.tmp_time
                d_service2 = s2.t_depart
                e_service2 = s2.num_in_system
                s2.vehicles = vehicle_service2
                clock_time2.append(a_service2)
                arrival_times2.append(b_service2)
                service_times2.append(c_service2)
                departures2.append(d_service2)
                number_vehicles2.append(vehicle_service2)
                event_list_service2 = {"Iteration": "I_" + str(n), "Event_time": a_service2, "Arrival_time": b_service2,
                                       "Service_time": c_service2,
                                       "Departure_time": d_service2, "number_vehicles": vehicle_service2}
                event_calendar_service2.append(event_list_service2)
                # print("event_calendar", event_calendar_service2)
                queuing_delay_service2 = 1 / abs((b_service2 - d_service2))
                delay_service2 = packet_size / (1 + achievable_data_rate_service2) + queuing_delay_service2 + \
                                 (packet_size / fronthaul_network())
                if delay_service2 <= delay_budget_list[2]:
                    delay_budget_fulfillment_service2.append(1)
                else:
                    delay_budget_fulfillment_service2.append(0)
                np_packet_s2 = np.random.uniform(low=7, high=40)
                Expected_packet_service2 = np_packet_s2 * vehicle_service2
                queue_status_parameter_S2 = max((Buffer_size - Expected_packet_service2), buffer_threshold)

                requirement_satisfaction_parameter_s2 = (sum(delay_budget_fulfillment_service2)) / \
                                                        len(delay_budget_fulfillment_service2)
                requirement_satisfaction_parameter_s2_array.append(requirement_satisfaction_parameter_s2)

                if queue_status_parameter_S2 == buffer_threshold:
                    orchestration_parameter_service2.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S2 > buffer_threshold:
                    orchestration_parameter_service2.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S2 == Buffer_size:
                    orchestration_parameter_service2.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service2.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation
            reward_service2 = sum(orchestration_parameter_service2) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001)*vehicle_service2)) + \
                              penality_violating_rb * (RB_service2 - rb_vehicle_service2)
            ################################################################################
            # Service or slice 3#
            if (RB_service3 != 0.0) and (RB_service3 in v):
                a = v.index(RB_service3)
                rb_vehicle_service3 = v[a] / vehicle_service3
                rb_vehicle_service3 = int(rb_vehicle_service3)
                MCS_spectrum_efficiency_service3 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)
                throughput_service3, achievable_data_rate_service3 = \
                    wireless_network(vehicle_service3, rb_vehicle_service3, MCS_spectrum_efficiency_service3,
                                     distance_vehicle_ORU)
                clock_time3 = []
                arrival_times3 = []
                service_times3 = []
                departures3 = []
                number_vehicles3 = []
                event_calendar_service3 = []

                # Call the queue model for the service 3
                s3 = Simulation(vehicle_service3)
                s3.advance_time()
                a_service3 = s3.clock
                b_service3 = s3.t_arrival
                c_service3 = s3.tmp_time
                d_service3 = s3.t_depart
                e_service3 = s3.num_in_system
                s3.vehicles = vehicle_service3
                clock_time3.append(a_service3)
                arrival_times3.append(b_service3)
                service_times3.append(c_service3)
                departures3.append(d_service3)
                number_vehicles3.append(vehicle_service3)
                event_list_service3 = {"Iteration": "I_" + str(n), "Event_time": a_service3, "Arrival_time": b_service3,
                                       "Service_time": c_service3,
                                       "Departure_time": d_service3, "number_vehicles": vehicle_service3}
                event_calendar_service3.append(event_list_service3)
                queuing_delay_service3 = 1 / abs((b_service3 - d_service3))
                delay_service3 = packet_size / (1 + achievable_data_rate_service3) + queuing_delay_service3 + \
                                 (packet_size / fronthaul_network())
                if delay_service3 <= delay_budget_list[3]:
                    delay_budget_fulfillment_service3.append(1)
                else:
                    delay_budget_fulfillment_service3.append(0)

                np_packet_s3 = np.random.uniform(low=7, high=40)
                Expected_packet_service3 = np_packet_s3 * vehicle_service3
                queue_status_parameter_S3 = max(abs(Buffer_size - Expected_packet_service3), buffer_threshold)

                requirement_satisfaction_parameter_s3 = (sum(delay_budget_fulfillment_service3)) / len(delay_budget_fulfillment_service3)
                requirement_satisfaction_parameter_s3_array.append(requirement_satisfaction_parameter_s3)

                if queue_status_parameter_S3 == buffer_threshold:
                    orchestration_parameter_service3.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S3 > buffer_threshold:
                    orchestration_parameter_service3.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S3 == Buffer_size:
                    orchestration_parameter_service3.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service3.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation
            reward_service3 = sum(orchestration_parameter_service3) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001) * vehicle_service3)) + \
                              penality_violating_rb * (RB_service3 - rb_vehicle_service3)
            #################################################################################
            # Service or slice 4#
            if (RB_service4 != 0.0) and (RB_service4 in v):
                a = v.index(RB_service4)
                rb_vehicle_service4 = v[a] / vehicle_service4
                rb_vehicle_service4 = int(rb_vehicle_service4)
                MCS_spectrum_efficiency_service4 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)
                throughput_service4, achievable_data_rate_service4 = \
                    wireless_network(vehicle_service4, rb_vehicle_service4, MCS_spectrum_efficiency_service4,
                                     distance_vehicle_ORU)
                clock_time4 = []
                arrival_times4 = []
                service_times4 = []
                departures4 = []
                number_vehicles4 = []
                event_calendar_service4 = []

                # Call the queue model for the service 4
                s4 = Simulation(vehicle_service4)
                s4.advance_time()
                a_service4 = s4.clock
                b_service4 = s4.t_arrival
                c_service4 = s4.tmp_time
                d_service4 = s4.t_depart
                e_service4 = s4.num_in_system
                s4.vehicles = vehicle_service3
                clock_time4.append(a_service4)
                arrival_times4.append(b_service4)
                service_times4.append(c_service4)
                departures4.append(d_service4)
                number_vehicles4.append(e_service4)
                event_list_service4 = {"Iteration": "I_" + str(n), "Event_time": a_service4, "Arrival_time": b_service4,
                                       "Service_time": c_service4,
                                       "Departure_time": d_service4, "number_vehicles": e_service4}
                event_calendar_service4.append(event_list_service4)
                # print("event_calendar", event_calendar_service4)
                queuing_delay_service4 = 1 / abs((b_service4 - d_service4))
                delay_service4 = packet_size / (1 + achievable_data_rate_service4) + queuing_delay_service4 + \
                                 (packet_size / fronthaul_network())
                if delay_service4 <= delay_budget_list[4]:
                    delay_budget_fulfillment_service4.append(1)
                else:
                    delay_budget_fulfillment_service4.append(0)
                np_packet_s4 = np.random.uniform(low=7, high=40)
                Expected_packet_service4 = np_packet_s4 * vehicle_service4
                queue_status_parameter_S4 = max(abs(Buffer_size - Expected_packet_service4), buffer_threshold)

                requirement_satisfaction_parameter_s4 = (sum(delay_budget_fulfillment_service4)) / len(delay_budget_fulfillment_service4)
                requirement_satisfaction_parameter_s4_array.append(requirement_satisfaction_parameter_s4)

                if queue_status_parameter_S4 == buffer_threshold:
                    orchestration_parameter_service4.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S4 > buffer_threshold:
                    orchestration_parameter_service4.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S4 == Buffer_size:
                    orchestration_parameter_service4.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service4.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation
            reward_service4 = sum(orchestration_parameter_service4) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001) * vehicle_service4)) + \
                              penality_violating_rb * (RB_service4 - rb_vehicle_service4)
            # to convert kb to megabyte: packet_size * 0.001
            #################################################################################
            # Service or slice 5#
            if (RB_service5 != 0.0) and (RB_service5 in v):
                a = v.index(RB_service5)
                rb_vehicle_service5 = v[a] / vehicle_service5
                rb_vehicle_service5 = int(rb_vehicle_service5)
                MCS_spectrum_efficiency_service5 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)
                throughput_service5, achievable_data_rate_service5 = \
                    wireless_network(vehicle_service5, rb_vehicle_service5, MCS_spectrum_efficiency_service5,
                                     distance_vehicle_ORU)
                clock_time5 = []
                arrival_times5 = []
                service_times5 = []
                departures5 = []
                number_vehicles5 = []
                event_calendar_service5 = []

                # Call the queue model for the service 5
                s5 = Simulation(vehicle_service5)
                s5.advance_time()
                a_service5 = s5.clock
                b_service5 = s5.t_arrival
                c_service5 = s5.tmp_time
                d_service5 = s5.t_depart
                e_service5 = s5.num_in_system
                s5.vehicles = vehicle_service5
                clock_time5.append(a_service5)
                arrival_times5.append(b_service5)
                service_times5.append(c_service5)
                departures5.append(d_service5)
                number_vehicles5.append(vehicle_service5)
                event_list_service5 = {"Iteration": "I_" + str(n), "Event_time": a_service5, "Arrival_time": b_service5,
                                       "Service_time": c_service5,
                                       "Departure_time": d_service5, "number_vehicles": vehicle_service5}
                event_calendar_service5.append(event_list_service5)
                # print("event_calendar", event_calendar_service5)
                queuing_delay_service5 = 1 / abs((b_service5 - d_service5))
                delay_service5 = packet_size / (1 + achievable_data_rate_service5) + queuing_delay_service5 + \
                                 (packet_size / fronthaul_capacity)
                if delay_service5 <= delay_budget_list[5]:
                    delay_budget_fulfillment_service5.append(1)
                else:
                    delay_budget_fulfillment_service5.append(0)
                np_packet_s5 = np.random.uniform(low=7, high=40)
                Expected_packet_service5 = np_packet_s5 * vehicle_service5
                queue_status_parameter_S5 = max(abs(Buffer_size - Expected_packet_service5), buffer_threshold)
                requirement_satisfaction_parameter_s5 = (sum(delay_budget_fulfillment_service5)) /len(delay_budget_fulfillment_service5)
                requirement_satisfaction_parameter_s5_array.append(requirement_satisfaction_parameter_s5)
                if queue_status_parameter_S5 == buffer_threshold:
                    orchestration_parameter_service5.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S5 > buffer_threshold:
                    orchestration_parameter_service5.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S5 == Buffer_size:
                    orchestration_parameter_service5.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service5.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation
            reward_service5 = sum(orchestration_parameter_service5) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001) * vehicle_service5)) + \
                              penality_violating_rb * (RB_service5 - rb_vehicle_service5)
            #################################################################################
            # Service or slice 6#
            if (RB_service6 != 0.0) and (RB_service6 in v):
                a = v.index(RB_service6)
                rb_vehicle_service6 = v[a] / vehicle_service6
                rb_vehicle_service6 = int(rb_vehicle_service6)
                MCS_spectrum_efficiency_service6 = np.random.choice(CQI_Table)  # MCS (Modulation and Coding Scheme)
                throughput_service6, achievable_data_rate_service6 = \
                    wireless_network(vehicle_service6, rb_vehicle_service6, MCS_spectrum_efficiency_service6,
                                     distance_vehicle_ORU)
                clock_time6 = []
                arrival_times6 = []
                service_times6 = []
                departures6 = []
                number_vehicles6 = []
                event_calendar_service6 = []

                # Call the queue model for the service 6
                s6 = Simulation(vehicle_service6)
                s6.advance_time()
                a_service6 = s6.clock
                b_service6 = s6.t_arrival
                c_service6 = s6.tmp_time
                d_service6 = s6.t_depart
                e_service6 = s6.num_in_system
                s6.vehicles = vehicle_service6
                clock_time6.append(a_service6)
                arrival_times6.append(b_service6)
                service_times6.append(c_service6)
                departures6.append(d_service6)
                number_vehicles6.append(vehicle_service5)
                event_list_service6 = {"Iteration": "I_" + str(n), "Event_time": a_service6, "Arrival_time": b_service6,
                                       "Service_time": c_service6,
                                       "Departure_time": d_service6, "number_vehicles": vehicle_service6}
                event_calendar_service6.append(event_list_service6)
                # print("event_calendar", event_calendar_service5)
                queuing_delay_service6 = 1 / abs((b_service6 - d_service6))
                delay_service6 = packet_size / (1 + achievable_data_rate_service6) + queuing_delay_service6 + \
                                 (packet_size / fronthaul_capacity)
                if delay_service6 <= delay_budget_list[6]:
                    delay_budget_fulfillment_service6.append(1)
                else:
                    delay_budget_fulfillment_service6.append(0)
                np_packet_s6 = np.random.uniform(low=7, high=40)
                Expected_packet_service6 = np_packet_s6 * vehicle_service6
                queue_status_parameter_S6 = max(abs(Buffer_size - Expected_packet_service6), buffer_threshold)
                requirement_satisfaction_parameter_s6 = (sum(delay_budget_fulfillment_service6)) / len(delay_budget_fulfillment_service6)
                requirement_satisfaction_parameter_s6_array.append(requirement_satisfaction_parameter_s6)
                if queue_status_parameter_S6 == buffer_threshold:
                    orchestration_parameter_service6.append(Buffer_size / buffer_threshold)
                    # slice resource scale-up: there are many incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S6 > buffer_threshold:
                    orchestration_parameter_service6.append(buffer_threshold / Buffer_size)
                    #  slice resource scale-down:there few incoming packets for slice $c$ associated to service $k$
                elif queue_status_parameter_S6 == Buffer_size:
                    orchestration_parameter_service6.append(0)
                    # RB are under utilized: there is no demands for slice
                else:
                    orchestration_parameter_service6.append(1)
                    # RB allocation is well performed: There is no need to update initial RB allocation0)
            reward_service6 = sum(orchestration_parameter_service6) + \
                              penality_fronthaul * (fronthaul_capacity - ((packet_size * 0.001) * vehicle_service6)) + \
                              penality_violating_rb * (RB_service6 - rb_vehicle_service6)
        # Calculate vODU utilization ratio as feedback for loop 1
        for k, v in RAN_vDUs.items():
            if k == 0:
                d1s0 = 0
                d1s1 = 0
                d1s2 = 0
                d1s3 = 0
                d1s4 = 0
                d1s5 = 0
                d1s6 = 0
                t1s0 = 0
                t1s1 = 0
                t1s2 = 0
                t1s3 = 0
                t1s4 = 0
                t1s5 = 0
                t1s6 = 0
                for i in range(len(list(v))):
                    if (i == 0) and (RB_service0 != 0.0):
                        d1s0 = rb_vehicle_service0
                        t1s0 = vehicle_service0
                    if (i == 1) and (RB_service1 != 0.0):
                        d1s1 = rb_vehicle_service1
                        t1s1 = vehicle_service1
                    elif (i == 2) and (RB_service2 != 0.0):
                        d1s2 = rb_vehicle_service2
                        t1s2 = vehicle_service2
                    elif (i == 3) and (RB_service3 != 0.0):
                        d1s3 = rb_vehicle_service3
                        t1s3 = vehicle_service3
                    elif (i == 4) and (RB_service4 != 0.0):
                        d1s4 = rb_vehicle_service4
                        t1s4 = vehicle_service4
                    elif (i == 5) and (RB_service5 != 0.0):
                        d1s5 = rb_vehicle_service5
                        t1s5 = vehicle_service5
                    elif (i == 6) and (RB_service6 != 0.0):
                        d1s6 = rb_vehicle_service6
                        t1s6 = vehicle_service6

                b_d1 = sum(list(v))
                u1 = (d1s6 + d1s5 + d1s4 + d1s3 + d1s2 + d1s1 + d1s0) / b_d1
                Utilization_vODU1.append(u1)
                t1 = int(t1s6 + t1s5 + t1s4 + t1s3 + t1s2 + t1s1 + t1s0 + np.random.uniform(low=1, high=4))
                Vehicle_vdu1.append(t1)
            elif k == 1:
                d2s0 = 0
                d2s1 = 0
                d2s2 = 0
                d2s3 = 0
                d2s4 = 0
                d2s5 = 0
                d2s6 = 0
                t2s0 = 0
                t2s1 = 0
                t2s2 = 0
                t2s3 = 0
                t2s4 = 0
                t2s5 = 0
                t2s6 = 0
                for i in range(len(list(v))):
                    if (i == 0) and (RB_service0 != 0.0):
                        d2s0 = rb_vehicle_service0
                        t2s0 = vehicle_service0
                    elif (i == 1) and (RB_service1 != 0.0):
                        d2s1 = rb_vehicle_service1
                        t2s1 = vehicle_service1
                    elif (i == 2) and (RB_service2 != 0.0):
                        d2s2 = rb_vehicle_service2
                        t2s2 = vehicle_service2
                    elif (i == 3) and (RB_service3 != 0.0):
                        d2s3 = rb_vehicle_service3
                        t2s3 = vehicle_service3
                    elif (i == 4) and (RB_service4 != 0.0):
                        d2s4 = rb_vehicle_service4
                        t2s4 = vehicle_service4
                    elif (i == 5) and (RB_service5 != 0.0):
                        d2s5 = rb_vehicle_service5
                        t2s5 = vehicle_service5
                    elif (i == 6) and (RB_service6 != 0.0):
                        d2s6 = rb_vehicle_service6
                        t2s6 = vehicle_service6
                t2 = t2s6 + t2s5 + t2s4 + t2s3 + t2s2 + t2s1 + t2s0
                Vehicle_vdu2.append(t2)
                b_d2 = sum(list(v))
                u2 = (d2s6 + d2s5 + d2s4 + d2s3 + d2s2 + d2s1 + d2s0) / b_d2
                Utilization_vODU2.append(u2)
            else:
                d3s0 = 0
                d3s1 = 0
                d3s2 = 0
                d3s3 = 0
                d3s4 = 0
                d3s5 = 0
                d3s6 = 0
                t3s0 = 0
                t3s1 = 0
                t3s2 = 0
                t3s3 = 0
                t3s4 = 0
                t3s5 = 0
                t3s6 = 0
                for i in range(len(list(v))):
                    if (i == 0) and (RB_service0 != 0.0):
                        d3s0 = RB_service0
                        t3s0 = vehicle_service0
                    elif (i == 1) and (RB_service1 != 0.0):
                        d3s1 = RB_service1
                        t3s1 = vehicle_service1
                    elif (i == 2) and (RB_service2 != 0.0):
                        d3s2 = RB_service2
                        t3s2 = vehicle_service2
                    elif (i == 3) and (RB_service3 != 0.0):
                        d3s3 = RB_service3
                        t3s3 = vehicle_service3
                    elif (i == 4) and (RB_service4 != 0.0):
                        d3s4 = RB_service4
                        t3s4 = vehicle_service4
                    elif (i == 5) and (RB_service5 != 0.0):
                        d3s5 = RB_service5
                        t3s5 = vehicle_service5
                    elif (i == 6) and (RB_service6 != 0.0):
                        d3s6 = RB_service6
                        t3s6 = vehicle_service6
                t3 = t3s6 + t3s5 + t3s4 + t3s3 + t3s2 + t3s1 + t3s0
                Vehicle_vdu3.append(t3)
                b_d3 = sum(list(v))
                u3 = (d3s6 + d3s5 + d3s4 + d3s3 + d3s2 + d3s1 + d3s0) / (1 + b_d3)
                Utilization_vODU3.append(u3)
        orchestration_parameter = np.array(
            [np.mean(orchestration_parameter_service0), np.mean(orchestration_parameter_service1),
             np.mean(orchestration_parameter_service2), np.mean(orchestration_parameter_service3),
             np.mean(orchestration_parameter_service4), np.mean(orchestration_parameter_service5),
             np.mean(orchestration_parameter_service6)])

    if not Utilization_vODU2:
        Utilization_vODU2.append(0.0)
    if not Utilization_vODU3:
        Utilization_vODU3.append(0.0)
    reward_loop1 = (reward_service0 + reward_service1 + reward_service2 + reward_service3 +
                    reward_service4 + reward_service5 + reward_service6)/len(list(v))

    return reward_loop1, RB_vO_DUs, orchestration_parameter, Utilization_vODU1, Utilization_vODU2, Utilization_vODU3


reward_loop1, RB_vO_DUs, orchestration_parameter, Utilization_vODU1, Utilization_vODU2, Utilization_vODU3 = \
    ClosedLoop1(RAN_vDUs)
max_valueRB = list(RB_vO_DUs)
orchestration_parameter = np.array([np.mean(orchestration_parameter_service0),
                                    np.mean(orchestration_parameter_service1), np.mean(orchestration_parameter_service2),
                                    np.mean(orchestration_parameter_service3), np.mean(orchestration_parameter_service4),
                                    np.mean(orchestration_parameter_service5), np.mean(orchestration_parameter_service6)])

orchestration_parameter = numpy.nan_to_num(orchestration_parameter, copy=True, nan=0.0, posinf=None, neginf=None)
#print("orchestration_parameter", orchestration_parameter)
#print("RAN_vDUs", RAN_vDUs)
#print("RB_vO_DUs", RB_vO_DUs)
#print("max_valueRB", np.float32(np.array(max_valueRB)))

#print("reward_loop1", reward_loop1)
utilization_vODU = np.array(Utilization_vODU1) + np.array(Utilization_vODU2) + np.array(Utilization_vODU3)
services_values = list(services.values())
Unallocation_RBs = InP_max_RBs - max_valueRB
services_values = np.array(services_values)
services_values = services_values
penalty_slice = 0.00006
penalty_reward1 = 0.000092


class ClosedLoopSlicing(Env):
    def __init__(self):
        # Actions we can take:  scale-up, scale-down, terminate, keep initial allocation
        self.action_space = Discrete(4)
        self.low_value = np.asarray(np.array([0.0, 0.0, 0.0])).astype(np.float32)
        self.max_value = np.asarray(RB_vO_DUs_copy).astype(np.float32)
        self.observation_space = Box(low=self.low_value, high=self.max_value, shape=(3,), dtype=np.float32)
        # Slice Resources
        self.state = np.array(RB_vO_DUs, dtype=float)
        self.Total_RBs = InP_max_RBs
        self.AllService = services_values
        self.RAN_vDUs = RAN_vDUs
        self.RB_vO_DUs = RB_vO_DUs
        self.Unallocation_RBs = Unallocation_RBs
        self.orchestration_parameter = orchestration_parameter
        self.Utilization_vODU1 = Utilization_vODU1
        self.Utilization_vODU2 = Utilization_vODU2
        self.Utilization_vODU3 = Utilization_vODU3
        self.reward_loop1 = reward_loop1
        self.timeframe = 200
        self.calling = 0

    def step(self, action):
        size_value = 0
        k = 0
        # Actions we can take:  0: scale-up 1.2, 1:scale-down 0.6, 2:terminate 0, 3: keep initial allocation 2
        refer_action = (2 - action) * 0.6
        if refer_action >= 0:
            refer_action = refer_action  # 1: scale-up, 2:scale-down, 3:terminate
        else:
            refer_action = 2  # 2: keep initial allocation, more preferable
        #print("action", action)
        #print("refer_action",  refer_action)
        AllService = np.floor(self.AllService * refer_action)
        RAN_vDUs_update = self.RAN_vDUs
        #print("self.AllService ", AllService)
        for i in range(len(self.RB_vO_DUs)):
            if i != 2:
                values_of_key = self.RAN_vDUs[i]
                size_value = size_value + len(values_of_key)
                RAN_vDUs_update[i] = list(AllService[k:size_value])
                k = i + size_value
            else:
                RAN_vDUs_update[2] = [AllService[-1]]

        vODU1_sum = sum(RAN_vDUs_update[0])
        vODU2_sum = sum(RAN_vDUs_update[1])
        vODU3_sum = sum(RAN_vDUs_update[2])
        self.RAN_vDUs = RAN_vDUs_update
        #print("self.RAN_vDUs", self.RAN_vDUs)
        reward_loop1, RB_vO_DUs, orchestration_parameter, Utilization_vODU1, Utilization_vODU2, \
        Utilization_vODU3 = ClosedLoop1(self.RAN_vDUs)
        global count_call
        self.calling = self.calling + 1
        loop1_calling.append(self.calling)
        global Reward_loo11
        Reward_loo11.append(self.reward_loop1)
        self.state = np.array(RB_vO_DUs, dtype=float)
        self.state = np.array([vODU1_sum, vODU2_sum, vODU3_sum])
        self.orchestration_parameter = orchestration_parameter
        self.Utilization_vODU1 = Utilization_vODU1
        self.Utilization_vODU2 = Utilization_vODU2
        self.Utilization_vODU3 = Utilization_vODU3
        self.reward_loop1 = reward_loop1
        #print("self.state ", self.state)
        #print("self.reward_loop1", self.reward_loop1)
        self.Unallocation_RBs = InP_max_RBs - sum(self.AllService)
        if sum(self.state) <= self.Total_RBs:
            reward2 = refer_action * (np.mean(np.nan_to_num(self.Utilization_vODU1)) +
                                      np.mean(np.nan_to_num(self.Utilization_vODU2)) + np.mean(self.Utilization_vODU3)
                                      + (penalty_slice * (InP_max_RBs - sum(self.AllService))))

            # We penalized reward_loop1 to avoid the closed loop 2 to be too influenced by closed loop 1
            reward = reward2 + penalty_reward1 * (self.reward_loop1/(self.timeframe * len(self.AllService)))
        else:
            # We penalized reward_loop1 to avoid the closed loop 2 to be too influenced by closed loop 1
            reward2 = 0
            reward = reward2 + penalty_reward1 * (self.reward_loop1/(self.timeframe * len(self.AllService)))
        #print("reward", reward)
        #print("tewsyy", np.mean(np.nan_to_num(self.Utilization_vODU1)), np.mean(np.nan_to_num(self.Utilization_vODU2)),
        #      np.mean(self.Utilization_vODU3), (penalty_slice * (InP_max_RBs - sum(self.AllService))))
        #print("reward2", reward2)
        self.timeframe -= 1
        # Break the loop for analysis
        #if self.Unallocation_RBs <= 0:
        if self.timeframe <= 0:
            done = True
        else:
            done = False
        # Return step information
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df1["Loop1Call"] = loop1_calling
        df2["RewardLoop1"] = Reward_loo11

        df1.to_csv('dataset/LoopOneCall.csv')
        df2.to_csv('dataset/RewardsonCall.csv')
        return self.state, reward, done, {}

    def render(self):
        pass

    def reset(self):
        self.state = np.array(RB_vO_DUs, dtype=float)
        self.Total_RBs = InP_max_RBs
        self.AllService = services_values
        self.Unallocation_RBs = Unallocation_RBs
        self.orchestration_parameter = orchestration_parameter
        self.Utilization_vODU1 = Utilization_vODU1
        self.Utilization_vODU2 = Utilization_vODU2
        self.Utilization_vODU3 = Utilization_vODU3
        self.reward_loop1 = reward_loop1
        self.timeframe = 200
        return self.state


x_axis = list(range(0, n_iteration))
x_axis = np.array(x_axis)
xnew11 = np.linspace(x_axis.min(), x_axis.max(), n_iteration)
print("all_vehicle_service0", all_vehicle_service0)
a_BSpline10 = make_interp_spline(x_axis, all_vehicle_service0, k=3)  # type: BSpline
y_all_vehicle_service0 = a_BSpline10(xnew11)
a_BSpline11 = make_interp_spline(x_axis, all_vehicle_service1, k=3)  # type: BSpline
y_all_vehicle_service1 = a_BSpline11(xnew11)

a_BSpline22 = make_interp_spline(x_axis, all_vehicle_service2, k=3)  # type: BSpline
y_all_vehicle_service2 = a_BSpline22(xnew11)

a_BSpline33 = make_interp_spline(x_axis, all_vehicle_service3, k=3)  # type: BSpline
y_all_vehicle_service3 = a_BSpline33(xnew11)

a_BSpline44 = make_interp_spline(x_axis, all_vehicle_service4, k=3)  # type: BSpline
y_all_vehicle_service4 = a_BSpline44(xnew11)

a_BSpline55 = make_interp_spline(x_axis, all_vehicle_service5, k=3)  # type: BSpline
y_all_vehicle_service5 = a_BSpline55(xnew11)

a_BSpline66= make_interp_spline(x_axis, all_vehicle_service6, k=3)  # type: BSpline
y_all_vehicle_service6 = a_BSpline66(xnew11)


df = pd.DataFrame({'Slice 0': all_vehicle_service0, 'Slice 1': all_vehicle_service1,
                   'Slice 2': all_vehicle_service2,
                   'Slice 3': all_vehicle_service3, 'Slice 4': all_vehicle_service4,
                   'Slice 5': all_vehicle_service5, 'Slice 6': all_vehicle_service6})
boxplot = df.plot.area()
boxplot.plot()
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Number of vehicles', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.show()

df = pd.DataFrame({'Slice 0': all_vehicle_service0, 'Slice 1': all_vehicle_service1,
                   'Slice 2': all_vehicle_service2,
                   'Slice 3': all_vehicle_service3, 'Slice 4': all_vehicle_service4,
                   'Slice 5': all_vehicle_service5, 'Slice 6': all_vehicle_service6})

# creating a dictionary with one specific color per group:

my_pal = {"Slice 0": "teal", "Slice 1": "red", "Slice 2": "green", "Slice 3": "blue", "Slice 4": "cyan",
          "Slice 5": "Yellow", "Slice 6": "pink"}
sns.violinplot(data=df, linewidth=2, cut=0, palette=my_pal)
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Number of cars', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.show()






plt.plot(y_all_vehicle_service0, label="Slice 0", )
plt.plot(y_all_vehicle_service1, label="Slice 1", )
plt.plot(y_all_vehicle_service2, label="Slice 2", )
plt.plot(y_all_vehicle_service3, label="Slice 3", )
plt.plot(y_all_vehicle_service4, label="Slice 4", )
plt.plot(y_all_vehicle_service5, label="Slice 5",)
plt.plot(y_all_vehicle_service6, label="Slice 6",)
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Number of cars', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.legend(fontsize=18)
plt.show()
print("(equirement_satisfaction_parameter_s0_array", requirement_satisfaction_parameter_s2_array)
plt.plot(requirement_satisfaction_parameter_s0_array, marker=1, label="Slice 0", )
plt.plot(requirement_satisfaction_parameter_s1_array, marker=2, label="Slice 1", )
plt.plot(requirement_satisfaction_parameter_s2_array, marker=3, label="Slice 2", )
plt.plot(requirement_satisfaction_parameter_s3_array, marker=4, label="Slice 3", )
plt.plot(requirement_satisfaction_parameter_s4_array, marker=5, label="Slice 4", )
plt.plot(requirement_satisfaction_parameter_s5_array, marker=6, label="Slice 5",)
plt.plot(requirement_satisfaction_parameter_s6_array, marker=7, label="Slice 6",)
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Slice requirement satisfaction', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.legend(fontsize=18)
plt.show()


print("Vehicle_vdu1",Vehicle_vdu1)
print("Vehicle_vdu2",Vehicle_vdu2)
print("Vehicle_vdu3",Vehicle_vdu3)
a_BSpline1 = interpolate.make_interp_spline(x_axis, Vehicle_vdu1)
y_Vehicle_vdu1 = a_BSpline1(xnew11)
a_BSpline2 = interpolate.make_interp_spline(x_axis, Vehicle_vdu2)
y_Vehicle_vdu2 = a_BSpline2(xnew11)

a_BSpline3 = interpolate.make_interp_spline(x_axis, Vehicle_vdu3)
y_Vehicle_vdu3 = a_BSpline3(xnew11)

print("RAN_vDUs.get(0)", RAN_vDUs.get(0), )
print("RAN_vDUs.get(1)", RAN_vDUs.get(1), )
print("RAN_vDUs.get(3)", RAN_vDUs.get(2), )
plt.plot(y_Vehicle_vdu1, label="vO-DU 1" + str(RAN_vDUs.get(0)))
plt.plot(y_Vehicle_vdu2, label="vO-DU 2" + str(RAN_vDUs.get(1)))
plt.plot(y_Vehicle_vdu3, label="vO-DU 3" + str(RAN_vDUs.get(2)))
plt.grid(color='gray', linestyle='dashed')
plt.legend(fontsize=18)
plt.ylabel('Number of cars', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.show()

# Get a color map
my_cmap = cm.get_cmap('jet')

# Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
keys_service = services.keys()
values_service = services.values()
plt.bar(keys_service, values_service, color=['teal', 'red', 'green', 'blue', 'cyan', 'Yellow', 'pink', 'Olive', 'chocolate'])

plt.plot(keys_service, values_service, marker='o', linewidth=3)
for a,b in zip(keys_service, values_service):
    plt.text(a, b, str(b), fontsize=18)
plt.ylabel('RBs Allocation', fontsize=18)
plt.xlabel('Services', fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.show()

#print("Utilization_vODU1", Utilization_vODU1)
#print("Utilization_vODU2", Utilization_vODU2)
#print("Utilization_vODU3", Utilization_vODU3)

plt.plot(Utilization_vODU1, label="vODU1", )
plt.plot(Utilization_vODU2, label="vODU2", )
plt.plot(Utilization_vODU3, label="vODU3", )
plt.axhline(y=1, color="red", linewidth=3)
plt.text(0, 1, 'Maximum RBs usage ratio at vODUs (100%)', fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('RBs usage ratio at vODUs', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.legend(fontsize=18, loc='lower right')
plt.show()


#print(df1)
#print(df2)

