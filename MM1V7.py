"""
Created on Thu May 14 21:01:25 2020
@author: gauravsharma
# Event-Based M/M/1 Queue Simulation in Python
https://medium.com/@gauravsharma_61093/event-based-m-m-1-queue-simulation-in-python-8405acb2154e
"""

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.DataFrame()


class Simulation:
    def __init__(self, NUM_vehicles):
        self.num_in_system = 0
        self.clock = 0.0
        self.t_arrival = self.generate_interarrival()
        self.t_depart = float('inf')
        self.num_arrivals = 0
        self.num_departs = 0
        self.total_wait = 0.0
        self.tmp_time = self.generate_service()
        self.vehicles = NUM_vehicles

    def advance_time(self):
        t_event = min(self.t_arrival, self.t_depart)
        self.total_wait += self.num_in_system * (t_event - self.clock)
        self.clock = t_event
        if self.t_arrival <= self.t_depart:
            self.handle_arrival_event()
        else:
            self.handle_depart_event()

    def handle_arrival_event(self):
        self.num_in_system = self.vehicles
        self.num_arrivals = self.vehicles
        if self.num_in_system <= self.vehicles:
            self.temp1 = self.generate_service()
            self.tmp_time = self.temp1
            self.t_depart = self.clock + self.temp1
        self.t_arrival = self.clock + self.generate_interarrival()

    def handle_depart_event(self):
        self.num_in_system -= 1
        self.num_departs += 1
        if self.num_in_system > 0:
            self.temp2 = self.generate_service()
            self.tmp_time = self.temp2
            self.t_depart = self.clock + self.temp2
        else:
            self.t_depart = float("inf")

    def generate_interarrival(self):
        return np.random.lognormal(mean=0.6, sigma=0.3)

    def generate_service(self):
        self.vehicles_service_time = np.random.uniform(low=5, high=8)
        return self.vehicles_service_time
