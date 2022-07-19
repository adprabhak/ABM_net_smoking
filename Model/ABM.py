# This version of the smoking model assumes that there are three spontaneous terms depicting smoking initiation, quitting and relapse. 
# and 4 terms for interactions
# The interaction parameters are directly connected to the spontaneous terms, and we calculate it using the christakis paper results.
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

from mesa.time import RandomActivation
from mesa.time import SimultaneousActivation
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa import Agent, Model
from enum import Enum
from typing import Any, Optional
class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    QUITTER = 2
def agentlist(model,state):
    lis =[]
    for a in model.grid.get_all_cell_contents():
        if a.state is state:
            lis.append(int(a.unique_id))
            #lis1.append(a) 
    return lis
def total_pop(model):
    return sum([1 for a in model.grid.get_all_cell_contents()])
def return_agents_S(model):
    return model.agentlist_S
def return_agents_Q(model):
    return model.agentlist_Q
def return_agents_I(model):
    return model.agentlist_I
class Imodel(Model):
    def __init__(self, i0 = 0.3, q0= 0.2, b=0.4, g= 0.3, d =0.02,o=0.1, s1= 0.01, s2 = 0.02,s3=0.01,net = nx.complete_graph(10),filename = 0,directory ='/exports/eddie/scratch/s2006399/data/',index = 0,style=0):
        self.G = net   # The network
        self.grid = NetworkGrid(self.G)
        self.i0 =i0       # Iniitial population of infected
        self.q0 = q0      # Initital population of quitters
        self.b = b        # Interaction parameter between S and I
        self.g = g        # Interaction parameter between I and Q
        self.d = d        # Interaction parameter between S and I leading to quitters
        self.o = o        # Interaction parameter between I and Q leading to relapse

        self.s1 = s1       #Spontaneous parameter for picking up smoking  *positive value*
        self.s2 = s2       #Spontaneous parameter for quitting *positive value*
        self.s3 = s3       # Spontaneous parameter for relapse *positive value*
        self.filename =filename    # for iteration - default function in MESA giving problems with parallel runs
        self.direct = directory

           
        self.eff_b= 1-pow(self.b* (pow(1-self.s1,4.5) - 1) +1,(1/4.5)) # Per year rate of smoking initiation per S-I edge from Christakis paper
        self.eff_g= 1-pow(self.g* (pow(1-self.s2,4.5) - 1) +1,(1/4.5))
        
        self.style= style
        self.index=index
        if self.style!=0:
            nx.write_gpickle(self.G, self.direct + '/Network_'+ str(self.index)+'.bz2')    
        self.agentlist_S =list()
        self.agentlist_I = list()
        self.agentlist_Q = list()

        self.schedule = SimultaneousActivation(self)
        # Create agents
        for i, node1 in enumerate(self.grid.G.nodes()):
            a = subjects(i, self, State.SUSCEPTIBLE)
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node1)
        n_nodes = len(self.grid.G.nodes())
        # Infect some nodes
        initial_inf = self.random.sample(self.grid.G.nodes(), round(self.i0 * n_nodes))
        for a in self.grid.get_cell_list_contents(initial_inf):
            a.state = State.INFECTED
   
        # Randomly select quitters from the noninfected pop
        quitlis=  list(set(self.grid.G.nodes())^set(initial_inf))     # Taking the symmetric difference : all elements in either list1 or list2 but not both
        initial_quitter = self.random.sample(quitlis, round(self.q0 * n_nodes))
        for a in self.grid.get_cell_list_contents(initial_quitter):
            a.state = State.QUITTER
       
        self.running = True

        self.datacollector = DataCollector({"agents_I": return_agents_I,
                                            "agents_S": return_agents_S,
                                            "agents_Q": return_agents_Q,
                                           })  
        
    def step(self):
        self.current_pop = total_pop(self)
        if (self.current_pop == 0):
            self.running = False
        
        if self.current_pop!=0:
            self.agentlist_I =(agentlist(self,State.INFECTED))
            self.agentlist_Q =(agentlist(self,State.QUITTER))
            self.agentlist_S = (agentlist(self,State.SUSCEPTIBLE))

        self.datacollector.collect(self)
        self.schedule.step() 

    def run_model(self, k):
        for i in range(k):
            self.step()
class subjects(Agent):
    def __init__(self, unique_id, model, initial_state):
        super().__init__(unique_id, model)
        self.state = initial_state
        self.newstate = initial_state
       
    def advance(self):
        self.state = self.newstate
        
    def susc_actions(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        infected_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 (agent.state is State.INFECTED)]

        if len(neighbors_nodes) != 0: 
       	    prob = (1 - pow(1-self.model.eff_b ,len(infected_neighbors))) * (len(infected_neighbors)/ len(neighbors_nodes))
            if (random.random() < prob):            
        	    self.newstate = State.INFECTED
       
        if (random.random() < self.model.s1):
        	self.newstate = State.INFECTED
        del neighbors_nodes, infected_neighbors
                
    def inf_actions(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        
        susceptible_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 (agent.state is State.SUSCEPTIBLE)]
        
        quit_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 (agent.state is State.QUITTER)]
        
        if len(neighbors_nodes) !=0: 
            prob1 = (1- pow(1-self.model.eff_g ,len(quit_neighbors)))  * (len(quit_neighbors)  / len(neighbors_nodes))
            if random.random() < prob1:
                self.newstate = State.QUITTER
            
            prob2 = (1- pow(1-self.model.d ,len(susceptible_neighbors)) ) * (len(susceptible_neighbors) / len(neighbors_nodes))
            if random.random() <prob2:
                self.newstate = State.QUITTER
                        
        if random.random() < self.model.s2:
            self.newstate = State.QUITTER
        
        del neighbors_nodes, susceptible_neighbors, quit_neighbors
           
    def quit_actions(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        infected_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 (agent.state is State.INFECTED)]

        if len(neighbors_nodes) != 0: 
       	    prob = (1 - pow(1-self.model.o ,len(infected_neighbors))) * (len(infected_neighbors)/ len(neighbors_nodes))
            if (random.random() < prob):            
        	    self.newstate = State.INFECTED

        if random.random() < abs(self.model.s3):
        	self.newstate = State.INFECTED
        
    def step(self):
        self.newstate = self.state       
        if self.state is State.SUSCEPTIBLE:
            self.susc_actions()
       
        if self.state is State.INFECTED:
            self.inf_actions()
    
        if self.state is State.QUITTER:
           self.quit_actions()   
        
