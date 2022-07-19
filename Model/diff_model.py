import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import cmath
import seaborn as sns
import pandas as pd
import time

# Model with 3 spont terms, and the interaction terms depended on the spont terms

def model_all_combo(y,t,M,b2,g2,d2,o2,s1,s2,s3):
    ### CAN ONLY BE USED FOR CONSTANT POPULATION MODELS
    N,S,Q = y    #  Never smoker, smoker, quitter

    dNdt = (N*S*(-1 + (1 + b2*(-1 + (1 - s1)**4.5))**(0.2222222222222222**S)))/M - N*s1
    dSdt = -(((1 - (1 - d2)**N)*N*S)/M) + ((1 - (1 - o2)**S)*Q*S)/M + (N*S*(1 - (1 + b2*(-1 + (1 - s1)**4.5))**(0.2222222222222222*S)))/M - (Q*S*(1 - (1 + g2*(-1 + (1 - s1)**4.5))**(0.2222222222222222*Q)))/M + N*s1 - S*s2 + Q*s3
    dQdt = ((1 - (1 - d2)**N)*N*S)/M - ((1 - (1 - o2)**S)*Q*S)/M + (Q*S*(1 - (1 + g2*(-1 + (1 - s1)**4.5))**(0.2222222222222222*Q)))/M + S*s2 - Q*s3
    x= dNdt + dSdt +dQdt 
    
    return dNdt,dSdt,dQdt

def solver2( t1,y0,M, b,g,d,o,s1,s2,s3):
   # Plots the timeseries of the model
# Initial conditions vector
    t = np.linspace(0, t1, t1+1)   
# Integrate the SIR equations over the time grid, t.
    ret = odeint(model_all_combo, y0, t,hmax=0.5, args=(M,b,g,d,o,s1,s2,s3))
    N,S,Q = ret.T

    return N,S,Q



def trial(y,t,M,b2,g2,d2,o2,s1,s2,s3):
    S,Q = y 
    b1= 1 - (1 + b2*(-1 + (1 - s1)**4.5))**0.2222222222222222
    g1= 1 - (1 + g2*(-1 + (1 - s1)**4.5))**0.2222222222222222
    dSdt= (S*(1 - (1 - b1)**S)*(M - Q - S))/M - (S*(M - Q - S)*(1 - (1 - d2)**(M - Q - S)))/M + (Q*S*(1 - (1 - g1)**Q))/M + (Q*S*(1 - (1 - o2)**S))/M + s1*(M - Q - S) + Q*s3 + S*(-s2)
    dQdt= ((1 - (1 - g1)**Q)*Q*S)/M - ((1 - (1 - o2)**S)*Q*S)/M + ((1 - (1 - d2)**(M - Q - S))*(M - Q - S)*S)/M + S*s2 - Q*s3
    
    return dSdt,dQdt

def solver( t1,y0,M, b,g,d,o,s1,s2,s3):
   # Plots the timeseries of the model
# Initial conditions vector
    t = np.linspace(0, t1, t1+1)   
# Integrate the SIR equations over the time grid, t.
    #ret = odeint(model_all_combo, y0, t,hmax=0.5, args=(M,b,g,d,o,s1,s2,s3))
    ret = odeint(trial, y0, t,hmax=0.5, args=(M,b,g,d,o,s1,s2,s3))
    S,Q = ret.T
    
    return S,Q




