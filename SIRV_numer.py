# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:54:35 2023

@author: pablo
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIRV(y,t,beta,gamm,tv):
    n=np.sum(y)
    dsdt=(-(beta)*(y[0]*y[1])/n)-tv*y[0]
    didt=(beta*(y[0]*y[1])/n)-gamm*y[1]
    drdt=gamm*y[1]
    dvdt=y[0]*tv
    return dsdt, didt, drdt, dvdt

s0=999
i0=1
r0=0
v0=0
beta=1/2
gamma=1/8
tv=0.05
y0=np.array([s0,i0,r0,v0])
T=100
t=np.arange(T)
sol= odeint(SIRV,y0,t,args=(beta, gamma, tv))
S,I,R,V=sol.T
# gráfica
fig,axs=plt.subplots()
axs.plot(np.arange(T),V,label="Vacunados")
axs.plot(np.arange(T),S,label="Susceptibles")
axs.plot(np.arange(T),I,label="Infectados")
axs.plot(np.arange(T),R,label="Recuperados")
axs.legend()
plt.show    
