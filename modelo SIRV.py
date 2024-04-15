# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:28:02 2023

@author: pablo
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def ceros(n,m):
    df=(np.zeros([n, m]))
    return df
#%% función de la infección completamente probabilística
def cur_inf(T,N,i0,C,B,v=0):
    c=1/C # probabilidad de contagio al entrar en contacto con un infectado
    r=1/B # probabilidad de recuperarse ya que se está infectado
    S=ceros(T,N)
    V=ceros(T,N)
    R=ceros(T,N)
    I=ceros(T,N)
    estad=ceros(T,N)
    # en t=0
    v0=0 # vacunados en t=0
    r0=0 # recuperados en t=0
    s0=N-i0 # susceptibles en t=0
    ii0=np.random.choice(np.arange(N),size=i0)
    I[0,ii0]=1
    ini=np.array(list(set(np.arange(N)).difference(set(ii0))))
    S[0,ini]=1
    # equivalencia en matriz estado
    # 271 (susceptible) 272 (infectado) 273 (recuperado) 274 (vacunado)
    estad[0,ini]=271
    estad[0,ii0]=272
    for t in range(1,T):
        V[t,:]=V[t-1,:]
        S[t,:]=S[t-1,:]
        I[t,:]=I[t-1,:]
        R[t,:]=R[t-1,:]
        estad[t,:]=estad[t-1,:]
        p_v=np.where(S[t,:]==1)[0]
        ipv=np.isin(np.arange(N),p_v)
        if p_v.size!=0:
            pv=np.random.rand(N)
            ev=pv<v
            vact=ipv & ev
            S[t,vact]=0
            V[t,vact]=1
            estad[t,vact]=274
        if t>1:
            p_r=np.where(I[t,:]==1)[0]
            ipr=np.isin(np.arange(N),p_r)
            if p_r.size!=0:
                pr=np.random.rand(N)
                er=(pr<=r)
                R[t,(ipr) & (er)]=1
                estad[t,(ipr) & (er)]=273
                I[t,(ipr) & (er)]=0
        for i in range(0,N):
            est1=estad[t,i]
            j=np.random.choice(np.array(list(set(np.arange(N)).difference(set({i})))),size=1)[0]
            est2=estad[t,j]
            if np.sum([est1==272,est2==272])==1:
                if est1==271:
                    a=np.random.rand()
                    if a<c:
                        S[t,i]=0
                        I[t,i]=1
                        estad[t,i]=272 
                if est2==271:
                    b=np.random.rand()
                    if b<c:
                        S[t,j]=0
                        I[t,j]=1
                        estad[t,j]=272 
    st=np.sum(S,axis=1)
    it=np.sum(I,axis=1)
    rt=np.sum(R,axis=1)
    vt=np.sum(V,axis=1)
    return st,it,rt,vt            
#%% función de la infección con infección probabilística y recuperación determinística
def cur_infpd(T,N,i0,C,B,v=0):
    c=1/C # probabilidad de contagio al entrar en contacto con un infectado
    r=1/B # probabilidad de recuperarse ya que se está infectado
    S=ceros(T,N)
    V=ceros(T,N)
    R=ceros(T,N)
    I=ceros(T,N)
    estad=ceros(T,N)
    # en t=0
    v0=0 # vacunados en t=0
    r0=0 # recuperados en t=0
    s0=N-i0 # susceptibles en t=0
    ii0=np.random.choice(np.arange(N),size=i0)
    I[0,ii0]=1
    ini=np.array(list(set(np.arange(N)).difference(set(ii0))))
    S[0,ini]=1
    # equivalencia en matriz estado
    # 271 (susceptible) 272 (infectado) 273 (recuperado) 274 (vacunado)
    estad[0,ini]=271
    estad[0,ii0]=272
    for t in range(1,T):
        V[t,:]=V[t-1,:]
        S[t,:]=S[t-1,:]
        I[t,:]=I[t-1,:]
        R[t,:]=R[t-1,:]
        estad[t,:]=estad[t-1,:]
        p_v=np.where(S[t,:]==1)[0]
        ipv=np.isin(np.arange(N),p_v)
        if p_v.size!=0:
            pv=np.random.rand(N)
            ev=pv<v
            vact=ipv & ev
            S[t,vact]=0
            V[t,vact]=1
            estad[t,vact]=274
        for k in range(0,N):
            if estad[t,k]==272:
                tr=np.where(I[:,k]==1)[0][0]
                if (t-tr)==B:
                    I[t,k]=0
                    R[t,k]=1
                    estad[t,k]=273
        for i in range(0,N):
            est1=estad[t,i]
            j=np.random.choice(np.array(list(set(np.arange(N)).difference(set({i})))),size=1)[0]
            est2=estad[t,j]
            if np.sum([est1==272,est2==272])==1:
                if est1==271:
                    a=np.random.rand()
                    if a<c:
                        S[t,i]=0
                        I[t,i]=1
                        estad[t,i]=272 
                if est2==271:
                    b=np.random.rand()
                    if b<c:
                        S[t,j]=0
                        I[t,j]=1
                        estad[t,j]=272 
    st=np.sum(S,axis=1)
    it=np.sum(I,axis=1)
    rt=np.sum(R,axis=1)
    vt=np.sum(V,axis=1)
    return st,it,rt,vt           
#%% función en forma no funcional. Totalmente probabilística
# definir agentes y parámetros
v=0.05# vacunación diaria
C=4
B=8
c=1/C # probabilidad de contagio al entrar en contacto con un infectado
r=1/B # probabilidad de recuperarse ya que se está infectado
N=1000 # número de ciudadanos
T=100 # periodos a simular
S=ceros(T,N)
V=ceros(T,N)
R=ceros(T,N)
I=ceros(T,N)
estad=ceros(T,N)
# en t=0
i0= 1 # número de pacientes 0
v0=0 # vacunados en t=0
r0=0 # recuperados en t=0
s0=N-i0 # susceptibles en t=0
ii0=np.random.choice(np.arange(N),size=i0)
I[0,ii0]=1
ini=np.array(list(set(np.arange(N)).difference(set(ii0))))
S[0,ini]=1
# equivalencia en matriz estado
# 271 (susceptible) 272 (infectado) 273 (recuperado) 274 (vacunado)
estad[0,ini]=271
estad[0,ii0]=272
for t in range(1,T):
    V[t,:]=V[t-1,:]
    S[t,:]=S[t-1,:]
    I[t,:]=I[t-1,:]
    R[t,:]=R[t-1,:]
    estad[t,:]=estad[t-1,:]
    p_v=np.where(S[t,:]==1)[0]
    ipv=np.isin(np.arange(N),p_v)
    if p_v.size!=0:
        pv=np.random.rand(N)
        ev=pv<v
        vact=ipv & ev
        S[t,vact]=0
        V[t,vact]=1
        estad[t,vact]=274
    if t>1:
        p_r=np.where(I[t,:]==1)[0]
        ipr=np.isin(np.arange(N),p_r)
        if p_r.size!=0:
            pr=np.random.rand(N)
            er=(pr<=r)
            R[t,(ipr) & (er)]=1
            estad[t,(ipr) & (er)]=273
            I[t,(ipr) & (er)]=0
    for i in range(0,N):
        est1=estad[t,i]
        j=np.random.choice(np.array(list(set(np.arange(N)).difference(set({i})))),size=1)[0]
        est2=estad[t,j]
        if np.sum([est1==272,est2==272])==1:
            if est1==271:
                a=np.random.rand()
                if a<c:
                    S[t,i]=0
                    I[t,i]=1
                    estad[t,i]=272 
            if est2==271:
                b=np.random.rand()
                if b<c:
                    S[t,j]=0
                    I[t,j]=1
                    estad[t,j]=272 
# gráficas       
fig,axs=plt.subplots()
axs.plot(np.arange(T),np.sum(V[:,:],axis=1),label="Vacunados")
axs.plot(np.arange(T),np.sum(S[:,:],axis=1),label="Susceptibles")
axs.plot(np.arange(T),np.sum(I[:,:],axis=1),label="Infectados")
axs.plot(np.arange(T),np.sum(R[:,:],axis=1),label="Recuperados")
axs.legend()
plt.show       
#%% contagio probabilístico y recuperación estocástica
v=0.05# vacunación diaria
C=4
B=8
c=1/C # probabilidad de contagio al entrar en contacto con un infectado
r=1/B # probabilidad de recuperarse ya que se está infectado
N=1000 # número de ciudadanos
T=100 # periodos a simular
S=ceros(T,N)
V=ceros(T,N)
R=ceros(T,N)
I=ceros(T,N)
estad=ceros(T,N)
# en t=0
i0= 1 # número de pacientes 0
v0=0 # vacunados en t=0
r0=0 # recuperados en t=0
s0=N-i0 # susceptibles en t=0
ii0=np.random.choice(np.arange(N),size=i0)
I[0,ii0]=1
ini=np.array(list(set(np.arange(N)).difference(set(ii0))))
S[0,ini]=1
# equivalencia en matriz estado
# 271 (susceptible) 272 (infectado) 273 (recuperado) 274 (vacunado)
estad[0,ini]=271
estad[0,ii0]=272
for t in range(1,T):
    V[t,:]=V[t-1,:]
    S[t,:]=S[t-1,:]
    I[t,:]=I[t-1,:]
    R[t,:]=R[t-1,:]
    estad[t,:]=estad[t-1,:]
    p_v=np.where(S[t,:]==1)[0]
    ipv=np.isin(np.arange(N),p_v)
    if p_v.size!=0:
        pv=np.random.rand(N)
        ev=pv<v
        vact=ipv & ev
        S[t,vact]=0
        V[t,vact]=1
        estad[t,vact]=274
    for k in range(0,N):
        if estad[t,k]==272:
            tr=np.where(I[:,k]==1)[0][0]
            if (t-tr)==B:
                I[t,k]=0
                R[t,k]=1
                estad[t,k]=273
    for i in range(0,N):
        est1=estad[t,i]
        j=np.random.choice(np.array(list(set(np.arange(N)).difference(set({i})))),size=1)[0]
        est2=estad[t,j]
        if np.sum([est1==272,est2==272])==1:
            if est1==271:
                a=np.random.rand()
                if a<c:
                    S[t,i]=0
                    I[t,i]=1
                    estad[t,i]=272 
            if est2==271:
                b=np.random.rand()
                if b<c:
                    S[t,j]=0
                    I[t,j]=1
                    estad[t,j]=272 
# gráficas       
fig,axs=plt.subplots()
axs.plot(np.arange(T),np.sum(V[:,:],axis=1),label="Vacunados")
axs.plot(np.arange(T),np.sum(S[:,:],axis=1),label="Susceptibles")
axs.plot(np.arange(T),np.sum(I[:,:],axis=1),label="Infectados")
axs.plot(np.arange(T),np.sum(R[:,:],axis=1),label="Recuperados")
axs.legend()
plt.show           
#%% curva suavizada
T=100
ns=127 # número de simulaciones
st=ceros(T,1)
it=ceros(T,1)
rt=ceros(T,1)
vt=ceros(T,1)

for q in tqdm(range(0,ns)):
    a=cur_inf(T,1000,1,4,8,0.05)
    st[:,0]+=a[0]
    it[:,0]+=a[1]
    rt[:,0]+=a[2]
    vt[:,0]+=a[3]
st=st/ns
it=it/ns
rt=rt/ns
vt=vt/ns
#%% gráfica
fig,axs=plt.subplots()
axs.plot(np.arange(T),vt,label="Vacunados")
axs.plot(np.arange(T),st,label="Susceptibles")
axs.plot(np.arange(T),it,label="Infectados")
axs.plot(np.arange(T),rt,label="Recuperados")
axs.legend()
plt.show          
   
    
    
    