# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:21:35 2019

@author: I345144
"""

import numpy as np
from scipy import special as sp

def qfunc(arg):
    return 0.5-0.5*sp.erf(arg/1.414)
def qfuncinv(arg):
    return 1.414*sp.erfinv(1-2*arg)

P=10;
A=1;
N=10;
EX=1000;

#V=np.linspace(1,P,P);


### equation for influence field
h = 1
xc = 5
yc = 5
varx = 1  
vary = 1
k=10
Xi = np.random.normal(xc, varx, k)
Yi = np.random.normal(yc, vary, k)
xi, yi = np.meshgrid(Xi, Yi)
xi = np.reshape(xi, (1, k*k))
yi = np.reshape(yi, (1, k*k))

 # bi is obtained in two different ways. Pick the one you prefer. Second
# approach takes a lot of computation time as the size is 10 times bigger 
bi = -0.5*(((Xi - xc)**2/varx) + ((Yi - yc)**2/vary))

#P=100
#N=100
#ci = -0.5*(((xi - xc)**2/varx) + ((yi - yc)**2/vary))
#bi=ci.ravel()

gi = h*np.exp(bi)


V=gi


# probability of false alarm
PFA=0.3;

## gamma value for threshold
Gamth=np.sqrt(V/N)*qfuncinv(PFA);


## probability of no detection-- H0 hypothesis
PDNP=np.zeros(P);

# probability of detection -- H1 hypothesis
PDB=np.zeros(P);


for vind in range(P):
    ## count if given observation belongs to H0 hypothesis
    CountD=0;
    ## count if given observation belongs to H1 hypothesis
    CountB=0;
    for i in range(EX):
        x=A+np.random.normal(0, np.sqrt(V[vind]),N);
        if (np.mean(x)>Gamth[vind]):
            CountD=CountD+1;
        if(np.mean(x)>(A/2)):
            CountB=CountB+1;
        
    PDNP[vind]=CountD/EX;
    PDB[vind]=CountB/EX;
    print("Iteration", vind)
    print("PDNP",PDNP)
    print("PDB",PDB)
    
    ## Liklihood ratio 
    LRT= PDNP[vind]/PDB[vind]
    print("LRT",LRT)

## Probability of detection
THPD=qfunc((Gamth-A)/np.sqrt(V/N));
print("THPD",THPD)

