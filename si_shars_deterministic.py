#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:42:21 2022

@author: fsaldana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:14:30 2022

@author: fsaldana
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


##################################################################
############   Parameters -initial conditions  ###################
##################################################################



"""
   IC human
"""

N = 10000
A0 = 0
H0 = 0
S0 = N -A0 - H0
R0 = 0


"""
   parameters human
"""

recovery = 1/15
mu = 1/(50*365)

beta = 0.5*recovery
phi = 1.5
eta = 0.2
alfa = 1/60

tau = 1e-4

"""
   IC animal
"""

M = N              # total population
Ia0 = 5000         # initial infected animal
Sa0 = M - Ia0      # initial susceptible animal


d = 1/(1*365)     # mortality rate
lamnda = 2*d      # transmission rate --> R0a = 2

"""
   human R0
"""

R0_h = (beta/recovery)*(eta + (1-eta)*phi)
print(R0_h)

Tau = np.array([1e-2, 1e-3, 5e-4, 1e-4]) 

##############################################################
############# Model equations ################################
##############################################################



def model(y, t, tau):
    Sr, Ir, S, H, A, R = y
    # Equations
    dSrdt = d*M - lamnda*Sr*Ir/M - d*Sr
    dIrdt = lamnda*Sr*Ir/M - d*Ir
    dSdt = mu*N - beta*S*H/N - beta*phi*S*A/N - tau*S*Ir/M + alfa*R - mu*S
    dHdt = eta*(beta*S*H/N + beta*phi*S*A/N + tau*S*Ir/M) - (recovery + mu)*H
    dAdt = (1-eta)*(beta*S*H/N + beta*phi*S*A/N + tau*S*Ir/M) - (recovery + mu)*A
    dRdt = recovery*(H+A) - (alfa + mu)*R
    return dSrdt, dIrdt, dSdt, dHdt, dAdt, dRdt





################################################################
############# Integration and plots  ###########################
################################################################

# A grid of time points (in days)
t = np.linspace(0, 500, 501)

# Initial conditions vector
y0 = Sa0, Ia0, S0, H0, A0, R0

plt.style.use('fivethirtyeight')

############# infected class  ##################################
fig, ax = plt.subplots(figsize=(16,10), tight_layout=True)


# Integrate the equations over the time grid, t.
dim_tau = len(Tau)
for i in range(dim_tau):
    sol = odeint(model, y0, t, args=(Tau[i],))
    Sr, Ir, S, H, A, R = sol.T
    tauLabel = Tau[i]
    ax.plot(t, H, linewidth=6, label=r'$\tau$=%.4f' %tauLabel)

# Customise some display properties
ax.set_xlabel('Time (days)', fontsize=30)
ax.set_ylabel('Hospitalized individuals', fontsize=30)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#ax.set_title('Hospitalized class', labelsize=30)
ax.legend(loc='best', fontsize=30)
#plt.savefig("endemic_reservoir.pdf", bbox_inches = 'tight')
plt.show()


############# asymptomatic class  ##################################
fig2, ax2 = plt.subplots(figsize=(16,10), tight_layout=True)


# Integrate the equations over the time grid, t.
dim_tau = len(Tau)
for i in range(dim_tau):
    sol2 = odeint(model, y0, t, args=(Tau[i],))
    Sr2, Ir2, S2, H2, A2, R2 = sol2.T
    tauLabel = Tau[i]
    ax2.plot(t, A2, linewidth=6, label=r'$\tau$=%.4f' %tauLabel)

# Customise some display properties
ax2.set_xlabel('Time (days)', fontsize=30)
ax2.set_ylabel('Asymptomatic individuals', fontsize=30)
ax2.tick_params(axis='x', labelsize=30)
ax2.tick_params(axis='y', labelsize=30)
#ax.set_title('Hospitalized class', labelsize=30)
ax2.legend(loc='best', fontsize=30)
#plt.savefig("endemic_reservoir.pdf", bbox_inches = 'tight')
plt.show()