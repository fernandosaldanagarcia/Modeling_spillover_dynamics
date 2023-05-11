#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:32:24 2022

@author: fsaldana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:46:14 2022

@author: fsaldana
"""

import numpy as np
import matplotlib.pyplot as plt



##############################################################
############        code chunk 1             #################
########### Setting the parameters and IC    #################
##############################################################

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

beta = 0.7*recovery/N
phi = 1.5
eta = 0.2
alfa = 0


"""
   spillover rate
"""

tau = 1e-5

"""
   IC animal
"""

M = N              # total population
Ia0 = 5000         # initial infected animal
Sa0 = M - Ia0      # initial susceptible animal


d = 1/(1*365)     # mortality rate
lamnda = 2*d      # transmission rate --> R0a = 2




##############################################################
############        code chunk 2             #################
########### Creating a GillesPy2 Model #######################
##############################################################

# Import the types that'll be needed to define your Model.
from gillespy2.core import (
    Model,
    Species,
    Reaction,
    Parameter
)



class SI_SHARS(Model):
     def __init__(self, parameter_values=None):

            
            Model.__init__(self, name="SI_SHARS")
            
            """
            Parameters
            """
            
            eta_beta = Parameter(name="eta_beta", expression=eta*beta)
            uno_eta_beta = Parameter(name="uno_eta_beta", expression=(1-eta)*beta)
            eta_phi_beta = Parameter(name="eta_phi_beta", expression=eta*phi*beta)
            uno_eta_phi_beta = Parameter(name="uno_eta_phi_beta", expression=(1-eta)*phi*beta)
            eta_tau = Parameter(name="eta_tau", expression=eta*tau/M)
            uno_eta_tau = Parameter(name="uno_eta_tau", expression=(1-eta)*tau/M)
            gamma = Parameter(name="gamma", expression=recovery)
            alpha = Parameter(name="alpha", expression=alfa)
            death = Parameter(name="mu", expression=mu)
            infectionA = Parameter(name="infectionA", expression=lamnda/M)
            deathA = Parameter(name="deathA", expression=d)
            
            # Add the Parameters to the Model.
            self.add_parameter([eta_beta, uno_eta_beta, eta_phi_beta, uno_eta_phi_beta,
                                eta_tau, uno_eta_tau, gamma, alpha, death, infectionA, deathA])
            
            """
            Species 
            """
            
            Sa = Species(name="Sa", initial_value=Sa0)
            Ia = Species(name="Ia", initial_value=Ia0)
            S = Species(name="S", initial_value=S0)
            H = Species(name="H", initial_value=H0)
            A = Species(name="A", initial_value=A0)
            R = Species(name="R", initial_value=R0)
            
            # Add the Species to the Model.
            self.add_species([Sa, Ia, S, H, A, R])
            
            """
            Reactions
            """

            r1 = Reaction(
                    name="infectionH",
                    reactants={S: 1, H: 1}, 
                    products={S: 0, H: 2},
                    rate=eta_beta
                )
            
            r2 = Reaction(
                    name="infectionH2",
                    reactants={S:1, H: 1}, 
                    products={S: 0, A: 1, H: 1},
                    rate=uno_eta_beta
                )
            
            r3 = Reaction(
                    name="infectionAs",
                    reactants={S: 1, A: 1}, 
                    products={S: 0, H: 1, A: 1},
                    rate=eta_phi_beta
                )
            
            r4 = Reaction(
                    name="infectionAs2",
                    reactants={S:1, A: 1}, 
                    products={S: 0, A: 2},
                    rate=uno_eta_phi_beta
                )
            
            r5 = Reaction(
                    name="spillover1",
                    reactants={S: 1, Ia: 1}, 
                    products= {S: 0, Ia: 1, H: 1},
                    rate=eta_tau
                )
            
            r6 = Reaction(
                    name="spillover2",
                    reactants={S: 1, Ia:1}, 
                    products= {S: 0, Ia: 1, A: 1},
                    rate=uno_eta_tau
                )
            
            
            r7 = Reaction(
                    name="recoveryH",
                    reactants={H: 1}, 
                    products={H: 0, R: 1},
                    rate=gamma
                )
            
            r8 = Reaction(
                    name="recoveryAs",
                    reactants={A: 1}, 
                    products={A: 0, R: 1},
                    rate=gamma
                )
            
            r9 = Reaction(
                    name="lossImmunity",
                    reactants={R: 1},
                    products={S: 1, R: 0},
                    rate=alpha
                )
            
            
            r10 = Reaction(
                    name="deathH",
                    reactants={H: 1}, 
                    products={S: 1, H: 0},
                    rate=death
                )
            
            r11 = Reaction(
                    name="deathAs",
                    reactants={A: 1}, 
                    products={S: 1, A: 0},
                    rate=death
                )
            
            r12 = Reaction(
                    name="deathR",
                    reactants={R: 1}, 
                    products={S: 1, R: 0},
                    rate=death
                )
            
            
            
            
            r13 = Reaction(
                    name="infectionAnimal",
                    reactants={Sa: 1, Ia: 1}, 
                    products={Sa: 0, Ia: 2},
                    rate=infectionA
                )
            
            
            r14 = Reaction(
                    name="deathIa",
                    reactants={Ia: 1}, 
                    products={Sa: 1, Ia: 0},
                    rate=deathA
                )
            
            
            self.add_reaction([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14])
            
            # Use NumPy to set the timespan of the Model.
            self.timespan(np.linspace(0, 800, 800))
            
            
            
model = SI_SHARS()



##############################################################
################   code chunk  3          ####################
#################### Solving the Model #######################
##############################################################




# ---------------------------------------Stochastic solution of the model
from gillespy2.solvers.numpy import NumPySSASolver


num_traj = 1
stochastic_sol = model.run(solver=NumPySSASolver, number_of_trajectories=num_traj, seed=8)




##############################################################
################   code chunk  4          ####################
############  Plot the results of the simulations ############
##############################################################




stochastic_sol.plot(
	# Set the title of the X and Y axis.
	xaxis_label="Time",
	yaxis_label="Individuals", 

	# Set the title of the plot.
	title=r'Susceptible',

	# Set to True to show the legend, False to hide it.
	show_legend=True,


	style="fivethirtyeight",
    
    # choose species to plot
    included_species_list=["S"]
)


stochastic_sol.plot(
	# Set the title of the X and Y axis.
	xaxis_label="Time",
	yaxis_label="Individuals", 

	# Set the title of the plot.
	title=r'Hospitalized',

	# Set to True to show the legend, False to hide it.
	show_legend=True,


	style="fivethirtyeight",
    
    # choose species to plot
    included_species_list=["H"]
)


stochastic_sol.plot(
	# Set the title of the X and Y axis.
	xaxis_label="Time",
	yaxis_label="Individuals", 

	# Set the title of the plot.
	title=r'Asymptomatic',

	# Set to True to show the legend, False to hide it.
	show_legend=True,


	style="fivethirtyeight",
    
    # choose species to plot
    included_species_list=["A"]
)


stochastic_sol.plot(
	# Set the title of the X and Y axis.
	xaxis_label="Time",
	yaxis_label="Individuals", 

	# Set the title of the plot.
	title=r'Recovered',

	# Set to True to show the legend, False to hide it.
	show_legend=True,


	style="fivethirtyeight",
    
    # choose species to plot
    included_species_list=["R"]
)
    
