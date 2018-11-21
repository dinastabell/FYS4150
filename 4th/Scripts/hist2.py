import numpy as np
import numba
import matplotlib.pyplot as plt
import time
import seaborn as sns
import ising_model as ising
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('talk')

MC_cycles = [400000, 1000000]
num_spins = 20
Temp = [1.0, 2.4]
ordered = True

def plotProbability(ordered):
    """
    This function show the probability distribution of energy as histograms for
    two different temperatures and number of Monte Carlo cycles.
    The calculations are done by the MC function found in the ising_model script.
    The mean energy and magnetization is evaluated from a point where steady
    state is reached.
    """
    for cycles, T in zip(MC_cycles, Temp):
        # Initializing the confiuration of the input spin matrix
        if ordered == False:
            #Random configuration
            spin_matrix = np.random.choice((-1, 1), (num_spins, num_spins))
        else:
            #Ground state
            spin_matrix = np.ones((num_spins,num_spins), np.int8)

        t0 = time.time()
        exp_values = ising.MC(spin_matrix, cycles, T)
        t1 = time.time()

        #If temperature is 1.0
        if T == 1.0:
            sp = 30

            norm = 1/float(cycles-20000)
            energy_avg = np.sum(exp_values[19999:, 0])*norm
            #print("1:",energy_avg**2)
            energy2_avg = np.sum(exp_values[19999:, 2])*norm #E^2
            #print("2:",energy2_avg)
            energy_var = (energy2_avg - energy_avg**2)/(num_spins**2)

            E = exp_values[19999:, 0]
            E = E/(num_spins**2)

        if T == 2.4:
            sp = 160

            norm = 1/float(cycles-40000)
            energy_avg = np.sum(exp_values[39999:, 0])*norm
            #print("1:",energy_avg**2)
            energy2_avg = np.sum(exp_values[39999:, 2])*norm
            #print("2:",energy2_avg)
            energy_var = (energy2_avg - energy_avg**2)/(num_spins**2)

            E = exp_values[19999:, 0]
            E = E/(num_spins**2)

        print("Variance T=%s:"%str(T), energy_var)

        n, bins, patches = plt.hist(E, sp, facecolor='plum')
        plt.xlabel('$E$')
        plt.ylabel('Energy distribution')
        plt.title('Energy distribution at $k_BT=%s$'%str(T))
        plt.grid(True)
        plt.show()

plotProbability(ordered)
