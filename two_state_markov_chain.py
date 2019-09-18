import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Define constants
alpha_0 = 0.7           # [=] 1/s
beta_0 = 0.9            # [=] 1/s
lambda_const = 0.01      # [=] 1/mV
mu = 0.01               # [=] 1/mV

###########################################################
################# Mean field response #####################
###########################################################

# Time series of simulation
delta_t = 0.01           # time step (delta t, sec)
t = np.arange(0,16,delta_t)         # simulation in time (sec)
n = 0 # gate starts in state 0 (single gate)

# Applied input voltage
def V_m_t(t):
    return 30 * np.heaviside(t-8, 0)

# Opening rate
def alpha_n(t):
    return alpha_0 * np.exp(lambda_const*V_m_t(t))

# Closing rate
def beta_n(t):
    return beta_0 * np.exp(-mu*V_m_t(t))

# Mean field Markov rate equation
def dn_dt(n, t):
    return alpha_n(t)*(1-n) - beta_n(t)*n

n_t = odeint(dn_dt, n, t)
plt.figure()
plt.plot(t, n_t, color='black')
plt.title("Mean field Markov response for one gate")
plt.xlabel('time (sec)')
plt.ylabel('Fraction of open channels')
plt.show()


###########################################################
##################### Markov response #####################
###########################################################

# Time series of simulation
delta_t = 0.01           # time step (delta t, sec)
t = np.arange(0,16,delta_t)         # simulation in time (sec)

# Applied input voltage
def V_m_t(t_index):
    return 30 * np.heaviside(t[t_index]-8, 0)

# Opening rate
def alpha_n(t_index):
    return alpha_0 * np.exp(lambda_const*V_m_t(t_index))

# Closing rate
def beta_n(t_index):
    return beta_0 * np.exp(-mu*V_m_t(t_index))

def markov(n_channels):
    n_open = np.zeros_like(t)
    gates = np.zeros(n_channels)
    for t_index in range(len(t)):
        for gate_index in range(len(gates)):
            prob_sample = np.random.uniform(0, 1)
            if gates[gate_index] == 0:
                gates[gate_index] = prob_sample < (delta_t * alpha_n(t_index))
            if gates[gate_index] == 1:
                gates[gate_index] = prob_sample < (1 - delta_t * beta_n(t_index))
        n_open[t_index] = sum(gates) / n_channels
    return n_open

n_open_N1 = markov(1)
n_open_N400 = markov(400)
n_open_N1000 = markov(1000)


plt.figure()
plt.plot(t, n_open_N1, label = 'N = 1')
plt.plot(t, n_open_N400, label = 'N = 400')
plt.plot(t, n_open_N1000, label = 'N = 1000')
plt.plot(t, n_t, color='black', label = 'Mean field')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Fraction of gates open')
plt.show()


###########################################################
############### Pooled Markov response ####################
###########################################################
n_gates = 100           # 100 gates to start (all closed)
#n_open = np.zeros(5) # 5 time steps
gates = np.zeros(n_gates)   # gate states
n_closed = n_gates
n_open = 0


# Time step 1
prob_sample = np.random.uniform()
if prob_sample > 0.3:
    n_open = round(prob_sample*n_gates)
    n_closed = n_gates - n_open

prob_sample = np.random.uniform()
if prob_sample > 0.3:
    # both transition
if prob_sample <= 0.3 and prob_sample > 0.1:
    # only open transitions to closed
    n_closed = n_closed + round(prob_sample*n_open)
    n_open = n_gates - n_closed
# else
    # nothing happen

prob_sample = np.random.uniform()
if prob_sample > 0.3:
    n_closed_transition = 



