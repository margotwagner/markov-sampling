import numpy as np
import matplotlib.pyplot as plt

def two_state_markov_binom_sampling(n_c0,n_c1, p_c0c1,p_c1c0):
    ''' Function that samples from a binomial distribution to find the number of channels that transition at each
    time step'''

    beta_c0 = p_c0c1 / (1 - p_c0c1)                 # shorthand
    beta_c1 = p_c1c0 / (1 - p_c1c0)                 # shorthand
    alpha_c0 = (1 - p_c0c1)**n_c0                   # probability no channels transition from c0 to c1
    alpha_c1 = (1 - p_c1c0)**n_c1                   # probability no channels transition from c0 to c1

    n_c0c1 = 0                                      # number of channels transitioning from c0 to c1
    n_c1c0 = 0                                      # number of channels transitioning from c1 to c0
    p_i_c0c1 = alpha_c0                             # probability i channels transition from c0 to c1
    sum_p_i_c0c1 = p_i_c0c1                         # probability i or fewer channels change from c0 to c1
    p_i_c1c0 = alpha_c1                             # probability i channels transition from c1 to c0
    sum_p_i_c1c0 = p_i_c1c0                         # probability i or fewer channels change from c0 to c1

    u_c0c1 = np.random.uniform()                    # random number generator from c0 to c1
    u_c1c0 = np.random.uniform()                    # random number generator from c1 to c0

    #print('C0C1 random number: ', u_c0c1)
    #print(n_c0c1,p_i_c0c1, sum_p_i_c0c1)

    #print('C1C0 random number: ', u_c1c0)
    #print(n_c1c0,p_i_c1c0, sum_p_i_c1c0)

    # (inverse transform sampling)
    while u_c0c1 > sum_p_i_c0c1:
        p_i_c0c1 = (n_c0 - n_c0c1)/(n_c0c1 + 1) * beta_c0 * p_i_c0c1          # update probabilitiees
        sum_p_i_c0c1 = sum_p_i_c0c1 + p_i_c0c1
        n_c0c1 = n_c0c1 + 1                                                   # update number of channels opening
        #print(n_c0c1,p_i_c0c1,sum_p_i_c0c1)

    #print(n_c0c1, 'channel(s) transition from c0 to c1')            # number of channels that opens

    while u_c1c0 > sum_p_i_c1c0:
        p_i_c1c0 = (n_c1 - n_c1c0)/(n_c1c0 + 1) * beta_c1 * p_i_c1c0          # update probabilitiees
        sum_p_i_c1c0 = sum_p_i_c1c0 + p_i_c1c0
        n_c1c0 = n_c1c0 + 1                                                   # update number of channels opening
        #print(n_c1c0,p_i_c1c0,sum_p_i_c1c0)

    #print(n_c1c0, 'channel(s) transition from c1 to c0')            # number of channels that opens

    n_c0 = n_c0 - n_c0c1 + n_c1c0
    n_c1 = n_c1 - n_c1c0 + n_c0c1

    # n_c1_frac = n_c1 / sum([n_c0, n_c1])

    return n_c0, n_c1

def V_m_t(t_index):
    ''' Applied input voltage'''
    return 30 * np.heaviside((t_array[t_index]-8), 0)

def alpha_c0c1(t_index):
    ''' Rate of transition from c0 to c1'''
    return alpha_c0c1_0 * np.exp(lambda_const*V_m_t(t_index))

def beta_c1c0(t_index):
    '''# Rate of transition from c1 to c0'''
    return beta_c1c0_0 * np.exp(-mu*V_m_t(t_index))

def markov_sampling_time_course(time_start,time_stop,time_step, n_c0, n_c1):
    t_array = np.arange(time_start, time_stop + time_step, time_step)
    n_c1_frac_array = []
    n_c1_frac_array = np.append(n_c1_frac_array, n_c1 / sum([n_c0, n_c1]))

    for t_index in range(1, len(t_array)):
        p_c0c1 = alpha_c0c1(t_index)*time_step
        p_c1c0 = beta_c1c0(t_index)*time_step
        [n_c0, n_c1] = two_state_markov_binom_sampling(n_c0,n_c1,p_c0c1,p_c1c0)
        n_c1_frac_array = np.append(n_c1_frac_array,n_c1/sum([n_c0,n_c1]))

    return t_array, n_c1_frac_array

# Define constants for transition probabilities
alpha_c0c1_0 = 0.3          # [=] 1/s
beta_c1c0_0 = 0.1       # [=] 1/s
lambda_const = 0.01      # [=] 1/mV
mu = 0.01               # [=] 1/mV

# Initial conditions
n_c0_0 = 100
n_c1_0 = 0
time_start = 0
time_step = 1
time_stop = 16

[t_array, n_c1_frac_array] = markov_sampling_time_course(time_start, time_stop, time_step, n_c0_0, n_c1_0)

plt.figure()
plt.plot(t_array, n_c1_frac_array, label = 'C1')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Fraction of gates open')
plt.show()
