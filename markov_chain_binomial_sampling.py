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

# Define constants for transition probabilities
p_c0c1_0 = 0.7           # [=] 1/s
p_c1c0_0 = 0.9            # [=] 1/s
lambda_const = 0.01      # [=] 1/mV
mu = 0.01               # [=] 1/mV

def V_m_t(t_index):
    ''' Applied input voltage'''
    return 30 * np.heaviside((t_array[t_index]-8), 0)

def p_c0c1(t_index):
    ''' Rate of transition from c0 to c1'''
    return p_c0c1_0 * np.exp(lambda_const*V_m_t(t_index))

def p_c1c0(t_index):
    '''# Rate of transition from c1 to c0'''
    return p_c1c0_0 * np.exp(-mu*V_m_t(t_index))

#p_c0c1 = 0.3
#p_c1c0 = 0.1

n_c0 = 100
n_c1 = 0
dt = 0.1
t_end = 16
t_array = np.arange(0, t_end+dt, dt)
n_c1_frac_array = []
n_c1_frac_array = np.append(n_c1_frac_array,n_c1/sum([n_c0,n_c1]))

for t_index in range(1, len(t_array)):
    [n_c0, n_c1] = two_state_markov_binom_sampling(n_c0, n_c1, 0.7, 0.9)
    [n_c0, n_c1] = two_state_markov_binom_sampling(n_c0,n_c1,p_c0c1(t_index),p_c1c0(t_index))
    print(n_c0, n_c1)
    n_c1_frac_array = np.append(n_c1_frac_array,n_c1/sum([n_c0,n_c1]))

plt.figure()
plt.plot(t_array, n_c1_frac_array, label = 'C1')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Fraction of gates open')
plt.show()
