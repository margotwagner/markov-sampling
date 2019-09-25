import numpy as np
import matplotlib.pyplot as plt

def V_m_t(t_index):
    ''' Applied input voltage'''
    return 30 * np.heaviside((t_array[t_index]-8), 0)

def alpha_c0c1(t_index):
    ''' Rate of transition from c0 to c1'''
    return alpha_c0c1_0 * np.exp(lambda_const*V_m_t(t_index))

def beta_c1c0(t_index):
    '''# Rate of transition from c1 to c0'''
    return beta_c1c0_0 * np.exp(-mu*V_m_t(t_index))

# n_per_state = [100, 0] number of channels in each state
    # n_c0 = n_states[0]
    # n_c1 = n_states[1]
# p_transition = [0.3, 0.1] probability of transition for each state
    # p_c0c1 = p_transition[0]
    # p_c1c0 = p_transition[1]

# n_transition = [26 0] number of channels transitioning between states
    # n_c0c1 = n_transition[0]      number of channels transitioning from c0 to c1
    # n_c1c0 = n_transition[1]      no. of channels transitioning from c1 to c0

# p_i_transition = [..., ...] probability i channels transitions between states
    # p_i_c0c1 = p_i_transition[0]
    # p_i_c1c0 = p_i_transition[1]

# cum_p_i_transition = [..., ...] cumulative probability of i or fewer channels transitioning
    # sum_p_i_c0c1 = cum_p_i_transition[0]
    # sum_p_i_c1c0 = cum_p_i_transition[1]

# u_transition = [.602, .257] uniform random numbers generated for stochastic transitions
    # u_c0c1 = u_transition[0]      unif for c0 to c1
    # u_c1c0 = u_transition[1]      unif for c1 to c0

# beta shorthand
    # beta_c0 = beta[0]
    # beta_c1 = beta[1]

# alpha initial probabilities
    # alpha_c0 = alpha[0]
    # alpha_c1 = alpha[1]



def two_state_markov_binom_sampling(n_per_state, p_transition):
    ''' Function that samples from a binomial distribution to find the number of channels that transition at each
    time step'''
    # n_per_state and p_transition should be numpy arrays!!

    # Initialize arrays
    beta = np.empty(2)
    alpha = np.empty(2)
    n_transition = np.zeros((2), dtype='int')
    p_i_transition = np.empty(2)
    cum_p_i_transition = np.empty(2)
    u_transition = np.empty(2)


    beta[0] = p_transition[0] / (1 - p_transition[0])                 # shorthand
    alpha[0] = (1 - p_transition[0]) ** n_per_state[0]  # probability no channels transition from c0 to c1
    p_i_transition[0] = alpha[0]                             # probability i channels transition from c0 to c1
    cum_p_i_transition[0] = p_i_transition[0]                        # probability i or fewer channels change from c0 to c1
    u_transition[0] = np.random.uniform()                    # random number generator from c0 to c1

    # (inverse transform sampling)
    while u_transition[0] > cum_p_i_transition[0]:
        p_i_transition[0]= (n_per_state[0] - n_transition[0])/(n_transition[0] + 1) * beta[0] * p_i_transition[0]         # update probabilitiees
        cum_p_i_transition[0] = cum_p_i_transition[0] + p_i_transition[0]
        n_transition[0] = n_transition[0] + 1


    beta[1] = p_transition[1] / (1 - p_transition[1])                 # shorthand
    alpha[1] = (1 - p_transition[1])**n_per_state[1]                   # probability no channels transition from c1 to c0
    p_i_transition[1] = alpha[1]                             # probability i channels transition from c1 to c0
    cum_p_i_transition[1] = p_transition[1]                          # probability i or fewer channels change from c0 to c1
    u_transition[1] = np.random.uniform()                    # random number generator from c1 to c0

    while u_transition[1] > cum_p_i_transition[1]:
        p_transition[1] = (n_per_state[1] - n_transition[1] )/(n_transition[1] + 1) * beta[1] * p_transition[1]         # update probabilitiees
        cum_p_i_transition[1] = cum_p_i_transition[1] + p_transition[1]
        n_transition[1] = n_transition[1]  + 1

    n_per_state[0] = n_per_state[0] - n_transition[0] + n_transition[1]
    n_per_state[1] = n_per_state[1] - n_transition[1] + n_transition[0]

    return n_per_state