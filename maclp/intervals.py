import numpy as np
import scipy.stats as sps
from numba import njit, prange, jit
import warnings

# checked 12-15: To-do: update to numba-stats    
def binomial_prop_itvls(x, n, alpha, itvl_type = 'Clopper-Pearson'):
    """
    Exact intervals:
        1. 'Clopper-Pearson'
        2. ...
        3. ...
    Asymptotic intervals:
        4. xxx
        5. xxx

    """
    if itvl_type == 'Clopper-Pearson':
        # Clopper-Pearson interval
        if x == 0:
            lower = 0.0; upper = 1 - (0.5*alpha)**(1/n)
        elif x == n:
            lower = (0.5*alpha)**(1/n); upper = 1.0
        else:
            lower = sps.beta.ppf(alpha / 2.0, x, n - x + 1)
            upper = sps.beta.ppf(1.0 - alpha / 2.0, x + 1, n - x)

    elif itvl_type == 'Jeffreys-Prior':
        lower = sps.beta.ppf(alpha / 2.0, x+0.5, n - x + 0.5)
        upper = sps.beta.ppf(1.0 - alpha / 2.0, x + 0.5, n - x + 0.5)

    elif itvl_type == 'Uniform-Prior':
        lower = sps.beta.ppf(alpha / 2.0, x+1.0, n - x + 1)
        upper = sps.beta.ppf(1.0 - alpha / 2.0, x + 1, n - x +1.0)

    else:
        raise ValueError(f'{itvl_type} is not prepared, please provide more info.')
    return np.array([lower, upper])

def find_C_itvls(confusion_matrix, sample_size, K, alpha_matrix=None, itvl_type = 'Clopper-Pearson'):
    # A float delta is the total error level of all K**2 + K intervals of C and q
    # (Bonferroni): each entry gets delta / (K**2 + K), so (C, q) has joint level 1 - delta.
    if isinstance(alpha_matrix, float):
        alpha_matrix = np.full((K, K), alpha_matrix / (K**2 + K))

    C_low = np.empty((K, K))
    C_up  = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            itvl = binomial_prop_itvls(confusion_matrix[i, j], sample_size, alpha_matrix[i, j], itvl_type)
            C_low[i, j], C_up[i, j] = itvl

    return {"C_low": C_low, "C_up": C_up}

def find_q_itvls(q_est, sample_size, K, alpha_vec=None, itvl_type = 'Clopper-Pearson'):
    # A float delta is split as in find_C_itvls: delta / (K**2 + K) per entry.
    if isinstance(alpha_vec, float):
        alpha_vec = np.full(K, alpha_vec / (K**2 + K))


    q_low = np.zeros(K)
    q_up  = np.zeros(K)
    for i in range(K):
        x = int(q_est[i])  # number of target points predicted as class i
        conf_itvl = binomial_prop_itvls(x, sample_size, alpha_vec[i], itvl_type)
        q_low[i] = conf_itvl[0]
        q_up[i]  = conf_itvl[1]
    #return np.column_stack([q_low, q_up])
    return {"q_low": q_low, "q_up": q_up}

if __name__=="__main__":
    print('This is MaC-LP intervals module.')

    




















