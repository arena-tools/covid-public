import numpy as np
from numba import njit


@njit
def simulate(S: np.array, I: np.array, N: int, lam=0.3, gamma=0.11):
    """
    Simple SIR (Susceptible-Infected-Recovered) model simulation. 
    Given a numpy array of susceptible and infected time series,
    iterate through time, updating values according to SIR ODE. 
    S[0] and I[0] are the two initial conditions of susceptible and
    infected populations at time=0.
    Args:
        S (np.array): Susceptible time series (zeroth entry is 
                        initial condition S_0)
        I (np.array): Infected time series (zeroth entry is 
                        initial condition I_0)
        lam (float): effective mixing rate (1 / average time between infections/contacts )
        gamma (float): effective removal rate ( 1 / serial_interval )
    Returns:
        S (np.array), I (np.array)
          Updated time series
    """
    for i in range(len(S) - 1):
        I_t = I[i]
        S_t = S[i]
        ds = -lam * I_t * S_t / N
        di = -ds - gamma * I_t
        S[i + 1] = S_t + ds
        I[i + 1] = I_t + di
    return S, I
