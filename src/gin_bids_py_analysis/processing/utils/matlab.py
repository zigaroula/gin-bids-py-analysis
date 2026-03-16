import numpy as np

def matlab_tukeywin(N, r):
    if r <= 0:
        return np.ones(N, dtype=np.float64)
    elif r >= 1:
        n = np.arange(N, dtype=np.float64)
        return 0.5*(1 - np.cos(2*np.pi*n/(N-1)))
    else:
        w = np.ones(N, dtype=np.float64)
        edge = int(np.floor(r*(N-1)/2.0))
        n = np.arange(0, edge+1, dtype=np.float64)
        w[:edge+1] = 0.5*(1 + np.cos(np.pi*(2*n/(r*(N-1)) - 1)))
        n = np.arange(N-edge-1, N, dtype=np.float64)
        w[N-edge-1:] = 0.5*(1 + np.cos(np.pi*(2*(n-(N-1))/(r*(N-1)) + 1)))
        return w