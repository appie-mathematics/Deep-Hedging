import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm

def brownian(T, dt, M):
    ''' 
    Generate a brownian path with from 0 to T with time step dt
    
    Based on:
        X(t+dt) = X(t) + N(0, delta**2 * dt)

    Code adapted from:
    "https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html"

    Parameters
    ----------
    T : float
        Final time of the brownian path
    dt : float
        Time step of the brownian path
    M : int
        Number of paths to generate
    '''

    K = int(T/dt)
    X = np.zeros((M, K+1))

    # Create the increments of the brownian motion
    r = norm.rvs(size=(M, K), scale=np.sqrt(dt))
    
    # Cumulative sum of the random numbers
    X[:, 1:] = np.cumsum(r, axis=1)

    return X.T

def geometric_brownian_motion(x0, T, dt, sigma, mu, M):
    brownian_path = brownian(T, dt, M)
    return x0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * brownian_path)

def bs_option(S0, strike, T, r, sigma, type):
    '''
    Black-Scholes option pricing formula

    Parameters
    ----------
    S0 : float
        Initial value of the underlying
    strike : float
        Strike price of the option
    T : float
        Time to maturity of the option
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying
    type : str
        Type of option, either 'call' or 'put'
    '''
    d1 = (np.log(S0/strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == 'call':
        return S0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    elif type == 'put':
        return strike * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError('Option type must be either "call" or "put"')

def bs_delta(S0, strike, T, r, sigma, type):
    '''
    Black-Scholes delta

    Parameters
    ----------
    S0 : float
        Initial value of the underlying
    strike : float
        Strike price of the option
    T : float
        Time to maturity of the option
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying
    type : str
        Type of option, either 'call' or 'put'
    '''
    d1 = (np.log(S0/strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if type == 'call':
        return norm.cdf(d1)
    elif type == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError('Option type must be either "call" or "put"')

if __name__ == "__main__":
    # Parameters
    S0 = 100
    strike = 100
    T = 30/365
    r = 0.02
    sigma = 0.2
    dt = 1/365
    M = 1

    delta = bs_delta(S0, strike, T, r, sigma, 'call')

    # plot surface of delta with respect to stock price and volatility
    # we will use the same parameters as before, but we will vary the stock price and volatility

    S0 = np.linspace(50, 150, 100)
    sigma = np.linspace(0.1, 0.5, 100)

    S0, sigma = np.meshgrid(S0, sigma)

    delta = bs_delta(S0, strike, T, r, sigma, 'call')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
    surf = ax.plot_surface(S0, sigma, delta, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Stock price')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Delta')
    plt.show()