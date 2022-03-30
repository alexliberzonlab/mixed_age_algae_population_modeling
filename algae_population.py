# collect all the functions to remove duplication
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# Create a new variable and store all
# built-in functions within it using dir( ).
not_my_data = set(dir())


# some definitions that we keep through the simulations
t0 = 0
tend = 360

# we solve for daily population discretization:
days = np.arange(t0, tend)
n_days = len(days)

K = 10  # kg/m^3 total density kind of thing
lamda = 1  # day by day aging

theta = 0.1*np.exp(-days/(120/math.log(2)))
mu = 0.05*np.exp(-days/(120/math.log(2)))

# default values were constant dilution ratio of 10%
dilution = 5.0  # percents
tau = np.inf

# external supply of inhibitor by nutrients, units of I,
# like direct supply to water
gammai = 0.0


# initial mass
m0 = 0.2

scenarios = {'100/0': [(0, m0)],
             '90/10': [(0, m0*0.90), (120, m0*0.10)],
             '80/20': [(0, m0*0.80), (120, m0*0.20)],
             '70/30': [(0, m0*0.70), (120, m0*0.30)],
             '60/40': [(0, m0*0.60), (120, m0*0.40)],
             '50/50': [(0, m0/2), (120, m0/2)],
             '40/60': [(0, m0*0.40), (120, m0*0.60)],
             '30/70': [(0, m0*0.30), (120, m0*0.70)],             
             '20/80': [(0, m0*0.2), (120, m0*0.8)],
             '10/90': [(0, m0*0.10), (120, m0*0.90)],             
             '0/100': [(120, m0)],

}


methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
method = methods[0]


def logistic(x,L=1,k=1,x0=0):
    """ General logistic function """
    return L / (1 + np.exp(-k*(x-x0)))

def f(x):
    """ inverted logistic function to encounter the effect of inhibitor 
        values decrease from 1 to 0 in the range of [0,1]
        todo: we need to find the range of inhibitor, so far considered as a 
        proportion of the total mass of algae
    """
    return 1-logistic(x,L=1,k=10,x0=.5)

def r(t):
    """ growth function 
        measured in percents of growth per day, e.g. 
        0.5 means 50% per day
        Alex G. suggested to use from 50% for the young ones to 5% for the
        120 days old ones and then keep it at 5% roughly

    """
    return 0.45*(0.1+np.exp(-t/(30/math.log(2))))

def sigma(t):
    """ destruction function 
    
    Alex G. suggested to set it to the 1/2 lambda, i.e. if 
    the growth is day by day, then the destruction is two days,
    empirically seems to be very strong, or the f(I) needs to be
    much weaker.
    """
    return 0.3 

def xi(t, dilution=10, tau=np.inf):
    """ time varying leakage or water replacement
        measured in percents of inhibitor concentration or amount 
        0.5 means 50% per day
        max = 100 # in per-cents - removes everything
        tau = 1 means replacement every day
        tai = np.inf means constant value in the output

    """
    tau = tau/np.log(2) # half-time
    dilution = dilution/100 # units not percents
    out = dilution * np.exp(-t/tau)
    return out

def evolution(t, y, K, lamda, xi, gammai, theta, mu, dilution, tau, sigma):
    """ t : time
        y : vector of state variables containing: 
            a : vector of age masses, let's start with a0, a1 
            I : inhibitor's content
        r : growth rate vector (per age)
        K : saturation (logistic growth model)
        lambda : vector of 1/time resolution, e.g. 1/7 (for day by day age)
        sigma : rate of algae degradation /destruction when there are no inhibitors
        theta : rate of creation of inhibitor
        mu : rate of uptake of inhibitor from the surrounding
        xi : rate of leakage, destruction of inhibitor, losses.
        gammai :   is the nutrient supply flux in the units of inhibitor concentration 
 

    """
    a = y[:-1]
    I = y[-1]
        
    dydt = np.zeros_like(y) 
    # age 0, birth
    dydt[0] = ((r(0)*a[0]*(1-np.sum(a)/K)) - lamda * a[0] - sigma(0)*a[0]*f(I))
    # from age 1 and on:
    for i in range(1,len(a)):
        dydt[i] = ((r(i)*a[i]*(1-np.sum(a)/K)) + lamda*a[i-1] - lamda*a[i] - sigma(i)*a[i]*f(I))

    dydt[-1] = np.sum(a@theta) - I * np.sum(a@mu) - xi(i, dilution, tau)*I + gammai

    # prevent negative population
    for i, j in zip(y, dydt):
        if i < 0.005 and j < 0.:  # less then 5 gram or .5% and negative slope
            j = 0.
        if i < 0:
            j = 0

    return dydt


def sporulation(t, y, K, lamda, xi, gammai, theta, mu, dilution, tau, sigma):
    """ this event tracking and events = limit, limit.terminal = True will
    stop the simulation on catastrophy of the death of the population
    when all the mass has disappeared
    """
    return np.sum(y[:-1])


def scenario_to_age_distribution(s):
    """ scenarios see above, every dict provides the name of the
    scenario and age mass distribution. here we convert it to the
    initial values for the ODE solver
    """

    ages = s[1]
    a = np.zeros((n_days))
    for ag in ages:
        a[ag[0]] = ag[1]

    return a


def plot_all_results(solutions, tend=None, K=10):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for sol in solutions:

        t0 = sol.t[0]
        if tend is None:
            tend = sol.t[-1]

        if sol.t_events[0].size > 0 and sol.t_events[0] < tend:
            print(f'sporulation event at {sol.t_events[0]}')
            tend = sol.t_events[0]

        t = np.arange(t0, tend)
        z = sol.sol(t)


        
        # mass and inhibitor
        biomass = z[:-1, :] 
        I = z[-1,:]

        # what we gain is:
        revenue = np.sum( biomass.T - biomass[:,0], axis=1)
        

        ax[0].plot(t, revenue,'-o',label = sol['s'][0])
        ax[0].set_xlabel('days')
        ax[0].set_ylabel(r'Revenue')
        ax[0].set_ylim([-1, K])
        ax[0].set_yscale('symlog')
        ax[0].legend()

        ax[1].plot(t,I,'-o',label= sol['s'][0])
        ax[1].set_xlabel('days')
        ax[1].set_ylabel(r'$I$')

        fmt = mpl.ticker.StrMethodFormatter("{x:g}")
        ax[0].yaxis.set_major_formatter(fmt)
        ax[0].yaxis.set_minor_formatter(fmt)

        ax[1].yaxis.set_major_formatter(fmt)
        ax[1].yaxis.set_minor_formatter(fmt)

    plt.show()
    return fig, ax

def interp2d_with_nan(x, y, z):
    """ Interpolates 2D matrix z removing NaNs """
    from scipy.interpolate import interp2d

    # Generate some test data:
    xx, yy = np.meshgrid(x, y)

    # Interpolation functions:
    nan_map = np.zeros_like( z )
    nan_map[ np.isnan(z) ] = 1

    filled_z = z.copy()
    filled_z[ np.isnan(z) ] = 0

    f = interp2d(x, y, filled_z, kind='linear')
    f_nan = interp2d(x, y, nan_map, kind='linear')     

    # Interpolation on new points:
    xnew = np.linspace(x.min(), x.max(), 30)
    ynew = np.linspace(y.min(), y.max(), 30)

    z_new = f(xnew, ynew)
    nan_new = f_nan( xnew, ynew )
    z_new[ nan_new > 0.5 ] = np.nan

    return xnew, ynew, z_new