import numpy as np
import scipy as sp
from scipy.stats import truncnorm
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['Katherine How',
                    'Kyle Miller']
    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'UKF'  

    B_nom = 0.8
    B_dist = truncnorm(-0.1*B_nom, 0.1*B_nom, loc=B_nom, scale=B_nom*0.1/2)
    r_nom = 0.425
    r_dist = truncnorm(-0.05*r_nom, 0.05*r_nom, loc=r_nom, scale=r_nom*0.05/2)
    
    if estimatorType == 'EKF':

        # x pos, y pos, theta, r, B
        x = np.array([0, 0, np.pi/4, 0.425, 0.8])
        P = np.diag([6, 6, np.pi/4, r_dist.var(), B_dist.var()])
        # ps, theta, gamma process uncertainty
        var_v = np.diag((20, np.pi/16, np.pi/16))
        # x & y measurement uncertainty
        var_w = np.diag((4.028517994824541, 0.7228186330417723))

        internalState = [x, P, var_v, var_w]
    elif estimatorType == 'UKF':
        # Initialization
        x = np.array([0, 0, np.pi/4, 0.425, 0.8])
        P = np.diag([5, 5, np.pi/4, r_dist.var(), B_dist.var()])
        # process uncertainty
        # var_v = np.diag((0.1, 0.1, np.pi/32, 0, 0))
        var_v = np.diag((15, np.pi/100, np.pi/100))
        # x & y measurement uncertainty
        var_w = np.diag((4.028517994824541, 0.7228186330417723))
        N = 5+3+2
        internalState = [x, P, var_v, var_w, N]
    elif estimatorType == 'PF':
        N = 10*4
        x = np.random.normal(0, 3, size=(N))
        y = np.random.normal(0, 3, size=(N))
        theta = np.random.normal(np.pi/4, np.pi/8, size=(N))
        r = r_dist.rvs(size=(N))
        B = B_dist.rvs(size=(N))

        internalState = [x, y, theta, r, B]

    return internalState, studentNames, estimatorType

