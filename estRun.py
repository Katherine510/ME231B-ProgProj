import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def q(x, ps, gamma, dt, v=np.array([0, 0, 0])):
    V = x[3]*(ps+v[0])/5
    x_new = V*np.cos(x[2]+v[1])
    y_new = V*np.sin(x[2]+v[1])
    theta_new = V/x[4]*np.tan(gamma+v[2])
    rB_new = np.zeros((1, 10**3))
    
    out = np.vstack((x_new, y_new, theta_new, rB_new, rB_new))
    
    return x + dt*out

def p(x):
    return np.array([x[0]+0.5*x[4]*np.cos(x[2]),
                        x[1]+0.5*x[4]*np.sin(x[2])])

def A(x, ps, gamma, dt):
    V = x[3]*ps/5
    dV = ps/5
    return np.eye(5) + dt*np.array([[0, 0, -V*np.sin(x[2]),       dV*np.sin(x[2]),                        0],
                                    [0, 0,  V*np.cos(x[2]),       dV*np.sin(x[2]),                        0],
                                    [0, 0,               0, dV/x[4]*np.tan(gamma), -V/x[4]**2*np.tan(gamma)],
                                    [0, 0,               0,                     0,                        0],
                                    [0, 0,               0,                     0,                        0]])

def L(x, ps, gamma, dt):
    V = x[3]*ps/5
    dV = x[3]/5
    return dt*np.array([[      dV*np.cos(x[2]), -V*np.sin(x[2]),                       0],
                        [      dV*np.sin(x[2]),  V*np.cos(x[2]),                       0],
                        [dV/x[4]*np.tan(gamma),               0, V/x[4]/np.cos(gamma)**2],
                        [                    0,               0,                       0],
                        [                    0,               0,                       0]])

def H(x):
    return np.array([[1, 0, -0.5*x[4]*np.sin(x[2]), 0, 0.5*np.cos(x[2])],
                        [0, 1,  0.5*x[4]*np.cos(x[2]), 0, 0.5*np.sin(x[2])]])

def fz_x(z, x):
    z_diff = z - p(x)
    std = np.sqrt(20)
    P = sp.stats.norm.cdf(z_diff/std)
    return P[0]*P[1]

def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement, estimatorType):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  x: your current best estimate for the bicycle's x-position
    #  y: your current best estimate for the bicycle's y-position
    #  theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:
    # x = internalStateIn[0]
    # y = internalStateIn[1]
    # theta = internalStateIn[2]
    # r = internalStateIn[3]
    # B = internalStateIn[4]
    # gamma = steeringAngle

    # r = np.random.normal(0.425, )
    # v = pedalSpeed * 1/2*np.pi * 2*np.pi*r
    
    # x_dot = v*np.cos(theta)
    # y_dot = v*np.sin(theta)
    # theta_dot = v/B * np.tan(gamma)
    
    # p = np.array([
    #     [x + 0.5 * B * np.cos(theta)],
    #     [y + 0.5 * B * np.sin(theta)]
    # ])
    
    """
    EKF - Extended Kalman Filter Code
    """
    if estimatorType == "EKF":
        x, P, var_v, var_w = internalStateIn
        
        # Prior Update
        A_k = A(x, pedalSpeed, steeringAngle, dt)
        L_k = L(x, pedalSpeed, steeringAngle, dt)
   
        x = q(x, pedalSpeed, steeringAngle, dt)
        P = A_k @ P @ A_k.T + L_k @ var_v @ L_k.T

        # Measurement Update
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # have a valid measurement
            z = np.array([measurement[0],
                          measurement[1]])
            
            H_k = H(x)

            K = P @ H_k.T @ np.linalg.inv(H_k @ P @ H_k.T + var_w)
            x = x + K @ (z - p(x))
            P = (np.eye(5) - K @ H_k) @ P

        internalStateOut = [x, P, var_v, var_w]
        x, y, theta, _, _ = x
    
    elif estimatorType == "UKF":
        x, P, var_v, var_w, N = internalStateIn
        
        xs = []
        dx = sp.linalg.sqrtm(N*P)
        for i in range(N):
            xs.append(q(x + dx[:,i], pedalSpeed, steeringAngle, dt))
            xs.append(q(x - dx[:,i], pedalSpeed, steeringAngle, dt))

        x = np.mean(xs, axis=0)
        P = np.zeros((N, N))
        for i in range(2*N):
            P = P + np.outer((xs[i] - x), (xs[i] - x)) / (2*N)
        P = P + + var_v

        # Measurement Update
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # have a valid measurement
            z = np.array([measurement[0],
                          measurement[1]])
            
            zs = []
            for s in xs:
                zs.append(p(s))

            zh = np.mean(zs, axis=0)

            Pxz = np.zeros((N, 2))
            Pzz = np.zeros((2, 2))
            for i in range(2*N):
                Pzz = Pzz + np.outer((zs[i] - z), (zs[i] - zh)) / (2*N)
                Pxz = Pxz + np.outer((xs[i] - x), (zs[i] - zh)) / (2*N)
            Pzz = Pzz + var_w
        
            K = Pxz @ np.linalg.inv(Pzz)
            x = x + K @ (z - zh)
            P = P - K @ Pzz @ K.T

        internalStateOut = [x, P, var_v, var_w, N]
        x, y, theta, _, _ = x
    
    elif estimatorType == "PF":
        x, y, theta, r, B, N = internalStateIn

        x_full = np.vstack((x, y, theta, r, B))
        
        # Prediction Step
        v_ps = np.random.normal(0, 0.1, size=((1, N)))
        v_theta = np.random.normal(0, np.pi/128, size=((1, N)))
        v_gamma = np.random.normal(0, np.pi/128, size=((1, N)))
        
        v = np.vstack((v_ps, v_theta, v_gamma))

        x_full = q(x_full, pedalSpeed, steeringAngle, dt, v).reshape((5, N))

        # Measurement Update
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # have a valid measurement
            z = np.array([measurement[0],
                          measurement[1]])

            B = np.array([fz_x(z, x_full[:, i]) for i in range(N)])
            B = B / np.sum(B)
            cdf = np.cumsum(B)

            # Resample
            ind = np.argwhere(cdf>np.random.uniform())
            print(ind)
            x_full = np.array([x_full[:, ind[0,0]] for i in range(N)]).T

        x, y, theta, r, B = x_full

        internalStateOut = [x, y, theta, r, B, N]
        
        x = np.mean(x)
        y = np.mean(y)
        theta = np.mean(theta)

    else:
        pass


    """
    #we're unreliable about our favourite colour: 
    if myColor == 'green':
        myColor = 'red'
    else:
        myColor = 'green'
    """


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    # internalStateOut = [x,
    #                  y,
    #                  theta
    #                  ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


