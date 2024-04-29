import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

# Estimate state space using bicycle dynamics
def q(ps, gamma, dt, x=np.zeros(5), v=np.zeros(3), xi=None):
    if xi is not None:
        x = xi[:5]
        v = xi[5:8]
        V = x[3]*(ps+v[0])*5
        return xi + dt*np.array([V*np.cos(x[2]+v[1]),
                                V*np.sin(x[2]+v[1]),
                                V/x[4]*np.tan(gamma+v[2]),
                                0, 0, 0, 0, 0, 0, 0])
    V = x[3]*(ps+v[0])*5
    return x + dt*np.array([V*np.cos(x[2]+v[1]),
                            V*np.sin(x[2]+v[1]),
                            V/x[4]*np.tan(gamma+v[2]),
                            0,
                            0])

# Estimate measurement using measurement model
def p(x=np.zeros(5), w=np.zeros(2), xi=None):
    if xi is not None:
        x = xi[:5]
        w = xi[8:]
    return np.array([x[0]+0.5*x[4]*np.cos(x[2])+w[0]*np.cos(x[2])+w[1]*np.sin(x[2]),
                     x[1]+0.5*x[4]*np.sin(x[2])+w[0]*np.sin(x[2])+w[1]*np.cos(x[2])])

# Jacobian Matrix A for EKF
def A(ps, gamma, dt, x):
    V = x[3]*ps*5
    dV = ps*5
    return np.eye(5) + dt*np.array([[0, 0, -V*np.sin(x[2]),       dV*np.sin(x[2]),                        0],
                                    [0, 0,  V*np.cos(x[2]),       dV*np.sin(x[2]),                        0],
                                    [0, 0,               0, dV/x[4]*np.tan(gamma), -V/x[4]**2*np.tan(gamma)],
                                    [0, 0,               0,                     0,                        0],
                                    [0, 0,               0,                     0,                        0]])

# Jacobian Matrix L for EKF
def L(ps, gamma, dt, x):
    V = x[3]*ps*5
    dV = x[3]*5
    return dt*np.array([[      dV*np.cos(x[2]), -V*np.sin(x[2]),                       0],
                        [      dV*np.sin(x[2]),  V*np.cos(x[2]),                       0],
                        [dV/x[4]*np.tan(gamma),               0, V/x[4]/np.cos(gamma)**2],
                        [                    0,               0,                       0],
                        [                    0,               0,                       0]])

# Jacobian Matrix H for EKF
def H(x):
    return np.array([[1, 0, -0.5*x[4]*np.sin(x[2]), 0, 0.5*np.cos(x[2])],
                        [0, 1,  0.5*x[4]*np.cos(x[2]), 0, 0.5*np.sin(x[2])]])

# Jacobian Matrix M for EKF
def M(x):
    return np.array([[np.cos(x[2]), np.sin(x[2])],
                     [np.sin(x[2]), np.cos(x[2])]])

# Update particle weights for PF
def pf_weights(z, x, N):
    z_dist = np.linalg.norm(z - p(x), axis=1)
    std = 0.3
    weights = sp.stats.norm(z_dist, std).pdf(z_dist)
    return weights

# Run the estimator 
def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement, estimatorType):
    """
    EKF - Extended Kalman Filter Code
    """
    if estimatorType == "EKF":
        # Pull relevant info from internal state
        x, P, var_v, var_w = internalStateIn
        
        # Prior Update
        A_k = A(pedalSpeed, steeringAngle, dt, x)
        L_k = L(pedalSpeed, steeringAngle, dt, x)
   
        x = q(pedalSpeed, steeringAngle, dt, x)
        P = A_k @ P @ A_k.T + L_k @ var_v @ L_k.T

        # Measurement Update
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # have a valid measurement
            z = np.array([measurement[0],
                          measurement[1]])
            
            H_k = H(x)
            M_k = M(x)

            K = P @ H_k.T @ np.linalg.inv(H_k @ P @ H_k.T + M_k @ var_w @ M_k.T)
            x = x + K @ (z - p(x))
            P = (np.eye(5) - K @ H_k) @ P

        internalStateOut = [x, P, var_v, var_w]
        x, y, theta, _, _ = x
    
    elif estimatorType == "UKF":
        # Pull relevant info from internal state
        x, P, var_v, var_w, N = internalStateIn
        
        # Create Sigma Points
        xi = np.concatenate((x, np.zeros(5)))
        dxi = np.zeros((N,N))
        dxi[:5,:5] = sp.linalg.sqrtm(N*P)
        dxi[5:8,5:8] = sp.linalg.sqrtm(N*var_v)
        dxi[8:,8:] = sp.linalg.sqrtm(N*var_w)

        xi_s = []
        for i in range(N):
            xi_s.append(q(pedalSpeed, steeringAngle, dt, xi=xi+dxi[:,i]))
            xi_s.append(q(pedalSpeed, steeringAngle, dt, xi=xi-dxi[:,i]))

        xi = np.mean(xi_s, axis=0)
        x = xi[:5]
        P = np.zeros((5,5))
        for i in range(2*N):
            P = P + np.outer((xi_s[i][:5] - x), (xi_s[i][:5] - x)) / (2*N)

        # Measurement Update
        # Check for a valid measurement
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):  
            z = np.array([measurement[0],
                          measurement[1]])
            
            zs = []
            for s in xi_s:
                zs.append(p(xi=s))

            zh = np.mean(zs, axis=0)

            Pxz = np.zeros((5, 2))
            Pzz = np.zeros((2, 2))
            for i in range(2*N):
                Pzz = Pzz + np.outer((zs[i] - z), (zs[i] - zh)) / (2*N)
                Pxz = Pxz + np.outer((xi_s[i][:5] - x), (zs[i] - zh)) / (2*N)
        
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

        x_full = np.array([q(pedalSpeed, steeringAngle, dt, x_full[:,i], v[:,i]) for i in range(N)]).T
    
        # Measurement Update
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # have a valid measurement
            z = np.array([[measurement[0]],
                          [measurement[1]]])

            weights = pf_weights(z, x_full, N)
            weights = weights / np.sum(weights)
            cdf = np.cumsum(weights)

            ind = np.argwhere(cdf>np.random.uniform())[0, 0]
            x_full = np.array([x_full[:, ind] for i in range(N)]).T
            
            K = 0.7
            E = np.max(x_full, axis=1) - np.min(x_full, axis=1)
            sigma = K*E*N**(-1/5)
            
            for i in range(5):
                x_full[i] += np.random.normal(0, sigma[i], size=(N))
        
        x, y, theta, r, B = x_full

        internalStateOut = [x, y, theta, r, B, N]
        
        x = np.mean(x)
        y = np.mean(y)
        theta = np.mean(theta)

    else:
        pass

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


