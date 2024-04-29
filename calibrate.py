import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

experimentalRun = 0

print('Loading the data file #', experimentalRun)
experimentalData = np.genfromtxt ('data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')

final_x = experimentalData[-1,5]
final_y = experimentalData[-1,6]
final_ang   = np.mod(experimentalData[-1,7]+np.pi,2*np.pi)-np.pi

print('Final values: ')
print('   pos x =',final_x,'m')
print('   pos y =',final_y,'m')
print('   angle =',final_ang,'rad')

meas_x = experimentalData[:,3]
meas_y = experimentalData[:,4]
meas_x = meas_x[~np.isnan(meas_x)]
meas_y = meas_y[~np.isnan(meas_y)]

print('Measurement Mean: ')
print('   meas x =',np.mean(meas_x),'m')
print('   meas y =',np.mean(meas_y),'m')

def B_error(B):
    return (np.mean(meas_x)-final_x-B*np.cos(final_ang)/2)**2 + (np.mean(meas_y)-final_y-B*np.sin(final_ang)/2)**2

res = sp.optimize.minimize(lambda B: B_error(B), x0 = 0.8)
B = res.x[0]

print('Min B: ')
print('   B =',B,'m')

est_meas_x = final_x + 1/2*B*np.cos(final_ang)
est_meas_y = final_y + 1/2*B*np.sin(final_ang)

print('Estimated measurement origin: ')
print('   pos x =',est_meas_x,'m')
print('   pos y =',est_meas_y,'m')

R = np.array([[np.cos(final_ang), np.sin(final_ang)],
              [np.sin(final_ang), np.cos(final_ang)]])

meas_err = np.array([meas_x-est_meas_x,
                     meas_y-est_meas_y])

meas_noise = np.linalg.inv(R) @ meas_err

print('Variances: ')
print('   var w1 =',np.var(meas_noise[0,:]),'m')
print('   var w2 =',np.var(meas_noise[1,:]),'m')

print('Generating plots')

figTopView, axTopView = plt.subplots(1, 1)
axTopView.plot(experimentalData[:,3], experimentalData[:,4], 'rx', label='Meas')
axTopView.plot(experimentalData[:,5], experimentalData[:,6], 'k:.', label='true')
x_angles = [-5*np.cos(final_ang), 5*np.cos(final_ang)] + final_x
y_angles = [-5*np.sin(final_ang), 5*np.sin(final_ang)] + final_y
axTopView.plot(x_angles, y_angles, label='theta')
axTopView.legend()
axTopView.set_xlabel('x-position [m]')
axTopView.set_ylabel('y-position [m]')

plt.show()

