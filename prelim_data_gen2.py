import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

nx = 6
ny = 6
nz = 1
nP = 1

x_j = np.linspace(0, 20, nx)
y_j = np.linspace(0, 20, ny)
z_j = np.linspace(0, 0, nz)
P_j = np.linspace(100, 100, nP)

xx_j, yy_j = np.meshgrid(x_j, y_j)

FREQ = 1575.42
GT = 1.
GR = 1.
PT = 100.
C = 299792458.
WAVELENGTH = C/(FREQ*1000000.)

nx_sensor = 3.
ny_sensor = 3.
nz_sensor = 1.

x_sensor = np.linspace(0, 20, nx_sensor)
y_sensor = np.linspace(0, 20, ny_sensor)
z_sensor = np.linspace(50, 50, nz_sensor)

xx, yy, zz = np.meshgrid(x_sensor, y_sensor, z_sensor)

xx = xx.reshape((np.prod(xx.shape),))
yy = yy.reshape((np.prod(yy.shape),))
zz = zz.reshape((np.prod(zz.shape),))

sensor_coords = zip(xx, yy, zz)

class TrainingPoint(object):
    def __init__(self, jammer_coords, PT, PR, quadrant):
        self.jammer_coords = jammer_coords
        self.PT = PT
        self.PR = PR
        self.quadrant = quadrant

def determine_quadrant(x_pos,y_pos):
    if (0 <= x <= 10 and 0 <= y <= 10):
        temp_quadrant = 0
    if (10 < x <= 20 and 0 <= y <= 10):
        temp_quadrant = 1
    if (0 <= x <= 10 and 10 < y <= 20):
        temp_quadrant = 2
    if (10 < x <= 20 and 10 < y <= 20):
        temp_quadrant = 3
    return temp_quadrant

PR = dict([])
jam_test = {'data':[], 'target':[], 'jam_coords':[], 'sensor_coords':sensor_coords}

for x in x_j:
    for y in y_j:
        for z in z_j:
            for P in P_j:
                temp_R = [distance.euclidean((x,y,z),sensor) for sensor in sensor_coords]
                temp_PR = [P*GT*GR*WAVELENGTH**2/(4*np.pi * R)**2 for R in temp_R]
                temp_quadrant = determine_quadrant(x,y)
                jam_test['data'].append(temp_PR)
                jam_test['target'].append(temp_quadrant)
                jam_test['jam_coords'].append([x,y])

jam_test['data'] = np.array(jam_test['data'])
jam_test['target'] = np.array(jam_test['target'])
jam_test['jam_coords'] = np.array(jam_test['jam_coords'])
jam_test['sensor_coords'] = np.array(jam_test['sensor_coords'])

np.save('test_data.npy', jam_test)

plt.scatter(xx_j,yy_j)
plt.plot([10, 10], [0, 20], 'k-', lw=2)
plt.plot([0, 20], [10, 10], 'k-', lw=2)
plt.show()