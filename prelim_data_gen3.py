import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools
import pandas

nx = 7
ny = 7
nz = 1
nP = 1
num_quads = 4
max_combo = 2
min_combo = 2

quad_combos = []
for L in range(min_combo, max_combo+1):
    for subset in itertools.combinations(range(0,num_quads),L):
        quad_combos.append(list(subset))
print(quad_combos)

x_j = np.linspace(0, 20, nx)
y_j = np.linspace(0, 20, ny)
z_j = np.linspace(0, 0, nz)
P_j = np.linspace(100, 100, nP)

xx_j, yy_j = np.meshgrid(x_j, y_j)

temp_xx_j = xx_j.reshape((np.prod(xx_j.shape),))
temp_yy_j = yy_j.reshape((np.prod(yy_j.shape),))

temp_jam_coords = zip(temp_xx_j, temp_yy_j)

jammer_combos = []

for L in range(min_combo, max_combo+1):
    for subset in itertools.combinations(temp_jam_coords,L):
        jammer_combos.append(subset)
max_jammer_combo = len(jammer_combos)

FREQ = 1575.42
GT = 1.
GR = 1.
PT = 100.
C = 299792458.
WAVELENGTH = C/(FREQ*1000000.)

nx_sensor = 20.
ny_sensor = 20.
nz_sensor = 1.

x_sensor = np.linspace(-5, 25, nx_sensor)
y_sensor = np.linspace(-5, 25, ny_sensor)
z_sensor = np.linspace(50, 50, nz_sensor)

xx, yy, zz = np.meshgrid(x_sensor, y_sensor, z_sensor)

xx = xx.reshape((np.prod(xx.shape),))
yy = yy.reshape((np.prod(yy.shape),))
zz = zz.reshape((np.prod(zz.shape),))

sensor_coords = zip(xx, yy, zz)

def determine_quadrant(x_pos,y_pos):
    if (0 <= x_pos <= 12 and 0 <= y_pos <= 12):
        temp_quadrant = 0
    if (12 < x_pos <= 22 and 0 <= y_pos <= 12):
        temp_quadrant = 1
    if (0 <= x_pos <= 12 and 12 < y_pos <= 22):
        temp_quadrant = 2
    if (12 < x_pos <= 22 and 12 < y_pos <= 22):
        temp_quadrant = 3
    return temp_quadrant

PR = dict([])
jam_test = {'data':[], 'target':[], 'jam_coords':[], 'sensor_coords':sensor_coords}


for combo in jammer_combos:
    temp_PR_list = [0]*len(sensor_coords)
    temp_quad_list = []
    for jammer in combo:
        x = jammer[0]
        y = jammer[1]
        z = z_j
        temp_R = [distance.euclidean((x,y,z_j),sensor) for sensor in sensor_coords]
        temp_PR = [PT*GT*GR*WAVELENGTH**2/(4*np.pi * R)**2 for R in temp_R]
        temp_PR_list = [a + b for a,b in zip(temp_PR_list,temp_PR)]
        temp_quadrant = determine_quadrant(x,y)
        temp_quad_list.append(temp_quadrant)
    if len(temp_quad_list) == len(set(temp_quad_list)):
        temp_quad_list = sorted(set(temp_quad_list))
        target_quad_combo = quad_combos.index(temp_quad_list)
        jam_test['data'].append(temp_PR)
        jam_test['target'].append(target_quad_combo)
        jam_test['jam_coords'].append(combo)


jam_test['data'] = np.array(jam_test['data'])
jam_test['target'] = np.array(jam_test['target'])
jam_test['jam_coords'] = np.array(jam_test['jam_coords'])
jam_test['sensor_coords'] = np.array(jam_test['sensor_coords'])

np.save('test_data.npy', jam_test)
print(jam_test['target'])
plt.scatter(xx_j,yy_j)
plt.scatter(jam_test['sensor_coords'][:,0],jam_test['sensor_coords'][:,1], color='r', marker='x')
plt.plot([12, 12], [0, 22], 'k-', lw=2)
plt.plot([0, 22], [12, 12], 'k-', lw=2)

plt.show()