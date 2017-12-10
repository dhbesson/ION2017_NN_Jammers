import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools
import sys
import random
import pandas

size_training_set = 1000
number_training_jammers = 50
nx_quadrants = 10.
ny_quadrants = 10.
x_min_quadrant = 0.
x_max_quadrant = 7000.
y_min_quadrant = 0.
y_max_quadrant = 4750.
x_length_quadrant = (x_max_quadrant - x_min_quadrant)/nx_quadrants
y_length_quadrant = (y_max_quadrant - y_min_quadrant)/ny_quadrants
quad_labels = range(0,int(nx_quadrants*ny_quadrants))
quad_labels = np.array(quad_labels)
quad_labels = np.reshape(quad_labels, (nx_quadrants,ny_quadrants))

np.random.seed(42)
nx = 20
ny = 20
nz = 1
nP = 1
num_quads = int(nx_quadrants*ny_quadrants)
max_combo = 4
min_combo = 1
max_jammers = nx*ny

quad_combos = []
for L in range(min_combo, max_combo+1):
    for subset in itertools.combinations(range(0,num_quads),L):
        quad_combos.append(list(subset))

x_j = np.linspace(1, 7000, nx)
y_j = np.linspace(1, 4750, ny)
z_j = np.linspace(0, 0, nz)
P_j = np.linspace(100, 100, nP)

xx_j, yy_j, zz_j = np.meshgrid(x_j, y_j, z_j)

temp_xx_j = xx_j.reshape((np.prod(xx_j.shape),))
temp_yy_j = yy_j.reshape((np.prod(yy_j.shape),))
temp_zz_j = zz_j.reshape((np.prod(zz_j.shape),))


temp_jam_coords = (zip(temp_xx_j, temp_yy_j, temp_zz_j))
print(len(temp_jam_coords))

jammer_combos = []

new_jammer_choices = np.random.choice(max_jammers, number_training_jammers, replace=False)
new_temp_jam_coords = [temp_jam_coords[i] for i in new_jammer_choices]
print(new_temp_jam_coords)
quad_combos = []

for L in range(min_combo, max_combo+1):
    for subset in itertools.combinations(new_temp_jam_coords,L):
        jammer_combos.append(subset)

max_jammer_combo = len(jammer_combos)

FREQ = 1575.42
GT = 1.
GR = 1.
PT = 100.
C = 299792458.
WAVELENGTH = C/(FREQ*1000000.)

nx_sensor = 10.
ny_sensor = 10.
nz_sensor = 1.

x_sensor = np.linspace(0, 7000, nx_sensor)
y_sensor = np.linspace(0, 4750, ny_sensor)
z_sensor = np.linspace(100, 100, nz_sensor)

xx, yy, zz = np.meshgrid(x_sensor, y_sensor, z_sensor)

xx = xx.reshape((np.prod(xx.shape),))
yy = yy.reshape((np.prod(yy.shape),))
zz = zz.reshape((np.prod(zz.shape),))

sensor_coords = zip(xx, yy, zz)

def determine_quadrant(x_pos,y_pos):

    x_coord_quadrant = int(x_pos/x_length_quadrant)
    y_coord_quadrant = int(y_pos/y_length_quadrant)
    if x_coord_quadrant == nx_quadrants:
        x_coord_quadrant = x_coord_quadrant - 1
    if y_coord_quadrant == ny_quadrants:
        y_coord_quadrant = y_coord_quadrant - 1
    temp_quadrant = quad_labels[x_coord_quadrant,y_coord_quadrant]
    return temp_quadrant

PR = dict([])
jam_test = {'data':[], 'target':[], 'jam_coords':[], 'sensor_coords':sensor_coords}

num_combo = len(jammer_combos)
temp_PR_matrix = {}
temp_quad_matrix = {}

for i, jammer in enumerate(new_temp_jam_coords):
    temp_PR_matrix[i] = []
    for j, sensor in enumerate(sensor_coords):
        x = jammer[0]
        y = jammer[1]
        z = z_j
        temp_R = distance.euclidean((x, y, z_j), sensor)
        temp_PR_matrix[i].append(PT*GT*GR*WAVELENGTH**2/((4*np.pi * temp_R)**2))
        temp_quad_matrix[i] = determine_quadrant(x,y)


new_jammer_combos_choices = np.random.choice(num_combo, size_training_set, replace=False)
jammer_combos = [jammer_combos[i] for i in new_jammer_combos_choices]

total = int(size_training_set)
point = total / 100
increment = total / 20

for i, combo in enumerate(jammer_combos):
    temp_PR_list = [0]*len(sensor_coords)
    temp_quad_list = np.zeros(num_quads)
    for jammer in combo:
        i_j = new_temp_jam_coords.index(jammer)
        x = jammer[0]
        y = jammer[1]
        z = z_j
        temp_PR = temp_PR_matrix[i_j]
        temp_PR_list = [a + b for a,b in zip(temp_PR_list,temp_PR)]
        temp_quadrant = temp_quad_matrix[i_j]
        temp_quad_list[temp_quadrant] = 1
    jam_test['data'].append(temp_PR_list)
    jam_test['target'].append(temp_quad_list)
    jam_test['jam_coords'].append(combo)
    if (i % (5 * point) == 0):
        sys.stdout.write("\r[" + "=" * (i / increment) + " " * ((total - i) / increment) + "]" + str(i / point) + "%")
        sys.stdout.flush()

jam_test['data'] = np.array(jam_test['data'])
jam_test['target'] = np.array(jam_test['target'])
jam_test['jam_coords'] = np.array(jam_test['jam_coords'])
jam_test['sensor_coords'] = np.array(jam_test['sensor_coords'])

np.save('test_data.npy', jam_test)

fig, ax = plt.subplots()

img = plt.imread('lax.png')
# ax.imshow(img, origin='lower', extent=[0,7000,0,4750])

# print(jam_test['target'])
[ax.scatter(i_jam[0],i_jam[1]) for i_jam in new_temp_jam_coords]
ax.scatter(jam_test['sensor_coords'][:,0],jam_test['sensor_coords'][:,1], color='r', marker='+')
for x_line in np.arange(x_min_quadrant,x_max_quadrant,x_length_quadrant):
    for y_line in np.arange(y_min_quadrant,y_max_quadrant,y_length_quadrant):
        ax.plot([x_line, x_line], [y_line, y_line+y_length_quadrant], 'k-', lw=2)
        ax.plot([x_line, x_line+x_length_quadrant], [y_line, y_line], 'k-', lw=2)

ax.plot([x_max_quadrant, x_max_quadrant], [y_min_quadrant, y_max_quadrant], 'k-', lw=2)
ax.plot([x_min_quadrant, x_max_quadrant], [y_max_quadrant, y_max_quadrant], 'k-', lw=2)

plt.show()

# plt.hist(jam_test['target'])
# plt.show()
#
# fig, axes = plt.subplots(2, 5)
# # use global min / max to ensure all weights are shown on the same scale
# flat_list = [item for sublist in jam_test['data'] for item in sublist]
# vmin, vmax = min(flat_list), max(flat_list)
# print vmin,vmax
# i_temp = 0
# for i_jammer, ax in zip(jam_test['data'], axes.ravel()):
#     jam1,jam2 = jam_test['jam_coords'][i_temp]
#     ax.matshow(i_jammer.reshape(nx_sensor, ny_sensor), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.scatter(1, 1)
#     ax.set_xticks(())
#     ax.set_yticks(())
#     i_temp= i_temp+1
#
# plt.show()