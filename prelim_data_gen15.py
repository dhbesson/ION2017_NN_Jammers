import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools
import sys
import math
import random
import pandas

# todo combine data gen and neural network into one script

size_training_set = 3000.
nx_sensor = 5.
ny_sensor = 5.
nz_sensor = 1.

nx_quadrants = 20.
ny_quadrants = 20.
number_training_jammers = int(nx_quadrants*ny_quadrants)
x_min_quadrant = 0.
x_max_quadrant = 10000.
y_min_quadrant = 0.
y_max_quadrant = 10000.
x_length_quadrant = (x_max_quadrant - x_min_quadrant)/nx_quadrants
y_length_quadrant = (y_max_quadrant - y_min_quadrant)/ny_quadrants
quad_labels = range(0,int(nx_quadrants*ny_quadrants))
quad_labels = np.array(quad_labels)
quad_labels = np.reshape(quad_labels, (nx_quadrants,ny_quadrants))

# np.random.seed(42)
nx = nx_quadrants
ny = ny_quadrants
nz = 1
nP = 1
num_quads = int(nx_quadrants*ny_quadrants)
max_combo = 10
min_combo = 1
max_jammers = nx*ny

# quad_combos = []
# for L in range(min_combo, max_combo+1):
#     for subset in itertools.combinations(range(0,num_quads),L):
#         quad_combos.append(list(subset))
#     print("Quad Combo Level Complete: %d" % L)
#     print(len(quad_combos))

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

combo_1 = max_jammers
combo_2 = nCr(max_jammers,2)

x_j = np.linspace((1/(2*nx))*x_max_quadrant, ((2*nx-1)/(2*nx))*x_max_quadrant, nx)
y_j = np.linspace((1/(2*ny))*y_max_quadrant, ((2*ny-1)/(2*ny))*y_max_quadrant, ny)
z_j = np.linspace(0, 0, nz)
P_j = np.linspace(100, 100, nP)

xx_j, yy_j, zz_j = np.meshgrid(x_j, y_j, z_j)

temp_xx_j = xx_j.reshape((np.prod(xx_j.shape),))
temp_yy_j = yy_j.reshape((np.prod(yy_j.shape),))
temp_zz_j = zz_j.reshape((np.prod(zz_j.shape),))


temp_jam_coords = (zip(temp_xx_j, temp_yy_j, temp_zz_j))

jammer_combos = []

np.random.seed(1)
new_jammer_choices = np.random.choice(int(max_jammers), number_training_jammers, replace=False)
np.random.seed(2)
test_jammer_choices = np.random.choice(int(max_jammers), number_training_jammers, replace=False)
new_temp_jam_coords = [temp_jam_coords[i] for i in new_jammer_choices]
test_temp_jam_coords = [temp_jam_coords[i] for i in test_jammer_choices]

jammer_combos = []
test_jammer_combos = []

for L in range(min_combo, max_combo+1):
    if L == 1:
        for subset in itertools.combinations(new_temp_jam_coords,L):
            jammer_combos.append(subset)
        for subset in itertools.combinations(test_temp_jam_coords,L):
            test_jammer_combos.append(subset)
        print("Jammer Combo Level Complete: %d" % L)
    else:
        np.random.seed(L+1)
        random_subset_choices = [random.sample(new_temp_jam_coords, L) for i in range(0,int(size_training_set))]
        [jammer_combos.append(subset) for subset in random_subset_choices]
        np.random.seed(L+10)
        test_random_subset_choices = [random.sample(new_temp_jam_coords, L) for i in range(0,int(size_training_set))]
        [test_jammer_combos.append(subset) for subset in test_random_subset_choices]
        print("Jammer Combo Level Complete: %d" % L)


max_jammer_combo = len(jammer_combos)

FREQ = 1575.42
GT = 1.
GR = 1.
PT = 100.
C = 299792458.
WAVELENGTH = C/(FREQ*1000000.)

x_sensor = np.linspace(0, x_max_quadrant, nx_sensor)
y_sensor = np.linspace(0, y_max_quadrant, ny_sensor)
z_sensor = np.linspace(121.92, 121.92, nz_sensor)

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
jam_test = {'data':[], 'target':[], 'jam_coords':new_temp_jam_coords,'jam_combo_coords':[], 'sensor_coords':sensor_coords}
final_test = {'data':[], 'target':[], 'jam_coords':test_temp_jam_coords,'jam_combo_coords':[], 'sensor_coords':sensor_coords}
macro_params = {'size_training_set':size_training_set, 'nx_sensor':nx_sensor, 'ny_sensor':ny_sensor,
                'nz_sensor':nz_sensor, 'number_training_jammers':number_training_jammers, 'nx_quadrants':nx_quadrants,
                'ny_quadrants':ny_quadrants,'x_min_quadrant':x_min_quadrant,'x_max_quadrant':x_max_quadrant,
                'y_min_quadrant':y_min_quadrant,'y_max_quadrant':y_max_quadrant,'x_length_quadrant':x_length_quadrant,
                'y_length_quadrant':y_length_quadrant,'quad_labels':quad_labels}



num_combo = len(jammer_combos)
temp_PR_matrix = {}
temp_quad_matrix = {}
test_temp_PR_matrix = {}
test_temp_quad_matrix = {}

for i, jammer in enumerate(new_temp_jam_coords):
    temp_PR_matrix[i] = []
    for j, sensor in enumerate(sensor_coords):
        x = jammer[0]
        y = jammer[1]
        z = z_j
        temp_R = distance.euclidean((x, y, z_j), sensor)
        temp_PR_matrix[i].append(PT*GT*GR*WAVELENGTH**2/((4*np.pi * temp_R)**2))
        temp_quad_matrix[i] = determine_quadrant(x,y)

for i, jammer in enumerate(test_temp_jam_coords):
    test_temp_PR_matrix[i] = []
    for j, sensor in enumerate(sensor_coords):
        x = jammer[0]
        y = jammer[1]
        z = z_j
        temp_R = distance.euclidean((x, y, z_j), sensor)
        test_temp_PR_matrix[i].append(PT*GT*GR*WAVELENGTH**2/((4*np.pi * temp_R)**2))
        test_temp_quad_matrix[i] = determine_quadrant(x,y)

# np.random.seed(3)
# print(combo_1,combo_2,num_combo)
# new_jammer_combos_choices = np.random.choice(xrange(int(combo_1+combo_2),int(num_combo)), size_training_set, replace=False)
# new_jammer_combos_choices = np.append(new_jammer_combos_choices,xrange(0,int(combo_1+combo_2)))
# np.random.seed(4)
# test_jammer_combos_choices = np.random.choice(xrange(int(combo_1+combo_2),int(num_combo)), size_training_set, replace=False)
# test_jammer_combos_choices = np.append(test_jammer_combos_choices,xrange(0,int(combo_1+combo_2)))
# jammer_combos = [jammer_combos[i] for i in new_jammer_combos_choices]
# test_jammer_combos = [test_jammer_combos[i] for i in test_jammer_combos_choices]

total = int(len(jammer_combos))
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
    jam_test['jam_combo_coords'].append(combo)
    if (i % (5 * point) == 0):
        sys.stdout.write("\r[" + "=" * (i / increment) + " " * ((total - i) / increment) + "]" + str(i / point) + "%")
        sys.stdout.flush()

for i, combo in enumerate(test_jammer_combos):
    test_temp_PR_list = [0]*len(sensor_coords)
    test_temp_quad_list = np.zeros(num_quads)
    for jammer in combo:
        i_j = test_temp_jam_coords.index(jammer)
        x = jammer[0]
        y = jammer[1]
        z = z_j
        test_temp_PR = test_temp_PR_matrix[i_j]
        test_temp_PR_list = [a + b for a,b in zip(test_temp_PR_list,test_temp_PR)]
        test_temp_quadrant = test_temp_quad_matrix[i_j]
        test_temp_quad_list[test_temp_quadrant] = 1
    final_test['data'].append(test_temp_PR_list)
    final_test['target'].append(test_temp_quad_list)
    final_test['jam_combo_coords'].append(combo)
    if (i % (5 * point) == 0):
        sys.stdout.write("\r[" + "=" * (i / increment) + " " * ((total - i) / increment) + "]" + str(i / point) + "%")
        sys.stdout.flush()

jam_test['data'] = np.array(jam_test['data'])
jam_test['target'] = np.array(jam_test['target'])
jam_test['jam_coords'] = np.array(jam_test['jam_coords'])
jam_test['jam_combo_coords'] = np.array(jam_test['jam_combo_coords'])
jam_test['sensor_coords'] = np.array(jam_test['sensor_coords'])

final_test['data'] = np.array(final_test['data'])
final_test['target'] = np.array(final_test['target'])
final_test['jam_coords'] = np.array(final_test['jam_coords'])
final_test['jam_combo_coords'] = np.array(final_test['jam_combo_coords'])
final_test['sensor_coords'] = np.array(final_test['sensor_coords'])

np.save('test_data_3k.npy', jam_test)
np.save('final_test_data_3k.npy', final_test)
np.save('macro_params_3k.npy', macro_params)

fig, ax = plt.subplots()

img = plt.imread('portland_100_km2.png')
ax.imshow(img, origin='upper', extent=[0,10000,0,10000])

# print(jam_test['target'])
[ax.scatter(i_jam[0],i_jam[1], s=10, color='r') for i_jam in new_temp_jam_coords]
# [ax.scatter(i_jam[0],i_jam[1], color='r',s=20) for i_jam in test_temp_jam_coords]

for x_line in np.arange(x_min_quadrant,x_max_quadrant,x_length_quadrant):
    for y_line in np.arange(y_min_quadrant,y_max_quadrant,y_length_quadrant):
        ax.plot([x_line, x_line], [y_line, y_line+y_length_quadrant], 'k-', lw=2)
        ax.plot([x_line, x_line+x_length_quadrant], [y_line, y_line], 'k-', lw=2)

ax.plot([x_max_quadrant, x_max_quadrant], [y_min_quadrant, y_max_quadrant], 'k-', lw=2)
ax.plot([x_min_quadrant, x_max_quadrant], [y_max_quadrant, y_max_quadrant], 'k-', lw=2)

ax.scatter(jam_test['sensor_coords'][:,0],jam_test['sensor_coords'][:,1], color='b', s=100)
ax.set_xlabel('Horizontal Position (m)')
ax.set_ylabel('Vertical Position (m)')
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