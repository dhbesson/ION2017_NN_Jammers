import numpy as np
from scipy.spatial import distance
from operator import add
import matplotlib.pyplot as plt
import itertools
import sys
import math
import random
import pandas

# todo combine data gen and neural network into one script

size_training_set = 1000
nx_sensor = 5.
ny_sensor = 5.
nz_sensor = 1.
nx_quadrants = 10.
ny_quadrants = 10.
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
max_combo = 5
min_combo = 1
max_jammers = int(nx*ny)

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

np.random.seed(69)
temp_xx_j = xx_j.reshape((np.prod(xx_j.shape),))
print(np.size(np.random.rand(np.size(temp_xx_j))))
temp_xx_j = map(add,temp_xx_j, -(1/(2*nx))*x_max_quadrant + (1/(nx))*x_max_quadrant*np.random.rand(np.size(temp_xx_j)))
print(temp_xx_j[0])
temp_yy_j = yy_j.reshape((np.prod(yy_j.shape),))
temp_yy_j = map(add,temp_yy_j, -(1/(2*ny))*y_max_quadrant + (1/(ny))*y_max_quadrant*np.random.rand(np.size(temp_yy_j)))
temp_zz_j = zz_j.reshape((np.prod(zz_j.shape),))


temp_jam_coords = (zip(temp_xx_j, temp_yy_j, temp_zz_j))

jammer_combos = []


new_jammer_choices = np.random.choice(max_jammers, number_training_jammers, replace=False)

new_temp_jam_coords = temp_jam_coords

# chosen_jammers = [1,5,20,35,62,80,89,95]
# chosen_jammers = [22,210,350,38]
# np.random.seed(50)
# chosen_jammers = np.random.choice(number_training_jammers, 3, replace=False)
# print(chosen_jammers)

# jammer_combos = [[new_temp_jam_coords[x_chosen] for x_chosen in chosen_jammers]]

for L in range(min_combo, max_combo+1):
    if L <= 2:
        for subset in itertools.combinations(new_temp_jam_coords,L):
            jammer_combos.append(subset)
        print("Jammer Combo Level Complete: %d" % L)
    else:
        np.random.seed(L+1)
        random_subset_choices = [random.sample(new_temp_jam_coords, L) for i in range(0,int(size_training_set))]
        [jammer_combos.append(subset) for subset in random_subset_choices]
        print("Jammer Combo Level Complete: %d" % L)
print(jammer_combos)

# for L in range(min_combo, max_combo+1):
#     for subset in itertools.combinations(new_temp_jam_coords,L):
#         jammer_combos.append(subset)
#     print("Jammer Combo Level Complete: %d" % L)

max_jammer_combo = len(jammer_combos)

FREQ = 1575.42
GT = 1.
GR = 1.
PT = 100.
C = 299792458.
WAVELENGTH = C/(FREQ*1000000.)

x_sensor = np.linspace(0, x_max_quadrant, nx_sensor)
y_sensor = np.linspace(0, y_max_quadrant, ny_sensor)
z_sensor = np.linspace(200, 200, nz_sensor)

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
jam_test = {'data':[], 'target':[], 'jam_coords':temp_jam_coords,'jam_combo_coords':[], 'sensor_coords':sensor_coords}

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


# np.random.seed(3)
# new_jammer_combos_choices = np.random.choice(xrange(combo_1+combo_2,num_combo), size_training_set, replace=False)
# new_jammer_combos_choices = np.append(new_jammer_combos_choices,xrange(0,combo_1+combo_2))

# jammer_combos = [jammer_combos[i] for i in new_jammer_combos_choices]


# total = int(size_training_set+combo_1+combo_2)
# point = total / 100
# increment = total / 20

for i, combo in enumerate(jammer_combos):
    temp_PR_list = [0]*len(sensor_coords)
    temp_quad_list = np.zeros(num_quads)
    print combo
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
    # if (i % (5 * point) == 0):
    #     sys.stdout.write("\r[" + "=" * (i / increment) + " " * ((total - i) / increment) + "]" + str(i / point) + "%")
    #     sys.stdout.flush()

jam_test['data'] = np.array(jam_test['data'])
jam_test['target'] = np.array(jam_test['target'])
jam_test['jam_coords'] = np.array(jam_test['jam_coords'])
jam_test['jam_combo_coords'] = np.array(jam_test['jam_combo_coords'])
jam_test['sensor_coords'] = np.array(jam_test['sensor_coords'])

np.save('test_data_chosen_messy_1k.npy', jam_test)
np.save('macro_params_chosen_messy_1k.npy', macro_params)

fig, ax = plt.subplots()

# img = plt.imread('lax.png')
# ax.imshow(img, origin='lower', extent=[0,7000,0,4750])

# print(jam_test['target'])
[ax.scatter(i_jam[0],i_jam[1], s=50, color='r') for i_jam in new_temp_jam_coords]

for x_line in np.arange(x_min_quadrant,x_max_quadrant,x_length_quadrant):
    for y_line in np.arange(y_min_quadrant,y_max_quadrant,y_length_quadrant):
        ax.plot([x_line, x_line], [y_line, y_line+y_length_quadrant], 'k-', lw=2)
        ax.plot([x_line, x_line+x_length_quadrant], [y_line, y_line], 'k-', lw=2)

ax.plot([x_max_quadrant, x_max_quadrant], [y_min_quadrant, y_max_quadrant], 'k-', lw=2)
ax.plot([x_min_quadrant, x_max_quadrant], [y_max_quadrant, y_max_quadrant], 'k-', lw=2)

ax.scatter(jam_test['sensor_coords'][:,0],jam_test['sensor_coords'][:,1], color='b', s=100)
plt.show()