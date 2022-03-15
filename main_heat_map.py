"""
This code creates the heat map, as presented in the article.
The heat map indicates the value of the Hill coefficient in which a saddle node is taking place depending on the parameter.
It also consider the parameter projection into [0,3]x[0,3] thanks to the DSGRN region definition
"""

from hill_model import *
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import numpy as np
import matplotlib.pyplot as plt
from models.TS_model import ToggleSwitch
import sys
from create_dataset import *

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# use dataset creation
# size of the sample
n_sample = 10 ** 4 # testing on 3, final run on 4
file_name = 'TS_data_100000.npz'
try:
    np.load(file_name)
except FileNotFoundError:
    n = 100000
    create_dataset_ToggleSwitch(100000, file_name)

file_storing = 'heat_map.npz'

data_subsample, region_subsample, coefs = subsample(file_name, n_sample)
a = np.transpose(data_subsample)
u, v = parameter_to_DSGRN_coord(a)


# test for nice heat map pictures
alpha1, beta1, alpha2, beta2 = parameter_to_alpha_beta(a)
alpha1 = np.sort(alpha1)
alpha2 = np.sort(alpha2)
ninety_percentile = int(np.ceil(len(alpha1)*0.9))
alphaMax = np.array([alpha1[ninety_percentile], alpha2[ninety_percentile]])

plt.figure()
dsgrn_plot(a, alphaMax)
plt.title('all parameters')
plt.savefig('all_parameters.pdf')

parameter_full = np.empty(shape=[0, 5])
lowest_hill = np.empty(0)
bad_parameters = np.empty(shape=[0, 5])
bad_candidates = []
boring_parameters = np.empty(shape=[0, 5])
multiple_saddles = np.empty(shape=[0, 5])
for j in range(n_sample):  # range(n_sample):
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, [1, 5, 10, 40, 80], a_j)
    if SNParameters and SNParameters != 0:
        for k in range(len(SNParameters)):
            # print('Saddle detected')
            if k == 0:
                parameter_full = np.append(parameter_full, [a_j], axis=0)
                lowest_hill = np.append(lowest_hill, SNParameters[k])
            if k > 0:
                print('\nMore than one saddle detected!')
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates and badCandidates != 0:
        print('\nA bad parameter')
        bad_parameters = np.append(bad_parameters, [a_j], axis=0)
        bad_candidates.append(badCandidates)
    printing_statement = 'Completion: ' + str(j) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

    if SNParameters == 0 and badCandidates == 0:
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)

np.savez('heat_map_data',
         u=u, v=v, a=a, parameter_full=parameter_full, lowest_hill=lowest_hill, bad_parameters=bad_parameters,
         bad_candidates=bad_candidates, boring_parameters=boring_parameters, n_sample=n_sample,
         multiple_saddles=multiple_saddles)

data = np.load('heat_map_data.npz', allow_pickle=True)

bad_parameters = data.f.bad_parameters
boring_parameters = data.f.boring_parameters
n_sample = data.f.n_sample
parameter_full = data.f.parameter_full
lowest_hill = data.f.lowest_hill
multiple_saddles = data.f.multiple_saddles

print('\nNumber of bad candidates', len(bad_parameters), 'out of ', n_sample)
print('Number of boring candidates', len(boring_parameters), 'out of ', n_sample)
print('Number of saddles', len(parameter_full), 'out of ', n_sample)
print('Number of parameters with multiple saddles', len(multiple_saddles), 'out of ', n_sample)

"""if bad_candidates is not None:
    fig1 = plt.figure()
    dsgrn_plot(bad_candidates, 10)
    plt.title('bad candidates')

if boring_parameters is not None:
    fig1 = plt.figure()
    dsgrn_plot(boring_parameters, 10)
    plt.title('No saddle detected')
"""
"""
parameter_DSGRN = parameter_to_DSGRN_coord(parameter_full)
parameter_DSGRN = np.array([parameter_DSGRN[0], parameter_DSGRN[1]])
unique_DSGRN = np.unique(parameter_DSGRN.round(decimals=5), axis=1)
average_sol = np.empty(0)
average_sol_long = 0 * solutions
for j in unique_DSGRN.T:
    # work in progress
    index_solution_j = np.where(abs(parameter_DSGRN[0, :] - j[0]) < 5 * 10 ** -5)
    index_solution_loc = np.where(abs(parameter_DSGRN[1, :] - j[1]) < 5 * 10 ** -5)
    index_solution_j = np.intersect1d(index_solution_j, index_solution_loc)
    if len(index_solution_j) > 0:
        average_sol = np.append(average_sol, np.mean(solutions[index_solution_j]))
        average_sol_long[index_solution_j] = np.mean(solutions[index_solution_j])
    else:
        print('wrong')
"""

plt.figure()
dsgrn_heat_plot(parameter_full, np.minimum(lowest_hill, 100), alphaMax=alphaMax)
plt.title('dsgrn_heat_plot')
plt.savefig('dsgrn_heat_plot.pdf')

plt.figure()
dsgrn_contour_plot(parameter_full, lowest_hill, alphaMax=alphaMax)
plt.title('dsgrn_contour_plot')
plt.savefig('dsgrn_contour_plot.pdf')

plt.figure()
dsgrn_plot(parameter_full, alphaMax=alphaMax)
plt.title('dsgrn_plot')
plt.savefig('dsgrn_plot.pdf')

if len(multiple_saddles) > 0:
    plt.figure()
    dsgrn_plot(multiple_saddles)
    plt.title('multiple_saddles')
    plt.savefig('multiple_saddles.pdf')

if len(bad_parameters) > 0:
    plt.figure()
    dsgrn_plot(bad_parameters)
    plt.title('bad_parameters')
    plt.savefig('bad_parameters.pdf')



plt.figure()
dsgrn_plot(a, color='k')
dsgrn_plot(parameter_full, color='g')
if len(multiple_saddles) > 0:
    dsgrn_plot(multiple_saddles, color='b')
if len(bad_parameters) > 0:
    dsgrn_plot(bad_parameters, color='r')

print('It is the end!')
