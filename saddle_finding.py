"""
Helper functions to find saddle nodes reliably

Main function: find_saddle_coef
Input:

"""

import numpy as np
import time
import matplotlib.pyplot as plt
from hill_model import *
from saddle_node import *
from models import ToggleSwitch
from scipy.interpolate import griddata


def count_eq(f, hill, p, gridDensity=10):
    """Count the number of equilibria found for a given parameter"""
    if is_vector(hill):
        countVector = np.zeros_like(hill)
        equilibria = []
        for j in range(len(countVector)):
            countVector[j], equilibria[j] = count_eq(hill[j], p)
        return countVector
    else:
        eq = f.find_equilibria(gridDensity, hill, p)
        if eq is not None:

            return np.shape(eq)[-1], eq  # number of rows is the number of equilibria found
        else:
            eq = f.find_equilibria(gridDensity * 2, hill, p)
            if eq is None:
                print(72)
            return np.shape(eq)[-1], eq


def estimate_saddle_node(f, hill, p, gridDensity=10):
    """Attempt to predict whether p admits any saddle-node points by counting equilibria at each value in the hill vector.
    If any values return multiple equilibria, attempt to bound the hill parameters for which these occur. Otherwise,
    return an empty interval."""

    hillIdx = 0
    # hill = ezcat(hill[0], hill)  # append 1 to the front of the hill vector
    # UPDATE: expect hill to already have at least two elements

    numEquilibria, Eq = count_eq(f, hill[0], p, gridDensity)
    numEquilibriaInf, Eqs = count_eq(f, hill[-1], p, gridDensity)

    hill_for_saddle = []
    equilibria_for_saddle = []

    if numEquilibriaInf > 1:
        n_steps = int(np.ceil((hill[-1] - hill[0]) / 5))
        # try:
        hill_SN, eqs = bisection(f, hill[0], hill[-1], p, n_steps)
        # except TypeError:
        #    print(hill[0], hill[-1], p, n_steps)

        hill_for_saddle.append(hill_SN)
        equilibria_for_saddle.append(eqs)
        return hill_for_saddle, equilibria_for_saddle

    while hillIdx < len(hill) - 1:
        hillMin = hill[hillIdx]  # update lower hill coefficient bound
        hillMax = hill[hillIdx + 1]  # update upper hill coefficient bound
        numEquilibriaOld = numEquilibria
        numEquilibria, eq = count_eq(f, hillMax, p, gridDensity)
        hillIdx += 1  # increment hill index counter
        if numEquilibria - numEquilibriaOld != 0:
            n_steps = int(np.ceil(log((hillMax - hillMin) / 5)))
            hill_SN, equilibria = bisection(f, hillMin, hillMax, p, n_steps)
            hill_for_saddle.append(hill_SN)
            equilibria_for_saddle.append(equilibria)

    return hill_for_saddle, equilibria_for_saddle


def bisection(f, hill0, hill1, p, n_steps):
    if n_steps is 0:
        return np.array([hill0, hill1])

    nEq0, Eq0 = count_eq(f, hill0, p)
    nEq1, Eq1 = count_eq(f, hill1, p)
    for i in range(n_steps):
        if hill1 - hill0 > 1:
            hill_middle = (hill0 + hill1) / 2
            nEqmiddle, EqMiddle = count_eq(f, hill_middle, p)

            if nEqmiddle == nEq0:
                hill0 = hill_middle
                nEq0 = nEqmiddle
                Eq0 = EqMiddle
            elif nEqmiddle == nEq1:
                hill1 = hill_middle
                nEq1 = nEqmiddle
                Eq1 = EqMiddle
            else:
                return hill_middle, from_eqs_select_saddle_eq(Eq0, EqMiddle)
                # doesn't matter which side you choose, the extra equilibrium will always be the saddle eqs
        else:
            break
    if nEq0 is not nEq1:
        if is_vector(Eq0):
            Eq0 = Eq0[np.newaxis, :]
        if is_vector(Eq1):
            Eq1 = Eq1[np.newaxis, :]
        if nEq0 > nEq1:
            hill = hill0
        else:
            hill = hill1
        return hill, from_eqs_select_saddle_eq(Eq0, Eq1)


def from_eqs_select_saddle_eq(equilibria_at_0, equilibria_at_1):
    # assumption: there is a different number of equilibria at 1 and at 0, select the equilibrium that is most
    # likely undergoing saddle node bifurcation
    # equilibria are stored as row vectors

    if equilibria_at_0.shape[0] == equilibria_at_1.shape[0]:
        warnings('NO - this cannot be - saddle nodes do not occur if the number of equilibria do not change')
    elif abs(equilibria_at_0.shape[0] - equilibria_at_1.shape[0])>2:
        warnings('NO - this cannot be - saddles do not occur if the number of equilibria changes by more than 2')
    if equilibria_at_0.shape[0] > equilibria_at_1.shape[0]:
        temp = equilibria_at_0
        equilibria_at_0 = equilibria_at_1
        equilibria_at_1 = temp
    # 0 has less equilibria than 1
    for i in range(equilibria_at_0.shape[0]):
        idx = find_nearest_row(equilibria_at_1, equilibria_at_0[i, :])
        equilibria_at_1 = np.delete(equilibria_at_1, [idx], axis=0)
    if equilibria_at_1.shape[1] == 1:
        return equilibria_at_1
    else:
        equilibrium = np.mean(equilibria_at_1, axis=1)
        return equilibrium


def find_nearest_row(array2D, value1D):
    match = np.zeros((1, array2D.shape[1]))
    for i in range(len(match)):
        match[i] = np.norm(array2D[i, :] - value1D)
    idx = match.argmin()
    return idx


def find_saddle_coef(hill_model, hillRange, parameter, freeParameter=0):
    """
    This function takes a Hill model and search for saddle nodes

    INPUTS:
    hill_model      a hill model with a singled parameter out - the one for the search
    hillRange       bounds on the singled parameter
    parameter       all but one parameter are fixed
    freePrameter    parameter w.r.t. which we are looking for the saddle node - default is the initial parameter

    OUTPUT
    FAILURE: 0,0            found no saddle node
    ELSE
    SNParameters            list of all singled parameter values undergoing a saddle node
    badCandidates           list of all singled parameter values that *should* be undergoing a saddle node, but we
                            couldn't find it
    """
    f = hill_model
    SN = SaddleNode(f)
    p = parameter
    badCandidates = []  # list for parameters which pass the candidate check but fail to find a saddle node
    SNParameters = []
    hill_for_saddle, equilibria_for_saddle = estimate_saddle_node(f, hillRange, p)
    # print('Coarse grid: {0}'.format(candidateInterval))
    if len(hill_for_saddle) == 0:
        return 0, 0
        # signature of monostability
    else:
        while hill_for_saddle:  # p should have at least one saddle node point
            candidateHill = np.array(hill_for_saddle.pop())
            equilibria = np.array(equilibria_for_saddle.pop())
            SN_candidate_eq = SN_candidates_from_bisection(equilibria)
            jkSols = SN.find_saddle_node(freeParameter, candidateHill, p, equilibria=SN_candidate_eq)
            jSols = ezcat(jkSols)
            if len(jSols) > 0:
                jSols = np.unique(np.round(jSols, 10))  # uniquify solutions
                SNParameters.append(jSols)
            else:
                badCandidates.append((candidateHill,
                                      equilibria))  # error in computation : there is a saddle node but we could not find it
    return SNParameters, badCandidates
