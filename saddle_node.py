"""
A saddle-node bifurcation class and related functionality.

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 5/18/20; Last revision: 6/25/20
"""
import numpy as np
from scipy import optimize
from hill_model import *


def SN_candidates_from_bisection(equilibria):
    """Given an array whose columns are equilibria, return the center of the midpoint between the two equilibria nearest
    to one another."""
    nEquilibria = np.shape(equilibria)[0]  # count columns of equilibrium data
    if nEquilibria == 1:
        return equilibria

    minDistance = np.inf  # initialize distance between nearest equilibrium pair
    eqPair = (0, 0)  # initialize indices for nearest equilibrium pair
    for idx1 in range(nEquilibria):
        for idx2 in range(idx1 + 1, nEquilibria):
            eqDistance = np.linalg.norm(equilibria[idx1, :] - equilibria[idx2, :])
            if eqDistance < minDistance:
                minDistance = eqDistance
                eqPair = (idx1, idx2)
    return np.row_stack((equilibria[eqPair[0], :] + equilibria[eqPair[1], :]) / 2 ) # return midpoint between 2 closest equilibria


class SaddleNode:
    """A constraint class for working with HillModels along surfaces of saddle-node bifurcations"""

    def __init__(self, hillModel, phaseCondition=lambda v: np.linalg.norm(v) - 1,
                 phaseConditionDerivative=lambda v: v / np.linalg.norm(v)):
        """Construct an instance of a saddle-node problem for specified HillModel"""

        self.model = hillModel
        self.phaseCondition = phaseCondition
        self.diffPhaseCondition = phaseConditionDerivative
        self.mapDimension = 2 * hillModel.dimension + 1  # degrees of freedom of the constraint

    def __call__(self, u):
        """Evaluation function for the saddle-node bifurcation constraints. This is a map of the form
        g: R^{2n + m} ---> R^{2n + 1} which is zero if and only if u corresponds to a saddle-node bifurcation point.

        INPUT: u = (x, v, p) where x is a state vector, v a tangent vector, and p a parameter vector."""

        stateVector, tangentVector, fullParameter = self.unpack_components(u)  # unpack input vector
        g1 = self.model(stateVector,
                        fullParameter)  # this is zero iff x is an equilibrium for the system at parameter value n

        g2 = self.model.dx(stateVector,
                           fullParameter) @ tangentVector  # this is zero iff v lies in the kernel of Df(x, n)
        g3 = self.phaseCondition(tangentVector)  # this is zero iff v satisfies the phase condition
        return ezcat(g1, g2, g3)

    def find_saddle_node(self, freeParameterIndex, *parameter, equilibria=None, freeParameterValues=None, uniqueDigits=5, flag_return=0):
        """Attempt to find isolated saddle-node points along the direction of the parameter at the
        freeParameterIndex. All other parameters are fixed. This is done by Newton iteration starting at each
        equilibrium found for the initial parameter. The function returns only values of the free parameter or returns
        None if it fails to find any

        Inputs:
            equilibria - Specify a list of equilibria to use as initial guesses. If none are specified it uses any equilibria
            which are found using the find_equilibria method.
        flag_return asks for complete info on the parameters and solutions at the saddle node"""

        if equilibria is None:  # start the saddle node search at the equilibria returned by the find_equilbria method
            equilibria = self.model.find_equilibria(10, *parameter)
        if equilibria is None:
            print('No equilibria found for parameter: {0} \n'.format(parameter))
            return []
        fullParameter = ezcat(*parameter)  # concatenate input parameter to full ordered parameter vector
        fixedParameter = fullParameter[[idx for idx in range(len(fullParameter)) if idx != freeParameterIndex]]
        if freeParameterValues is None:
            freeParameter = [fullParameter[freeParameterIndex]]
        else:
            freeParameter = ezcat(fullParameter[freeParameterIndex], freeParameterValues)

        def curry_parameters(u):
            x, v, p0 = self.unpack_components(u)  # layout components in (R^n, R^n, R)
            return ezcat(x, v, np.insert(fixedParameter, freeParameterIndex,
                                         p0))  # embed into (R^n, R^n, R^m) by currying fixed Parameters

        def init_eigenvector(equilibrium, rho):
            """Choose an initial eigenvector for the saddle-node root finding problem"""
            p = np.insert(fixedParameter, freeParameterIndex, rho)
            tangentVector = -np.linalg.solve(self.model.dx(equilibrium, p),
                                             self.model.diff(equilibrium, p, diffIndex=freeParameterIndex))
            # in R, we still have one choice of orientation once we fix the norm - this is done by checking the sign of
            # the first element of the eigenvector. If we extend to C, more work will be needed
            if tangentVector[0] < 0:
                tangentVector = - tangentVector
            return tangentVector / np.linalg.norm(tangentVector)

        def root(u0):
            """Attempts to return a single root of the SaddleNode rootfinding problem"""

            return find_root(lambda u: self.__call__(curry_parameters(u)),
                             lambda u: self.diff(curry_parameters(u), diffIndex=freeParameterIndex),
                             u0,
                             diagnose=True)

        if flag_return is 0:
            saddleNodePoints = []
        else:
            saddleNodePoints = np.empty((0, self.mapDimension))

        for parmValue in freeParameter:
            saddleNodeZeros = list(filter(lambda soln: soln.success,
                                          [root(ezcat(equilibria[:, j], init_eigenvector(equilibria[:, j], parmValue),
                                                      parmValue))
                                           for j in
                                           range(equilibria.shape[1])]))  # return equilibria which converged
            if saddleNodeZeros and flag_return is 0:
                addSols = np.array([sol.x[-1] for sol in saddleNodeZeros])
                saddleNodePoints = ezcat(saddleNodePoints, addSols[addSols > 0])
            elif saddleNodeZeros:
                addSols = np.array([sol.x for sol in saddleNodeZeros])
                saddleNodePoints = np.append(saddleNodePoints, addSols, axis=0)

        if flag_return is 0 and len(saddleNodePoints) > 0:
            return np.unique(np.round(saddleNodePoints, uniqueDigits))
        elif len(saddleNodePoints) > 0:
            return np.unique(np.round(saddleNodePoints, uniqueDigits), axis=0)  # remove duplicates and return values
        else:  # return empty array. nothing found
            return saddleNodePoints

    def unpack_components(self, u):
        """Unpack the input vector for a SaddleNode problem into 3 component vectors of the form (x, v, p) where:

            x is the state vector in R^n
            v is the tangent vector in R^n
            p is the parameter vector in R^m
            """
        n = self.model.dimension
        return u[:n], u[n:2 * n], u[2 * n:]

    def diff(self, u, diffIndex=None):
        """Evaluate the derivative of the zero finding map. This is a matrix valued function of the form
        Dg: R^{2n+M} ---> M_{2n+1 x 2n+m}(R).
        INPUT: u = (x, v, all parameters) where x is a state vector, v a tangent vector."""

        # unpack input vector and set dimensions for Jacobian blocks
        n = self.model.dimension
        parameterDim = self.model.nVariableParameter if diffIndex is None else len(ezcat(diffIndex))
        dimension_space = 2 * n + parameterDim
        mapDimension = 2 * n + 1

        stateVector, tangentVector, fullParameter = self.unpack_components(u)  # unpack input vector
        Dg = np.zeros([mapDimension, dimension_space])  # initialize (2n+1)-by-(2n+m) matrix
        Dxf = self.model.dx(stateVector, fullParameter)  # store derivative of vector field which appears in 2 blocks

        # the indices of x v and p respectively
        index1 = np.arange(n)
        index2 = np.arange(n) + n
        index3 = 2 * n + np.arange(parameterDim)
        if parameterDim is 1:
            # BLOCK ROW 1
            Dg[np.ix_(index1, index1)] = Dxf  # block - (1,1)
            # block - (1, 2) is an n-by-n zero block
            a = self.model.diff(stateVector, fullParameter, diffIndex=diffIndex)  # block - (1,3)
            Dg[index1, index3] = a
            # BLOCK ROW 2
            Dg[np.ix_(index2, index1)] = np.einsum('ijk,j', self.model.dx2(stateVector, fullParameter),
                                                   tangentVector)  # block - (2,1)
            Dg[np.ix_(index2, index2)] = Dxf  # block - (2,2)
            Dg[index2, index3] = np.einsum('ij, j',
                                           self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex),
                                           tangentVector)  # block - (2,3)
            # BLOCK ROW 3
            # block - (3, 1) is a 1-by-n zero block
            Dg[index3, np.ix_(index2)] = self.diffPhaseCondition(tangentVector)  # block - (3,2)
            # block - (3, 1) is a 1-by-1 zero block
        else:
            # BLOCK ROW 1
            Dg[np.ix_(index1, index1)] = Dxf  # block - (1,1)
            # block - (1, 2) is an n-by-n zero block
            a = self.model.diff(stateVector, fullParameter, diffIndex=diffIndex)  # block - (1,3)
            Dg[np.ix_(index1, index3)] = a
            # BLOCK ROW 2
            Dg[np.ix_(index2, index1)] = np.einsum('ijk,j', self.model.dx2(stateVector, fullParameter),
                                                   tangentVector)  # block - (2,1)
            Dg[np.ix_(index2, index2)] = Dxf  # block - (2,2)
            Dg[np.ix_(index2, index3)] = np.einsum('ijk, j',
                                                   self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex),
                                                   tangentVector)  # block - (2,3)
            # BLOCK ROW 3
            # block - (3, 1) is a 1-by-n zero block
            Dg[index3[0], np.ix_(index2)] = self.diffPhaseCondition(tangentVector)  # block - (3,2)
            # block - (3, 1) is a 1-by-1 zero block
        return Dg

    def diff2(self, u, diffIndex=None):
        """Evaluate the second derivative of the zero finding map. This is a function of the form
        D^2g: R^{2n+M} ---> R^{2n+1 x 2n+1+M x 2n+1+M}, M length of diffIndex.
        INPUT: u = (x, v, all parameters) where x is a state vector, v a tangent vector."""

        # unpack input vector and set dimensions for Hessian blocks
        n = self.model.dimension
        parameterDim = self.model.nVariableParameter if diffIndex is None else len(ezcat(diffIndex))
        dimension_space = 2 * n + parameterDim
        mapDimension = 2 * n + 1

        stateVector, tangentVector, fullParameter = self.unpack_components(u)  # unpack input vector
        Dg = np.zeros([mapDimension, dimension_space, dimension_space])  # initialize (2n+1)-by-(2n+m)-by-(2n+m) matrix
        Dxxf = self.model.dx2(stateVector, fullParameter)  # 3D tensor
        Dxpf = self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxxxf = self.model.dx3(stateVector, fullParameter)
        Dxxpf = self.model.dx2diff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxppf = self.model.dxdiff2(stateVector, fullParameter, diffIndex=[diffIndex, diffIndex])
        Dppf = self.model.diff2(stateVector, fullParameter, diffIndex=[diffIndex, diffIndex])

        # the indices of x v and p respectively
        index1 = np.array(range(n))
        index2 = np.array(range(n)) + n
        index3 = 2 * n + np.array(range(parameterDim))
        if parameterDim is 1:
            # ROW 1
            Dg[np.ix_(index1, index1, index1)] = Dxxf  # block - (1,1,1)
            Dg[np.ix_(index1, index1), index3] = Dxpf
            Dg[np.ix_(index1), index3, np.ix_(index1)] = Dxpf
            Dg[np.ix_(index1), index3, index3] = Dppf

            # BLOCK ROW 2 - derivatives of Dxf*v
            Dg[np.ix_(index2, index1, index1)] = np.einsum('ijkl,j', Dxxxf, tangentVector)
            Dg[np.ix_(index2, index1, index2)] = Dxxf
            Dg[np.ix_(index2, index1), index3] = np.einsum('ijk,j', Dxxpf, tangentVector)
            Dg[np.ix_(index2, index2, index1)] = Dxxf
            Dg[np.ix_(index2, index2), index3] = Dxpf
            Dg[np.ix_(index2), index3, np.ix_(index1)] = np.einsum('ijk,j', Dxxpf, tangentVector)
            Dg[np.ix_(index2), index3, np.ix_(index2)] = Dxpf
            Dg[np.ix_(index2), index3, index3] = Dxppf

            # BLOCK ROW 3
            # block - (3, :, :) is a 1-by-dim-by-dim zero block because the phase condition is linear
        else:

            # ROW 1
            Dg[np.ix_(index1, index1, index1)] = Dxxf  # block - (1,1,1)
            Dg[np.ix_(index1, index1, index3)] = Dxpf
            Dg[np.ix_(index1, index3, index1)] = np.swapaxes(Dxpf, 1, 2)
            Dg[np.ix_(index1, index3, index3)] = Dppf

            # BLOCK ROW 2 - derivatives of Dxf*v
            Dg[np.ix_(index2, index1, index1)] = np.einsum('ijkl,j', Dxxxf, tangentVector)
            Dg[np.ix_(index2, index1, index2)] = Dxxf
            Dg[np.ix_(index2, index1, index3)] = np.swapaxes(np.einsum('ijk...,j...', Dxxpf, tangentVector), 0, 2)
            Dg[np.ix_(index2, index2, index1)] = Dxxf
            Dg[np.ix_(index2, index2, index3)] = Dxpf
            Dg[np.ix_(index2, index3, index1)] = np.swapaxes(np.einsum('ijk...,j...', Dxxpf, tangentVector), 0, 1)
            Dg[np.ix_(index2, index3, index2)] = np.swapaxes(Dxpf, 1, 2)
            Dg[np.ix_(index2, index3, index3)] = np.swapaxes(np.einsum('ijk...,j...', Dxppf, tangentVector), 0, 1)

            # BLOCK ROW 3
            # block - (3, :, :) is a 1-by-dim-by-dim zero block because the phase condition is linear
        return Dg

    def global_jac(self, par, u):
        """Evaluation of the Jacobian of the saddle node problem with respect to all variables. This is a map of the
        form
        g: R^{2n + m} ---> M_{2n + m} (R)
        INPUTS:
        u = (x, v, hillCoefficient) where x is a state vector, v a tangent vector,
        par in R^m """
        mapDimension = 1
        J = np.zeros([mapDimension, mapDimension])
        return J

    def call_grid(self, parameter, gridBounds=(2, 4), nIter=10):
        # # to avoid getting to negative parameter values, we just return infinity if the parameters are biologically not good
        # if np.any(parameter < 0):
        #     return np.inf
        #
        # # set multiple values of the Hill coefficient and test for the convergence to a saddle node
        # n0Grid = np.linspace(*gridBounds, nIter)
        # minHill = np.inf
        # kIter = 1
        # while kIter < nIter and minHill == np.inf:
        #     minHill = self(parameter, n0Grid[kIter])
        #     kIter += 1
        # return minHill
        print('This function needs to be updated before being called')
        return

    def find_minimizer(self, parameter):
        """Find a minimizer of a given loss function with respect to remaining parameters via gradient descent"""

        # TODO: 1. This needs to be overhauled to match the fully general class
        #
        # def local_function(parameter):
        #     return self.call_grid(parameter)
        #
        # minima = optimize.minimize(local_function, parameter, method='nelder-mead', options={'xatol': 1e-2})
        # return minima

        print('This function needs to be updated before being called')
        return
