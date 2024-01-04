"""
Classes and methods for constructing, evaluating, and doing parameter continuation of Hill Models

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 2/29/2020
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
from itertools import product, permutations
from scipy import optimize, linalg
from numpy import log
import textwrap

# ignore overflow and division by zero warnings:
np.seterr(over='ignore', invalid='ignore')


def npA(size, dim=2):
    """Return a random square integer matrix of given size for testing numpy functions."""
    A = np.random.randint(1, 10, dim * [size])
    return np.asarray(A, dtype=float)


def is_vector(array):
    """Returns true if input is a numpy vector i.e. has shape (n,). Returns false for column or row vectors i.e.
    objects with shape (n,1) or (1,n)."""

    return len(np.shape(array)) == 1


def ezcat(*coordinates):
    """A multiple dispatch concatenation function for numpy arrays. Accepts arbitrary inputs as int, float, tuple,
    list, or numpy array and concatenates into a vector returned as a numpy array. This is recursive so probably not
    very efficient for large scale use."""

    if len(coordinates) == 1:
        if isinstance(coordinates[0], list):
            return np.array(coordinates[0])
        elif isinstance(coordinates[0], np.ndarray):
            return coordinates[0]
        else:
            return np.array([coordinates[0]])

    try:
        return np.concatenate([coordinates[0], ezcat(*coordinates[1:])])
    except ValueError:
        return np.concatenate([np.array([coordinates[0]]), ezcat(*coordinates[1:])])


def find_root(f, Df, initialGuess, diagnose=False):
    """Default root finding method to use if one is not specified"""

    solution = optimize.root(f, initialGuess, jac=Df, method='hybr', tol=10 ** -10)  # set root finding algorithm
    x = skinny_newton(f, Df, solution.x, maxDefect=1e-10)
    if np.any(x != solution.x):  # sign that Newton converged
        if diagnose:
            solution.x = x
            solution.success = True
            solution.message = 'Used skinny_newton for convergence'
            # TODO: fix the other attributes
            return solution
        return x
    if diagnose:
        return solution  # return the entire solution object including iterations and diagnostics
    else:
        return solution.x  # return only the solution vector


def skinny_newton(f, Df, x0, maxDefect=1e-13):
    """A full Newton based root finding algorithm"""

    def is_singular(matrix, rank):
        """Returns true if the derivative becomes singular for any reason"""
        return np.isnan(matrix).any() or np.isinf(matrix).any() or np.linalg.matrix_rank(matrix) < rank

    fDim = len(x0)  # dimension of the domain/image of f
    maxIterate = 10

    if not is_vector(x0):  # an array whose columns are initial guesses
        print('not implemented yet')
        return np.NAN

    # initialize iteration
    x = x0.copy()
    y = f(x)
    Dy = Df(x)
    iDefect = np.linalg.norm(y)  # initialize defect
    iIterate = 1
    while iDefect > maxDefect and iIterate < maxIterate:
        if fDim == 1:
            x -= y / Dy
        else:
            x -= np.linalg.solve(Dy, y)  # update x
        y = f(x)  # update f(x)
        Dy = Df(x)  # update Df(x)
        iDefect = np.linalg.norm(y)  # initialize defect
        iIterate += 1

    if iDefect < maxDefect or iDefect < np.linalg.norm(f(x0)):
        return x
    else:
        return x0


def full_newton(f, Df, x0, maxDefect=1e-13):
    """A full Newton based root finding algorithm"""

    def is_singular(matrix, rank):
        """Returns true if the derivative becomes singular for any reason"""
        return np.isnan(matrix).any() or np.isinf(matrix).any() or np.linalg.matrix_rank(matrix) < rank

    fDim = len(x0)  # dimension of the domain/image of f
    maxIterate = 100

    if not is_vector(x0):  # an array whose columns are initial guesses
        print('not implemented yet')

    else:  # x0 is a single initial guess
        # initialize iteration
        x = x0.copy()
        y = f(x)
        Dy = Df(x)
        iDefect = np.linalg.norm(y)  # initialize defect
        iIterate = 1
        while iDefect > maxDefect and iIterate < maxIterate and not is_singular(Dy, fDim):
            if fDim == 1:
                x -= y / Dy
            else:
                x -= np.linalg.solve(Dy, y)  # update x

            y = f(x)  # update f(x)
            Dy = Df(x)  # update Df(x)
            iDefect = np.linalg.norm(y)  # initialize defect
            iIterate += 1

        if iDefect < maxDefect:
            return x
        else:
            print('Newton failed to converge')
            return np.nan


def verify_call(func):
    """Evaluation method decorator for validating evaluation calls. This can be used when troubleshooting and
    testing and then omitted in production runs to improve efficiency. This decorates any HillModel or
    HillCoordinate method which has inputs of the form (self, x, *parameter, **kwargs)."""

    def func_wrapper(*args, **kwargs):
        hillObj = args[0]  # a HillModel or HillCoordinate instance is passed as 1st position argument to evaluation
        # method
        x = args[1]  # state vector passed as 2nd positional argument to evaluation method
        if issubclass(type(hillObj), HillCoordinate):
            N = hillObj.nState
            parameter = args[2]  # parameter vector passed as 3rd positional argument to evaluation method
        elif issubclass(type(hillObj), HillModel):
            N = hillObj.dimension
            parameter = hillObj.parse_parameter(*args[2:])  # parameters passed as variable argument to
            # evaluation method and need to be parsed to obtain a single parameter vector
        else:
            raise TypeError('First argument must be a HillCoordinate or HillModel instance. Instead it received {'
                            '0}'.format(type(hillObj)))

        if len(x) != N:  # make sure state input is the correct size
            raise IndexError(
                'State vector for this evaluation should be size {0} but received a vector of size {1}'.format(N,
                                                                                                               len(x)))
        elif len(parameter) != hillObj.nParameter:
            raise IndexError(
                'Parsed parameter vector for this evaluation should be size {0} but received a vector of '
                'size {1}'.format(
                    hillObj.nParameter, len(parameter)))

        else:  # parameter and state vectors are correct size. Pass through to evaluation method
            return func(*args, **kwargs)

    return func_wrapper


PARAMETER_NAMES = ['ell', 'delta', 'theta', 'hillCoefficient']  # ordered list of HillComponent parameter names


class HillComponent:
    """A component of a Hill system of the form ell + delta*H(x; ell, delta, theta, n) where H is an increasing or decreasing Hill function.
    Any of these parameters can be considered as a fixed value for a Component or included in the callable variables. The
    indices of the edges associated to ell, and delta are different than those associated to theta."""

    def __init__(self, productionSign, **kwargs):
        """A Hill function with parameters [ell, delta, theta, n] of productionType in {-1, 1} to denote H^-, H^+ """
        # TODO: Class constructor should not do work!

        self.sign = productionSign
        self.parameterValues = np.zeros(4)  # initialize vector of parameter values
        parameterNames = PARAMETER_NAMES.copy()  # ordered list of possible parameter names
        parameterCallIndex = {parameterNames[j]: j for j in range(4)}  # calling index for parameter by name
        for parameterName, parameterValue in kwargs.items():
            setattr(self, parameterName, parameterValue)  # fix input parameter
            self.parameterValues[
                parameterCallIndex[parameterName]] = parameterValue  # update fixed parameter value in evaluation vector
            del parameterCallIndex[parameterName]  # remove fixed parameter from callable list

        self.variableParameters = list(parameterCallIndex.keys())  # set callable parameters
        self.parameterCallIndex = list(parameterCallIndex.values())  # get indices for callable parameters
        self.fixedParameter = [parameterName for parameterName in parameterNames if
                               parameterName not in self.variableParameters]
        #  set callable parameter name functions
        for idx in range(len(self.variableParameters)):
            self.add_parameter_call(self.variableParameters[idx], idx)

    def __iter__(self):
        """Make iterable"""
        yield self

    def add_parameter_call(self, parameterName, parameterIndex):
        """Adds a call by name function for variable parameters to a HillComponent instance"""

        def call_function(self, parameter):
            """returns a class method which has the given parameter name. This method slices the given index out of a
            variable parameter vector"""
            return parameter[parameterIndex]

        setattr(HillComponent, parameterName, call_function)  # set dynamic method name

    def curry_parameters(self, parameter):
        """Returns a parameter evaluation vector in R^4 with fixed and variable parameters indexed properly"""

        # TODO: When all parameters of this component are fixed this function still requires an empty list as an argument.
        parameterEvaluation = self.parameterValues.copy()  # get a mutable copy of the fixed parameter values
        parameterEvaluation[self.parameterCallIndex] = parameter  # slice passed parameter vector into callable slots
        return parameterEvaluation

    def __call__(self, x, *parameter):
        """Evaluation method for a Hill component function instance"""

        # TODO: Handle the case that negative x values are passed into this function. - it is well defined

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.

        # evaluation rational part of the Hill function
        if self.sign == -1:
            fraction = x / theta
        else:
            fraction = theta / x

        return ell + delta / (1 + fraction ** hillCoefficient)

    def __repr__(self):
        """Return a canonical string representation of a Hill component"""

        reprString = 'Hill Component: \n' + 'sign = {0} \n'.format(self.sign)
        for parameterName in PARAMETER_NAMES:
            if parameterName not in self.variableParameters:
                reprString += parameterName + ' = {0} \n'.format(getattr(self, parameterName))
        reprString += 'Variable Parameters: {' + ', '.join(self.variableParameters) + '}\n'
        return reprString

    def dx(self, x, parameter):
        """Evaluate the derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        if self.sign == -1:
            fraction_power = (x / theta) ** hillCoefficient
        else:
            fraction_power = (theta / x) ** hillCoefficient

        if fraction_power == 0:
            return 0

        return self.sign * hillCoefficient * delta / (x * (1 / fraction_power + 2 + fraction_power))

    def dx2(self, x, parameter):
        """Evaluate the second derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        if self.sign == -1:
            fraction_power = (x / theta) ** hillCoefficient
        else:
            fraction_power = (theta / x) ** hillCoefficient
        if fraction_power == 0:
            return 0

        return hillCoefficient * delta / x ** 2 * (
                2 * hillCoefficient / (1 / fraction_power ** 2 + 3 / fraction_power + 3 + fraction_power) +
                (-self.sign - hillCoefficient) / (1 / fraction_power + 2 + fraction_power))

    def diff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 1.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        if self.sign == -1:
            fraction = (x / theta)
            fraction_power = (x / theta) ** hillCoefficient
        else:
            fraction = theta / x
            fraction_power = (theta / x) ** hillCoefficient

        if diffParameter == 'delta':
            thetaPower = theta ** hillCoefficient  # compute theta^hillCoefficient only once
            dH = 1 / (fraction_power + 1)

        elif diffParameter == 'theta':
            dH = -self.sign * (delta * hillCoefficient) / (theta * (fraction_power + 2 + 1 / fraction_power))

        elif diffParameter == 'hillCoefficient':
            dH = - delta * log(fraction) / (fraction_power + 2 + 1 / fraction_power)

        return dH

    def diff2(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        # ordering of the variables decrease options
        if diffIndex[0] > diffIndex[1]:
            diffIndex = diffIndex[[1, 0]]

        diffParameter0 = self.variableParameters[diffIndex[0]]  # get the name of the differentiation variable
        diffParameter1 = self.variableParameters[diffIndex[1]]  # get the name of the differentiation variable

        if diffParameter0 == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        # precompute some powers
        if self.sign == -1:
            fraction = (x / theta)
            fraction_power = (x / theta) ** hillCoefficient
        else:
            fraction = theta / x
            fraction_power = (theta / x) ** hillCoefficient

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = -self.sign * hillCoefficient / theta / (fraction_power + 2 + 1 / fraction_power)
            if diffParameter1 == 'hillCoefficient':
                dH = - log(fraction) / (fraction_power + 2 + 1 / fraction_power)

        elif diffParameter0 == 'theta':
            if diffParameter1 == 'theta':
                dH = 2 * hillCoefficient / (1 / fraction_power ** 2 + 3 / fraction_power + 3 + fraction_power) - (
                            hillCoefficient -self.sign) / (fraction_power + 2 + 1 / fraction_power)
                dH = dH * delta * hillCoefficient / theta ** 2
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * 2 * delta * hillCoefficient * log(fraction) / theta / (fraction_power + 3 +
                                3 / fraction_power + 1 / fraction_power ** 2) - self.sign * delta * (
                                1 + hillCoefficient * log(fraction)) / theta / (fraction_power + 2 + 1 / fraction_power)
                # dH = self.sign * -delta * hillCoefficient * xPower * thetaPowerSmall / ((thetaPower + xPower) ** 2)

        elif diffParameter0 == 'hillCoefficient':
            # then diffParameter1 = 'hillCoefficient'
            dH = 2 * delta * log(fraction)**2/(fraction_power+3+3/fraction_power+1/fraction_power**2) - delta * log(fraction)/(fraction_power+2+1/fraction_power)

        return dH

    def dxdiff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

            if self.sign == -1:
                fraction = (x / theta)
                fraction_power = (x / theta) ** hillCoefficient
            else:
                fraction = theta / x
                fraction_power = (theta / x) ** hillCoefficient

        if diffParameter == 'delta':
            ddH = self.sign * hillCoefficient / x / (fraction_power + 2 + 1/fraction_power)

        elif diffParameter == 'theta':
            ddH = - 2/(fraction_power + 3 + 3/fraction_power + 1/fraction_power**2) + 1/(fraction_power+2+1/fraction_power)
            ddH = ddH * delta * hillCoefficient**2 /(theta*x)

        elif diffParameter == 'hillCoefficient':
            ddH = -self.sign * 2 * delta * hillCoefficient * log(fraction)/x/(fraction_power+3+3/fraction_power+1/fraction_power**2) - \
                  -self.sign * delta * (hillCoefficient * log(fraction)+1)/x/(fraction_power+2+1/fraction_power)
        return ddH

    def dx2diff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the HillComponent."""
        # TODO: get the derivatives more numerically stable

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        hill = hillCoefficient
        if self.sign == -1:
            fraction = (x / theta)
            fraction_power = (x / theta) ** hill
        else:
            fraction = theta / x
            fraction_power = (theta / x) ** hill


        if diffParameter == 'delta':
            d3H = 2 * hill / (fraction_power + 3 + 3/fraction_power + 1/fraction_power**2) - (hill - self.sign)/(fraction_power+2+1/fraction_power)
            d3H = hill/x**2 * d3H

        elif diffParameter == 'theta':
            third_power = fraction_power + 3 + 3/fraction_power + 1/fraction_power**2
            fourth_power = fraction_power + 4 + 6/fraction_power + 4/fraction_power**2 + 1/fraction_power**3
            d3H = -self.sign * 6 * hill /fourth_power + (self.sign * hill + 2)/third_power
            d3H = delta * hill**2/(theta * x**2) * d3H

        elif diffParameter == 'hillCoefficient':
            third_power = fraction_power + 3 + 3/fraction_power + 1/fraction_power**2
            fourth_power = fraction_power + 4 + 6/fraction_power + 4/fraction_power**2 + 1/fraction_power**3
            log_frac = log(fraction)
            d3H = - 6 * hill * log_frac/fourth_power + (6 * hill* log_frac + self.sign * 2 * log_frac + 4)/third_power + log_frac *(-hill - self.sign)/(fraction_power + 2 + 1/fraction_power)
            d3H = delta * hill/(x**2) * d3H

        return d3H

    def dxdiff2(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""
        # TODO: get the derivatives more numerically stable

        # ordering of the variables decrease options
        if diffIndex[0] > diffIndex[1]:
            diffIndex = diffIndex[[1, 0]]

        diffParameter0 = self.variableParameters[diffIndex[0]]  # get the name of the differentiation variable
        diffParameter1 = self.variableParameters[diffIndex[1]]  # get the name of the differentiation variable

        if diffParameter0 == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        hill = hillCoefficient

        # precompute some powers
        # this is the only power of x e will need
        xPower_minus = x ** (hill - 1)
        xPower = x * xPower_minus
        # here we check which powers of theta we will need and compute them
        if diffParameter0 == 'theta' and diffParameter1 == 'theta':
            thetaPower_minusminus = theta ** (hillCoefficient - 2)
            thetaPower_minus = theta * thetaPower_minusminus  # compute power of theta only once
            thetaPower = theta * thetaPower_minus

        else:
            if diffParameter0 == 'theta' or diffParameter1 == 'theta':
                thetaPower_minus = theta ** (hillCoefficient - 1)  # compute power of theta only once
                thetaPower = theta * thetaPower_minus
            else:
                thetaPower = theta ** hillCoefficient

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = self.sign * hill ** 2 * thetaPower_minus * xPower_minus * (xPower - thetaPower) / \
                     ((thetaPower + xPower) ** 3)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * ((thetaPower * xPower_minus * (-hill * (thetaPower - xPower) * (log(theta) - log(x)) +
                                                                thetaPower + xPower))) / ((thetaPower + xPower) ** 3)

        elif diffParameter0 == 'theta':
            if diffParameter1 == 'theta':
                dH = (self.sign * delta * hill ** 2 * thetaPower_minusminus * xPower_minus * (
                        (hill + 1) * thetaPower ** 2
                        - 4 * hill * thetaPower * xPower + (hill - 1) * xPower ** 2)) / ((thetaPower + xPower) ** 4)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * (delta * hill * thetaPower_minus * xPower_minus * (-2 * thetaPower ** 2 +
                                                                                    hill * thetaPower ** 2 - 4 * thetaPower * xPower + xPower ** 2) *
                                  (log(theta) - log(x)) + 2 * xPower ** 2) / ((thetaPower + xPower) ** 4)

        elif diffParameter0 == 'hillCoefficient':
            # then diffParameter1 = 'hillCoefficient'
            dH = self.sign * (delta * thetaPower * xPower_minus * (log(theta) - log(x)) * (-2 * thetaPower ** 2 + hill *
                                                                                           (
                                                                                                   thetaPower ** 2 - 4 * thetaPower * xPower + xPower ** 2) * (
                                                                                                   log(theta) - log(
                                                                                               x)) +
                                                                                           2 * xPower ** 2) / (
                                      (thetaPower + xPower) ** 4))

        return dH

    def dx3(self, x, parameter):
        """Evaluate the second derivative of a Hill component with respect to x"""
        # TODO: get the derivatives more numerically stable

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        hill = hillCoefficient
        thetaPower = theta ** hillCoefficient
        theta2Power = thetaPower ** 2
        xPower_der3 = x ** (hill - 3)
        xPower_der2 = x * xPower_der3
        xPower_der = x * xPower_der2  # compute x^{hillCoefficient-1}
        xPower = xPower_der * x
        x2Power = xPower ** 2
        hillsquare = hill ** 2
        return self.sign * (hill * delta * thetaPower * xPower_der3) / ((xPower + thetaPower) ** 4) * \
               (hillsquare * theta2Power - 4 * hillsquare * thetaPower * xPower + hillsquare * x2Power - \
                3 * hill * theta2Power + 2 * theta2Power + 4 * thetaPower * xPower + 3 * hill * x2Power + 2 * x2Power)

    def image(self, parameter=None):
        """Return the range of this HillComponent given by (ell, ell+delta)"""

        if 'ell' in self.variableParameters:
            ell = self.ell(parameter)
        else:
            ell = self.ell

        if 'delta' in self.variableParameters:
            delta = self.delta(parameter)
        else:
            delta = self.delta

        return np.array([ell, ell + delta])


class HillCoordinate:
    """Define a single scalar coordinate of a Hill system. This function always takes the form of a linear decay term
    and a nonlinear production term defined by composing a polynomial interaction function with HillCoordinates
    which each depend only on a single variable. Specifically, a HillCoordinate represents a function, f : R^K ---> R.
    which describes rate of production and decay of a scalar quantity x. If x does not have a nonlinear self production,
    (i.e. no self loop in the network topology) then this is a scalar differential equation taking the form
    x' = -gamma*x + p(H_1, H_2, ...,H_{K-1})
    where each H_i is a Hill function depending on a single state variable y_i != x. Otherwise, if x does contribute to
    its own nonlinear production, then f takes the form,
    x' = -gamma*x + p(H_1, H_2,...,H_K)
    where H_1 depends on x and H_2,...,H_K depend on a state variable y_i != x."""

    def __init__(self, parameter, productionSign, productionType, nStateVariables, gamma=np.nan):
        """Hill Coordinate instantiation with the following syntax:
        INPUTS:
            gamma - (float) decay rate for this coordinate or NaN if gamma is a variable parameter which is callable as
                the first component of the parameter variable vector.
            parameter - (numpy array) A K-by-4 array of Hill component parameters with rows of the form [ell, delta, theta, hillCoefficient]
                Entries which are NaN are variable parameters which are callable in the function and all derivatives.
            productionSign - (list) A vector in F_2^K carrying the sign type for each Hill component
            productionType - (list) A vector describing the interaction type of the interaction function specified as an
                    ordered integer partition of K.
            nStateVariables - (integer) Report how many state variables this HillCoordinate depends on. All evaluation methods will expect a state vector of
            this size. """

        # TODO: 1. Class constructor should not do work!
        self.gammaIsVariable = np.isnan(gamma)
        if ~np.isnan(gamma):
            self.gamma = gamma  # set fixed linear decay
        self.nState = nStateVariables  # dimension of state vector input to HillCoordinate
        self.parameterValues = parameter  # initialize array for the fixed (non-variable) parameter values
        self.nProduction = len(productionSign)  # number of incoming edges contributing to nonlinear production. In the
        # current version this is always equal to self.nState - 1 (no self edge) or self.nState (self edge)
        self.productionIndex = list(range(self.nState)[slice(-self.nProduction, None,
                                                             1)])  # state variable selection for the production term are the trailing
        # K variables. If this coordinate has a self edge this is the entire vector, otherwise, it selects all state
        # variables except the first state variable.
        self.productionComponents, self.nParameterByProductionIndex, self.productionParameterIndexRange = self.set_production(
            parameter, productionSign)
        self.productionType = productionType  # specified as an integer partition of K
        self.summand = self.set_summand()

        self.nParameter = sum(
            self.nParameterByProductionIndex) + int(
            self.gammaIsVariable)  # number of variable parameters for this coordinate.

    def parse_parameters(self, parameter):
        """Returns the value of gamma and slices of the parameter vector divided by component"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            gamma = parameter[0]
        else:
            gamma = self.gamma
        return gamma, [parameter[self.productionParameterIndexRange[j]:self.productionParameterIndexRange[j + 1]] for
                       j in range(self.nProduction)]

    # def verify_call(self, x, parameter):
    #     """A function to insert into evaluation methods to make sure the input variables have correct dimension and shape."""
    #
    #     if len(x) != self.nState:  # make sure input is the correct size
    #         raise IndexError(
    #             'State vector for this Hill Coordinate should be size {0} but received a vector of size {1}'.format(
    #                 self.nState, len(x)))
    #     elif len(parameter) != self.nParameter:
    #         raise IndexError(
    #             'Parameter vector for this Hill Coordinate should be size {0} but received a vector of size {1}'.format(
    #                 self.nParameter, len(parameter)))
    #     return

    def parameter_to_production_index(self, linearIndex):
        """Convert a linear parameter index to an ordered pair, (i, j) where the specified parameter is the j^th variable
         parameter of the i^th Hill production function."""

        if self.gammaIsVariable and linearIndex == 0:
            print('production index for a decay parameter is undefined')
            raise KeyboardInterrupt
        componentIndex = np.searchsorted(self.productionParameterIndexRange,
                                         linearIndex + 0.5) - 1  # get the production index which contains the variable parameter. Adding 0.5
        # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
        parameterIndex = linearIndex - self.productionParameterIndexRange[
            componentIndex]  # get the local parameter index in the HillComponent for the variable parameter
        return componentIndex, parameterIndex

    def component_to_parameter_index(self, componentIdx, localIdx):
        """Given an input (i,j), return a linear index for the j^th local parameter of the i^th Hill component"""

        return self.productionParameterIndexRange[componentIdx] + localIdx

    @verify_call
    def __call__(self, x, parameter):
        """Evaluate the Hill coordinate on a vector of state and parameter variables. This is a
        map of the form  g: R^n x R^m ---> R where n is the number of state variables for this Hill coordinate (in the current version
        with Hill functions in the decay term this is either n = K or n = K + 1). m is the number of variable parameters for this
        Hill coordinate (at most m = 1 + 4K). When calling this function for (x_1,...,x_n) is is REQUIRED that the global index of
        x_1 is the state variable associated with this HillCoordinate."""

        # TODO: Currently the input parameter must be a numpy array even if there is only a single parameter.
        # Evaluate coordinate for a single x in R^n. Slice callable parameters into a list of length K. The j^th list contains the variable parameters belonging to
        # the j^th Hill function in the production term.

        gamma, parameterByComponent = self.parse_parameters(parameter)
        productionComponentValues = self.evaluate_production_components(x, parameter)
        summandValues = self.evaluate_summand(productionComponentValues)
        nonlinearProduction = np.prod(summandValues)
        # TODO: Below is the old version. This should be removed once the refactored classes are fully vetted.
        # nonlinearProduction = self.evaluate_production_interaction(
        #     productionHillValues)  # compose with production interaction function
        return -gamma * x[0] + nonlinearProduction

    def __repr__(self):
        """Return a canonical string representation of a Hill coordinate"""

        reprString = 'Hill Coordinate: \n' + 'Production Type: p = ' + (
                '(' + ')('.join(
            [' + '.join(['z_{0}'.format(idx + 1) for idx in summand]) for summand in self.summand]) + ')\n') + (
                             'Components: H = (' + ', '.join(
                         map(lambda i: 'H+' if i == 1 else 'H-', [H.sign for H in self.productionComponents])) + ') \n')

        # initialize index strings
        stateIndexString = 'State Variables: x = (x_i; '
        variableIndexString = 'Variable Parameters: lambda = ('
        if self.gammaIsVariable:
            variableIndexString += 'gamma, '

        for k in range(self.nProduction):
            idx = self.productionIndex[k]
            stateIndexString += 'x_{0}, '.format(idx)
            if self.productionComponents[k].variableParameters:
                variableIndexString += ', '.join(
                    [var + '_{0}'.format(idx) for var in self.productionComponents[k].variableParameters])
                variableIndexString += ', '

        # remove trailing commas and close brackets
        variableIndexString = variableIndexString[:-2]
        stateIndexString = stateIndexString[:-2]
        variableIndexString += ')\n'
        stateIndexString += ')\n'
        reprString += stateIndexString + '\n          '.join(textwrap.wrap(variableIndexString, 80))
        return reprString

    def evaluate_production_components(self, x, parameter):
        """Evaluate each HillComponent for the production term. Returns an ordered vector in R^K."""

        gamma, parameterByProductionComponent = self.parse_parameters(parameter)
        return np.array(
            list(map(lambda H, x_i, parm: H(x_i, parm), self.productionComponents, x[self.productionIndex],
                     parameterByProductionComponent)))  # evaluate Hill productionComponents

    def summand_index(self, componentIdx):
        """Returns the summand index of a component index. This is a map of the form, I : {1,...,K} --> {1,...,q} which
        identifies to which summand of the production interaction the k^th production component contributes."""

        return self.summand.index(filter(lambda L: componentIdx in L, self.summand).__next__())

    def evaluate_summand(self, componentValues):
        """Evaluate the summands of the production interaction function. This is a map taking values in R^q where the input is
        a vector in R^K obtained by evaluating the Hill production components. The component values which contribute to the same summand are then
        summed according to the productionType."""

        return np.array([np.sum(componentValues[self.summand[j]]) for j in range(len(self.summand))])

    def evaluate_production_interaction(self, componentValues):
        """Evaluate the production interaction function at vector of HillComponent values: (H1,...,HK). This is the second evaluation
        in the composition which defines the production term."""

        # TODO: This function is deprecated but it is still used in the HillModel.eq_interval method. This usage
        #  should be replaced with calls to evaluate_summand and np.prod instead.
        # print('Deprecation Warning: This function should no longer be called. Use the evaluate_summand method and '
        #       'np.prod() instead.')
        if len(self.summand) == 1:  # this is the all sum interaction type
            return np.sum(componentValues)
        else:
            return np.prod([sum([componentValues[idx] for idx in summand]) for summand in self.summand])

    def diff_production(self, x, parameter, diffOrder, diffIndex=None):
        """Return the differential of the specified order for the production interaction function in the coordinate specified by
        diffIndex. The differential is evaluated at the vector of HillComponent evaluations i.e. this function serves as the
         outer function call when evaluating chain rule derivatives for HillCoordinates with respect to state or parameter vectors.
         If diffIndex is not specified, it returns the full derivative as a vector with all K partials of
        order diffOrder."""

        def nonzero_index(order):
            """Return the indices for which the given order derivative of an interaction function is nonzero. This happens
            precisely for every multi-index in the tensor for which each component is drawn from a different summand."""

            summandTuples = permutations(self.summand, order)
            summandProducts = []  # initialize cartesian product of all summand tuples
            for tup in summandTuples:
                summandProducts += list(product(*tup))

            return np.array(summandProducts)

        nSummand = len(self.productionType)  # number of summands
        if diffIndex is None:  # compute the full differential of p as a vector in R^K with each component evaluated at
            # H_k(x_k, p_k).

            if diffOrder == 1:  # compute first derivative of interaction function composed with Hill Components
                if nSummand == 1:  # the all sum special case
                    return np.ones(self.nProduction)
                else:
                    productionComponentValues = self.evaluate_production_components(x, parameter)
                    summandValues = self.evaluate_summand(productionComponentValues)
                    fullProduct = np.prod(summandValues)
                    DxProducts = fullProduct / summandValues  # evaluate all partials only once using q multiplies. The m^th term looks like P/p_m.
                    return np.array([DxProducts[self.summand_index(k)] for k in
                                     range(
                                         self.nProduction)])  # broadcast values to all members sharing the same summand

            elif diffOrder == 2:  # compute second derivative of interaction function composed with Hill Components as a 2-tensor
                if nSummand == 1:  # the all sum special case
                    return np.zeros(diffOrder * [self.nProduction])  # initialize Hessian of interaction function

                elif nSummand == 2:  # the 2 summands special case
                    DpH = np.zeros(diffOrder * [self.nProduction])  # initialize derivative tensor
                    idxArray = nonzero_index(diffOrder)  # array of nonzero indices for derivative tensor
                    DpH[idxArray[:, 0], idxArray[:, 1]] = 1  # set nonzero terms to 1
                    return DpH

                else:
                    DpH = np.zeros(2 * [self.nProduction])  # initialize Hessian of interaction function
                    # compute Hessian matrix of interaction function by summand membership
                    productionComponentValues = self.evaluate_production_components(x, parameter)
                    summandValues = self.evaluate_summand(productionComponentValues)
                    fullProduct = np.prod(summandValues)
                    DxProducts = fullProduct / summandValues  # evaluate all partials using only nSummand-many multiplies
                    DxxProducts = np.outer(DxProducts,
                                           1.0 / summandValues)  # evaluate all second partials using only nSummand-many additional multiplies.
                    # Only the cross-diagonal terms of this matrix are meaningful.
                    for row in range(nSummand):  # compute Hessian of interaction function (outside term of chain rule)
                        for col in range(row + 1, nSummand):
                            Irow = self.summand[row]
                            Icolumn = self.summand[col]
                            DpH[np.ix_(Irow, Icolumn)] = DpH[np.ix_(Icolumn, Irow)] = DxxProducts[row, col]
                    return DpH

            elif diffOrder == 3:  # compute third derivative of interaction function composed with Hill Components as a 3-tensor
                if nSummand <= 2:  # the all sum or 2-summand special cases
                    return np.zeros(diffOrder * [self.nProduction])  # initialize Hessian of interaction function

                elif nSummand == 3:  # the 2 summands special case
                    DpH = np.zeros(diffOrder * [self.nProduction])  # initialize derivative tensor
                    idxArray = nonzero_index(diffOrder)  # array of nonzero indices for derivative tensor
                    DpH[idxArray[:, 0], idxArray[:, 1], idxArray[:, 2]] = 1  # set nonzero terms to 1
                    return DpH
                else:
                    raise KeyboardInterrupt

        else:  # compute a single partial derivative of p
            if diffOrder == 1:  # compute first partial derivatives
                if len(self.productionType) == 1:
                    return 1.0
                else:
                    productionComponentValues = self.evaluate_production_components(x, parameter)
                    summandValues = self.evaluate_summand(productionComponentValues)
                    I_k = self.summand_index(diffIndex)  # get the summand index containing the k^th Hill component
                    return np.prod(
                        [summandValues[m] for m in range(len(self.productionType)) if
                         m != I_k])  # multiply over
                # all summands which do not contain the k^th component
            else:
                raise KeyboardInterrupt

    def diff_production_component(self, x, parameter, diffOrder, *diffIndex, fullTensor=True):
        """Compute derivative of component vector, H = (H_1,...,H_K) with respect to state variables or parameters. This is
        the inner term in the chain rule derivative for the higher order derivatives of a HillCoordinate. diffOrder has the form
         [xOrder, parameterOrder] which specifies the number of derivatives with respect to state variables and parameter
         variables respectively. Allowable choices are: {[1,0], [0,1], [2,0], [1,1], [0,2], [3,0], [2,1], [1,2]}"""

        xOrder = diffOrder[0]
        parameterOrder = diffOrder[1]
        gamma, parameterByComponent = self.parse_parameters(parameter)
        xProduction = x[
            self.productionIndex]  # extract only the coordinates of x that contribute to the production term.

        if parameterOrder == 0:  # return partials of H with respect to x as a length K vector of nonzero values. dH is
            # obtained by taking the diag operator to broadcast this vector to a tensor of correct rank.

            if xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, xProduction,
                             parameterByComponent)))  # evaluate vector of first order state variable partial derivatives for Hill productionComponents
            elif xOrder == 2:
                DH_nonzero = np.array(
                    list(map(lambda H_k, x_k, p_k: H_k.dx2(x_k, p_k), self.productionComponents, xProduction,
                             parameterByComponent)))  # evaluate vector of second order state variable partial derivatives for Hill productionComponents
            elif xOrder == 3:
                DH_nonzero = np.array(
                    list(map(lambda H_k, x_k, p_k: H_k.dx3(x_k, p_k), self.productionComponents, xProduction,
                             parameterByComponent)))  # evaluate vector of third order state variable partial derivatives for Hill productionComponents

            if fullTensor:
                DH = np.zeros((1 + xOrder) * [self.nProduction])
                np.einsum(''.join((1 + xOrder) * 'i') + '->i', DH)[:] = DH_nonzero
                return DH
            else:
                return DH_nonzero

        elif parameterOrder == 1:  # return partials w.r.t parameters specified by diffIndex as a vector of nonzero productionComponents.

            if not diffIndex:  # no optional argument means return all component parameter derivatives (i.e. all parameters except gamma)
                diffIndex = list(range(int(self.gammaIsVariable), self.nParameter))
            parameterComponentIndex = [self.parameter_to_production_index(linearIdx) for linearIdx in
                                       diffIndex]  # a list of ordered pairs for differentiation parameter indices

            if xOrder == 0:  # Compute D_lambda(H)
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].diff(xProduction[idx[0]],
                                                                                parameterByComponent[idx[0]],
                                                                                idx[1]),
                             parameterComponentIndex)))  # evaluate vector of first order partial derivatives for Hill productionComponents

            elif xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].dxdiff(xProduction[idx[0]],
                                                                                  parameterByComponent[idx[0]],
                                                                                  idx[1]),
                             parameterComponentIndex)))  # evaluate vector of second order mixed partial derivatives for Hill productionComponents
            elif xOrder == 2:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].dx2diff(xProduction[idx[0]],
                                                                                   parameterByComponent[idx[0]],
                                                                                   idx[1]),
                             parameterComponentIndex)))  # evaluate vector of third order mixed partial derivatives for Hill productionComponents

            if fullTensor:
                tensorDims = (1 + xOrder) * [self.nProduction] + [self.nParameter - self.gammaIsVariable]
                DH = np.zeros(tensorDims)
                nonzeroComponentIdx = list(zip(*parameterComponentIndex))[
                    0]  # zip into a pair of tuples for last two einsum indices
                nonzeroIdx = tuple((1 + xOrder) * [nonzeroComponentIdx] + [
                    tuple(range(tensorDims[-1]))])  # prepend copies of the Hill component index for xOrder derivatives
                DH[nonzeroIdx] = DH_nonzero
                return DH
            else:
                return DH_nonzero

        elif parameterOrder == 2:  # 2 partial derivatives w.r.t. parameters.

            if not diffIndex:  # no optional argument means return all component parameter derivatives twice (i.e. all parameters except gamma)
                from itertools import product
                diffIndex = []  # initialize a list of parameter pairs
                for idx in range(self.nProduction):
                    parameterSlice = range(self.productionParameterIndexRange[idx],
                                           self.productionParameterIndexRange[idx + 1])
                    diffIndex += list(product(parameterSlice, parameterSlice))

            parameterComponentIndex = [
                ezcat(self.parameter_to_production_index(idx[0]), self.parameter_to_production_index(idx[1])[1]) for idx
                in diffIndex]
            # a list of triples stored as numpy arrays of the form (i,j,k) where lambda_j, lambda_k are both parameters for H_i

            if xOrder == 0:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].diff2(xProduction[idx[0]],
                                                                                 parameterByComponent[idx[0]],
                                                                                 idx[1:]),
                             parameterComponentIndex)))  # evaluate vector of second order pure partial derivatives for Hill productionComponents

            elif xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].dxdiff2(xProduction[idx[0]],
                                                                                   parameterByComponent[idx[0]],
                                                                                   idx[1:]),
                             parameterComponentIndex)))  # evaluate vector of third order mixed partial derivatives for Hill productionComponents

            if fullTensor:
                tensorDims = (1 + xOrder) * [self.nProduction] + 2 * [self.nParameter - self.gammaIsVariable]
                DH = np.zeros(tensorDims)
                nonzeroTripleIdx = list(zip(*parameterComponentIndex))
                nonzeroComponentIdx = nonzeroTripleIdx[0]
                nonzeroLambdaIdx = [tuple(
                    self.component_to_parameter_index(nonzeroComponentIdx[j], nonzeroTripleIdx[1][j]) - int(
                        self.gammaIsVariable) for j in
                    range(len(nonzeroComponentIdx))),
                    tuple(self.component_to_parameter_index(nonzeroComponentIdx[j],
                                                            nonzeroTripleIdx[2][j]) - int(self.gammaIsVariable) for j in
                          range(len(nonzeroTripleIdx[0])))
                ]

                nonzeroIdx = tuple((1 + xOrder) * [
                    nonzeroComponentIdx] + nonzeroLambdaIdx)  # prepend copies of the Hill component index for xOrder derivatives
                DH[nonzeroIdx] = DH_nonzero
                return DH
                # return DH, DH_nonzero, nonzeroIdx
            else:
                return DH_nonzero

    @verify_call
    def dx(self, x, parameter):
        """Return the derivative as a gradient vector evaluated at x in R^n and p in R^m"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        Df = np.zeros(self.nState, dtype=float)
        diffInteraction = self.diff_production(x, parameter,
                                               1)  # evaluate derivative of production interaction function (outer term in chain rule)
        DHillComponent = np.array(
            list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, x[self.productionIndex],
                     parameterByComponent)))  # evaluate vector of partial derivatives for production Hill Components (inner term in chain rule)
        Df[
            self.productionIndex] = diffInteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
        Df[0] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
        return Df

    @verify_call
    def diff(self, x, parameter, diffIndex=None):
        """Evaluate the derivative of a Hill coordinate with respect to a parameter at the specified local index.
           The parameter must be a variable parameter for one or more HillComponents."""

        if diffIndex is None:  # return the full gradient with respect to parameters as a vector in R^m
            return np.array([self.diff(x, parameter, diffIndex=k) for k in range(self.nParameter)])

        else:  # return a single partial derivative as a scalar
            if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
                return -x[0]
            else:  # First obtain a local index in the HillComponent for the differentiation variable
                diffComponent = np.searchsorted(self.productionParameterIndexRange,
                                                diffIndex + 0.5) - 1  # get the component which contains the differentiation variable. Adding 0.5
                # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
                diffParameterIndex = diffIndex - self.productionParameterIndexRange[
                    diffComponent]  # get the local parameter index in the HillComponent for the differentiation variable

                # Now evaluate the derivative through the HillComponent and embed into tangent space of R^n
                gamma, parameterByComponent = self.parse_parameters(parameter)
                xProduction = x[
                    self.productionIndex]  # extract only the coordinates of x that contribute to the production of this HillCoordinate
                diffInteraction = self.diff_production(x, parameter, 1,
                                                       diffIndex=diffComponent)  # evaluate outer term in chain rule
                dpH = self.productionComponents[diffComponent].diff(xProduction[diffComponent],
                                                                    parameterByComponent[
                                                                        diffComponent],
                                                                    diffParameterIndex)  # evaluate inner term in chain rule
                return diffInteraction * dpH

    @verify_call
    def dx2(self, x, parameter):
        """Return the second derivative (Hessian matrix) with respect to the state variable vector evaluated at x in
        R^n and p in R^m as a K-by-K matrix"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        xProduction = x[
            self.productionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        D2f = np.zeros(2 * [self.nState], dtype=float)

        D2HillComponent = np.array(
            list(map(lambda H_k, x_k, p_k: H_k.dx2(x_k, p_k), self.productionComponents, x[self.productionIndex],
                     parameterByComponent)))
        # evaluate vector of second partial derivatives for production Hill Components
        nSummand = len(self.productionType)  # number of summands

        if nSummand == 1:  # production is all sum
            D2Nonlinear = np.diag(D2HillComponent)
        # TODO: Adding more special cases for 2 and even 3 summand production types will speed up the computation quite a bit.
        #       This should be done if this method ever becomes a bottleneck.

        else:  # production interaction function contributes derivative terms via chain rule

            # compute off diagonal terms in Hessian matrix by summand membership
            productionComponentValues = self.evaluate_production_components(x, parameter)
            summandValues = self.evaluate_summand(productionComponentValues)
            fullProduct = np.prod(summandValues)
            DxProducts = fullProduct / summandValues  # evaluate all partials using only nSummand-many multiplies

            # initialize Hessian matrix and set diagonal terms
            DxProductsByComponent = np.array([DxProducts[self.summand_index(k)] for k in range(self.nProduction)])
            D2Nonlinear = np.diag(D2HillComponent * DxProductsByComponent)

            # set off diagonal terms of Hessian by summand membership and exploiting symmetry
            DxxProducts = np.outer(DxProducts,
                                   1.0 / summandValues)  # evaluate all second partials using only nSummand-many additional multiplies.
            # Only the cross-diagonal terms of this matrix are meaningful.

            offDiagonal = np.zeros_like(D2Nonlinear)  # initialize matrix of mixed partials (off diagonal terms)
            for row in range(
                    nSummand):  # compute Hessian of production interaction function (outside term of chain rule)
                for col in range(row + 1, nSummand):
                    offDiagonal[np.ix_(self.summand[row], self.summand[col])] = offDiagonal[
                        np.ix_(self.summand[col], self.summand[row])] = DxxProducts[row, col]

            DHillComponent = np.array(
                list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, x[self.productionIndex],
                         parameterByComponent)))  # evaluate vector of partial derivatives for Hill productionComponents
            mixedPartials = np.outer(DHillComponent,
                                     DHillComponent)  # mixed partial matrix is outer product of gradients!
            D2Nonlinear += offDiagonal * mixedPartials
            # NOTE: The diagonal terms of offDiagonal are identically zero for any interaction type which makes the
            # diagonal terms of mixedPartials irrelevant
        D2f[np.ix_(self.productionIndex, self.productionIndex)] = D2Nonlinear
        return D2f

    @verify_call
    def dxdiff(self, x, parameter, diffIndex=None):
        """Return the mixed second derivative with respect to x and a scalar parameter evaluated at x in
        R^n and p in R^m as a gradient vector in R^K. If no parameter index is specified this returns the
        full second derivative as the m-by-K Hessian matrix of mixed partials"""

        if diffIndex is None:
            return np.column_stack(
                list(map(lambda idx: self.dxdiff(x, parameter, idx), range(self.nParameter))))

        else:
            D2f = np.zeros(self.nState, dtype=float)  # initialize derivative as a vector

            if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
                D2f[0] = -1
                return D2f

            gamma, parameterByComponent = self.parse_parameters(parameter)
            xProduction = x[
                self.productionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
            diffComponent = np.searchsorted(self.productionParameterIndexRange,
                                            diffIndex + 0.5) - 1  # get the component which contains the differentiation variable. Adding 0.5
            # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
            diffParameterIndex = diffIndex - self.productionParameterIndexRange[
                diffComponent]  # get the local parameter index in the HillComponent for the differentiation variable

            # initialize inner terms of chain rule derivatives of f
            # DH = np.zeros(2 * [self.nProduction])  # initialize diagonal tensor for DxH as a 2-tensor
            DHillComponent = np.array(
                list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, xProduction,
                         parameterByComponent)))  # 1-tensor of partials for DxH
            # np.einsum('ii->i', DH)[:] = DHillComponent  # build the diagonal tensor for DxH
            DpH = self.productionComponents[diffComponent].diff(xProduction[diffComponent],
                                                                parameterByComponent[diffComponent],
                                                                diffParameterIndex)

            D2H = self.productionComponents[diffComponent].dxdiff(xProduction[diffComponent],
                                                                  parameterByComponent[diffComponent],
                                                                  diffParameterIndex)  # get the correct mixed partial derivative of H_k

            # initialize outer terms of chain rule derivatives of f
            Dp = self.diff_production(x, parameter, 1)[diffComponent]  # k^th index of Dp(H) is a 0-tensor (scalar)
            D2p = self.diff_production(x, parameter, 2)[diffComponent]  # k^th index of D^2p(H) is a 1-tensor (vector)

            D2f[self.productionIndex] += DpH * DHillComponent * D2p  # contribution from D2(p(H))*D_parm(H)*DxH
            D2f[self.productionIndex[diffComponent]] += D2H * Dp  # contribution from Dp(H)*D_parm(DxH)
            return D2f

    @verify_call
    def diff2(self, x, parameter, *diffIndex, fullTensor=True):
        """Return the second derivative with respect to parameters specified evaluated at x in
        R^n and p in R^m as a Hessian matrix. If no parameter index is specified this returns the
        full second derivative as the m-by-m Hessian matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DlambdaH = self.diff_production_component(x, parameter, [0, 1], fullTensor=fullTensor)
        D2lambdaH = self.diff_production_component(x, parameter, [0, 2], fullTensor=fullTensor)

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ik,kl,ij', D2p, DlambdaH, DlambdaH)
            term2 = np.einsum('i,ijk', Dp, D2lambdaH)
            DpoH = term1 + term2
        else:
            raise ValueError

        if self.gammaIsVariable:
            D2lambda = np.zeros(2 * [self.nParameter])
            D2lambda[1:, 1:] = DpoH
            return D2lambda
        else:
            return DpoH

    @verify_call
    def dx3(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector evaluated at x in
        R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_production_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DxxH = self.diff_production_component(x, parameter, [2, 0], fullTensor=fullTensor)
        DxxxH = self.diff_production_component(x, parameter, [3, 0], fullTensor=fullTensor)

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor
        D3p = self.diff_production(x, parameter, 3)  # 3-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ikq,qr,kl,ij', D3p, DxH, DxH, DxH)
            term2 = np.einsum('ik,kl,ijq', D2p, DxH, DxxH)
            term3 = np.einsum('ik,ij,klq', D2p, DxH, DxxH)
            term4 = np.einsum('il,lq,ijk', D2p, DxH, DxxH)
            term5 = np.einsum('i, ijkl', Dp, DxxxH)
            return term1 + term2 + term3 + term4 + term5
        else:  # this code is the faster version but it is not quite correct. The .multiply method needs to be combined appropriately with
            # tensor reshaping.

            return D3p * DxH * DxH * DxH + 3 * D2p * DxH * DxxH + Dp * DxxxH

    @verify_call
    def dx2diff(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector (twice) and then the parameter
        (once) evaluated at x in R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_production_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DxxH = self.diff_production_component(x, parameter, [2, 0], fullTensor=fullTensor)
        DlambdaH = self.diff_production_component(x, parameter, [0, 1],
                                                  fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal Km 2-tensor
        Dlambda_xH = self.diff_production_component(x, parameter, [1, 1],
                                                    fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal KKm 3-tensor
        Dlambda_xxH = self.diff_production_component(x, parameter, [2, 1],
                                                     fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal KKKm 4-tensor

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor
        D3p = self.diff_production(x, parameter, 3)  # 3-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ikq,qr,kl,ij', D3p, DlambdaH, DxH, DxH)
            term2 = np.einsum('ik,kl,ijq', D2p, DxH, Dlambda_xH)
            term3 = np.einsum('ik,ij,klq', D2p, DxH, Dlambda_xH)
            term4 = np.einsum('il,lq,ijk', D2p, DlambdaH, DxxH)
            term5 = np.einsum('i, ijkl', Dp, Dlambda_xxH)
            DpoH = term1 + term2 + term3 + term4 + term5
        else:
            raise ValueError

        if self.gammaIsVariable:
            Dlambda_xx = np.zeros(2 * [self.nState] + [self.nParameter])
            Dlambda_xx[:, :, 1:] = DpoH
            return Dlambda_xx
        else:
            return DpoH

    @verify_call
    def dxdiff2(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector (once) and the parameters (twice)
        evaluated at x in R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_production_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DlambdaH = self.diff_production_component(x, parameter, [0, 1], fullTensor=fullTensor)  # Km 2-tensor
        Dlambda_xH = self.diff_production_component(x, parameter, [1, 1], fullTensor=fullTensor)  # KKm 3-tensor
        D2lambdaH = self.diff_production_component(x, parameter, [0, 2], fullTensor=fullTensor)
        D2lambda_xH = self.diff_production_component(x, parameter, [1, 2], fullTensor=fullTensor)  # KKKm 4-tensor

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor
        D3p = self.diff_production(x, parameter, 3)  # 3-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ikq,qr,kl,ij', D3p, DlambdaH, DlambdaH, DxH)
            term2 = np.einsum('ik,kl,ijq', D2p, DlambdaH, Dlambda_xH)
            term3 = np.einsum('ik,ij,klq', D2p, DxH, D2lambdaH)
            term4 = np.einsum('il,lq,ijk', D2p, DlambdaH, Dlambda_xH)
            term5 = np.einsum('i, ijkl', Dp, D2lambda_xH)
            DpoH = term1 + term2 + term3 + term4 + term5
        else:
            raise ValueError

        if self.gammaIsVariable:
            D2lambda_x = np.zeros([self.nState] + 2 * [self.nParameter])
            D2lambda_x[:, 1:, 1:] = DpoH
            return D2lambda_x
        else:
            return DpoH

    def set_production(self, parameter, productionSign):
        """Return a list of Hill functions contributing to the production term of this HillCoordinate"""

        def row2dict(row):
            """convert ordered row of parameter matrix to kwarg"""
            return {PARAMETER_NAMES[j]: row[j] for j in range(4) if
                    not np.isnan(row[j])}

        # set up production Hill component functions
        if self.nProduction == 1:
            productionComponents = [HillComponent(productionSign[0], **row2dict(parameter))]
        else:
            productionComponents = [HillComponent(productionSign[k], **row2dict(parameter[k, :])) for k in
                                    range(self.nProduction)]  # list of ordered HillComponents for the production term

        # get a list of the number of variable parameters for each component in the production term.
        if self.nProduction == 1:  # production function consists of a single Hill function
            nParameterByProductionIndex = list(
                map(lambda j: np.count_nonzero(np.isnan(self.parameterValues)), range(self.nProduction)))
        else:  # production consists of multiple Hill functions
            nParameterByProductionIndex = list(
                map(lambda j: np.count_nonzero(np.isnan(self.parameterValues[j, :])), range(self.nProduction)))

        # get a list of endpoints for the concatenated parameter vector (linearly indexed) for each production component
        productionParameterIndexRange = np.cumsum([self.gammaIsVariable] + nParameterByProductionIndex)
        # endpoints for concatenated parameter vector split by production component. This is a
        # vector of length K+1. The kth component parameters are the slice productionParameterIndexRange[k:k+1] for k = 0...K-1

        return productionComponents, nParameterByProductionIndex, productionParameterIndexRange

    def set_summand(self):
        """Return the list of lists containing the summand indices defined by the production type.
        EXAMPLE:
            productionType = [2,1,3,1] returns the index partition [[0,1], [2], [3,4,5], [6]]"""

        sumEndpoints = np.insert(np.cumsum(self.productionType), 0,
                                 0)  # summand endpoint indices including initial zero
        localIndex = list(range(self.nProduction))
        return [localIndex[sumEndpoints[i]:sumEndpoints[i + 1]] for i in range(len(self.productionType))]

    def eq_interval(self, parameter=None):
        """Return a closed interval which must contain the projection of any equilibrium onto this coordinate"""

        if parameter is None:
            # all parameters are fixed
            # TODO: This should only require all ell, delta, and gamma variables to be fixed.
            minProduction = self.evaluate_production_interaction(
                [H.ell for H in self.productionComponents]) / self.gamma
            maxProduction = self.evaluate_production_interaction(
                [H.ell + H.delta for H in self.productionComponents]) / self.gamma

        else:
            # some variable parameters are passed in a vector containing all parameters for this Hill Coordinate
            gamma, parameterByComponent = self.parse_parameters(parameter)
            rectangle = np.row_stack(
                list(map(lambda H, parm: H.image(parm), self.productionComponents, parameterByComponent)))
            minProduction = self.evaluate_production_interaction(
                rectangle[:, 0]) / gamma  # min(f) = p(ell_1, ell_2,...,ell_K)
            maxProduction = self.evaluate_production_interaction(
                rectangle[:, 1]) / gamma  # max(f) = p(ell_1 + delta_1,...,ell_K + delta_K)

        return [minProduction, maxProduction]


class HillModel:
    """Define a Hill model as a vector field describing the derivatives of all state variables. The i^th coordinate
    describes the derivative of the state variable, x_i, as a function of x_i and the state variables influencing
    its production nonlinearly represented as a HillCoordinate. The vector field is defined coordinate-wise as a vector
    of HillCoordinate instances."""

    def __init__(self, gamma, parameter, productionSign, productionType, productionIndex):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^n of linear decay rates
            parameter - A length n list of K_i-by-4 parameter arrays
                    Note: If K_i = 1 then productionSign[i] should be a vector, not a matrix i.e. it should have shape
                    (4,) as opposed to (1,4). If the latter case then the result will be squeezed since otherwise HillCoordinate
                    will throw an exception during construction of that coordinate.
            productionSign - A length n list of lists in F_2^{K_i}
            productionType - A length n list of length q_i lists describing an integer partitions of K_i
            productionIndex - A length n list whose i^th element is a length K_i list of global indices for the nonlinear
                interactions for node i. These are specified in any order as long as it is the same order used for productionSign
                and the rows of parameter. IMPORTANT: The exception to this occurs if node i has a self edge. In this case i must appear as the first
                index."""

        # TODO: Class constructor should not do work!
        self.dimension = len(gamma)  # Dimension of vector field
        coordinateDims = [len(set(productionIndex[j] + [j])) for j in range(self.dimension)]
        self.coordinates = [HillCoordinate(np.squeeze(parameter[j]), productionSign[j],
                                           productionType[j], coordinateDims[j], gamma=gamma[j]) for j in
                            range(
                                self.dimension)]  # A list of HillCoordinates specifying each coordinate of the vector field
        self.productionIndex = productionIndex  # store the list of global indices which contribute to the production term of each coordinate.
        self.stateIndexByCoordinate = [self.state_variable_selection(idx) for idx in range(self.dimension)]
        # create a list of selections which slice the full state vector into subvectors for passing to evaluation functions of each coordinate.
        self.nParameterByCoordinate = list(f_i.nParameter for f_i in
                                           self.coordinates)  # number of variable parameters by coordinate
        parameterIndexEndpoints = np.insert(np.cumsum(self.nParameterByCoordinate), 0,
                                            0)  # endpoints for concatenated parameter vector by coordinate
        self.parameterIndexByCoordinate = [list(range(parameterIndexEndpoints[idx], parameterIndexEndpoints[idx + 1]))
                                           for idx in
                                           range(self.dimension)]
        self.nParameter = sum(self.nParameterByCoordinate)  # number of variable parameters for this HillModel

    def state_variable_selection(self, idx):
        """Return a list which selects the correct state subvector for the component with specified index."""

        if idx in self.productionIndex[idx]:  # This coordinate has a self edge in the GRN
            if self.productionIndex[idx][0] == idx:
                return self.productionIndex[idx]
            else:
                raise IndexError(
                    'Coordinates with a self edge must have their own index appearing first in their interaction index list')

        else:  # This coordinate has no self edge. Append its own global index as the first index of the selection slice.
            return [idx] + self.productionIndex[idx]

    def parse_parameter(self, *parameter):
        """Default parameter parsing if input is a single vector simply returns the same vector. Otherwise, it assumes
        input parameters are provided in order and concatenates into a single vector. This function is included in
        function calls so that subclasses can redefine function calls with customized parameters and overload this
        function as needed. Overloaded versions should take a variable number of numpy arrays as input and must always
        return a single numpy vector as output.

        OUTPUT: A single vector of the form:
            lambda = (gamma_1, ell_1, delta_1, theta_1, hill_1, gamma_2, ..., hill_2, ..., gamma_n, ..., hill_n).
        Any of these parameters which are not a variable for the model are simply omitted in this concatenated vector."""

        if parameter:
            parameterVector = ezcat(*parameter)

            return ezcat(*parameter)
        else:
            return np.array([])

    def unpack_parameter(self, parameter):
        """Unpack a parameter vector for the HillModel into disjoint parameter slices for each distinct coordinate"""

        return [parameter[idx] for idx in self.parameterIndexByCoordinate]

    def unpack_state(self, x):
        """Unpack a state vector for the HillModel into a length-n list of state vector slices to pass for evaluation into
         distinct coordinate. The slices are not necessarily disjoint since multiple coordinates can depend on the same
         state variable."""

        return [x[idx_slice] for idx_slice in self.stateIndexByCoordinate]

    def unpack_by_coordinate(self, x, *parameter):
        """Unpack a parameter and state vector into subvectors for each coordinate. This is called by all evaluation functions."""

        parameterByCoordinate = self.unpack_parameter(
            self.parse_parameter(*parameter))  # concatenate all parameters into
        # a vector and unpack by coordinate
        stateByCoordinate = self.unpack_state(x)  # unpack state variables by coordinate
        return stateByCoordinate, parameterByCoordinate

    @verify_call
    def __call__(self, x, *parameter):
        """Evaluate the vector field defined by this HillModel instance. This is a function of the form
        f: R^n x R^{m_1} x ... x R^{m_n} ---> R^n where the j^th Hill coordinate has m_j variable parameters. The syntax
        is f(x,p) where p = (p_1,...,p_n) is a variable parameter vector constructed by ordered concatenation of vectors
        of the form p_j = (p_j1,...,p_jK) which is also an ordered concatenation of the variable parameters associated to
        the K-HillComponents for the j^th HillCoordinate.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        return np.array(
            list(map(lambda f_i, x_i, p_i: f_i(x_i, p_i), self.coordinates, stateByCoordinate, parameterByCoordinate)))

    @verify_call
    def dx(self, x, *parameter):
        """Return the first derivative of the HillModel vector field with respect to x as a rank-2 tensor (matrix). The i-th row
        of this tensor is the differential (i.e. gradient) of the i-th coordinate of f. NOTE: This function is not vectorized. It assumes x
        and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        Dxf = np.zeros(2 * [self.dimension])  # initialize Derivative as 2-tensor of size NxN
        for (i, f_i) in enumerate(self.coordinates):
            # get gradient values for f_i and embed derivative of f_i into the full derivative of f.
            Dxf[np.ix_([i], self.stateIndexByCoordinate[
                i])] = f_i.dx(stateByCoordinate[i], parameterByCoordinate[i])
        return Dxf

    @verify_call
    def diff(self, x, *parameter, diffIndex=None):
        """Return the first derivative of the HillModel vector field with respect to a specific parameter (or the full parameter vector) as
        a vector (or matrix). In the latter case, the i-th row of this tensor is the differential of
        the i-th coordinate of f with respect to parameters. NOTE: This function is not vectorized. It assumes x
        and parameter represent a single state and parameter vector. NOTE: This function is not vectorized."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dpf = np.zeros(
                [self.dimension, self.nParameter])  # initialize Derivative as 2-tensor of size NxM
            for (i, f_i) in enumerate(self.coordinates):
                Dpf[np.ix_([i], self.parameterIndexByCoordinate[i])] = f_i.diff(stateByCoordinate[i],
                                                                                parameterByCoordinate[
                                                                                    i])  # insert derivative of this coordinate
            return Dpf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def dx2(self, x, *parameter):
        """Return the second derivative of the HillModel vector field with respect to x (twice) as a rank-3 tensor. The i-th matrix
        of this tensor is the Hessian matrix of the i-th coordinate of f. NOTE: This function is not vectorized. It assumes x
        and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        Dxf = np.zeros(3 * [self.dimension])  # initialize Derivative as 3-tensor of size NxNxN
        for (i, f_i) in enumerate(self.coordinates):
            # get second derivatives (Hessian matrices) for f_i and embed each into the full derivative of f.
            Dxf[np.ix_([i], self.stateIndexByCoordinate[
                i], self.stateIndexByCoordinate[
                           i])] = f_i.dx2(stateByCoordinate[i], parameterByCoordinate[i])
        return Dxf

    @verify_call
    def dxdiff(self, x, *parameter, diffIndex=None):
        """Return the second derivative of the HillModel vector field with respect to the state and parameter vectors (once each)
        as a rank-3 tensor. The i-th matrix of this tensor is the matrix of mixed partials of the i-th coordinate of f.
        NOTE: This function is not vectorized. It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dpxf = np.zeros(2 * [self.dimension] + [self.nParameter])  # initialize Derivative as 3-tensor of size NxNxM
            for (i, f_i) in enumerate(self.coordinates):
                Dpxf[np.ix_([i], self.stateIndexByCoordinate[i], self.parameterIndexByCoordinate[i])] = f_i.dxdiff(
                    stateByCoordinate[i], parameterByCoordinate[
                        i])  # insert derivative of this coordinate
            return Dpxf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def diff2(self, x, *parameter, diffIndex=None):
        """Return the second derivative of the HillModel vector field with respect to parameter vector (twice)
        as a rank-3 tensor. The i-th matrix of this tensor is the Hessian matrix of the i-th coordinate of f with respect
        to x. NOTE: This function is not vectorized. It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative with respect to all state and parameter vectors
            Dppf = np.zeros([self.dimension] + 2 * [self.nParameter])  # initialize Derivative as 3-tensor of size NxMxM
            for (i, f_i) in enumerate(self.coordinates):
                Dppf[np.ix_([i], self.parameterIndexByCoordinate[i], self.parameterIndexByCoordinate[i])] = f_i.diff2(
                    stateByCoordinate[i], parameterByCoordinate[
                        i])  # insert derivative of this coordinate
            return Dppf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def dx3(self, x, *parameter):
        """Return the third derivative of the HillModel vector field with respect to x (three times) as a rank-4 tensor. The i-th
        rank-3 subtensor of this tensor is the associated third derivative of the i-th coordinate of f. NOTE: This function is not vectorized.
        It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        Dxxxf = np.zeros(4 * [self.dimension])  # initialize Derivative as 4-tensor of size NxNxNxN
        for (i, f_i) in enumerate(self.coordinates):
            Dxxxf[np.ix_([i], self.stateIndexByCoordinate[i], self.stateIndexByCoordinate[i],
                         self.stateIndexByCoordinate[i])] = f_i.dx3(stateByCoordinate[i], parameterByCoordinate[
                i])  # insert derivative of this coordinate
        return Dxxxf

    @verify_call
    def dx2diff(self, x, *parameter, diffIndex=None):
        """Return the third derivative of the HillModel vector field with respect to parameters (once) and x (twice) as a rank-4 tensor. The i-th
        rank-3 subtensor of this tensor is the associated third derivative of the i-th coordinate of f. NOTE: This function is not vectorized.
        It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dpxxf = np.zeros(
                3 * [self.dimension] + [self.nParameter])  # initialize Derivative as 4-tensor of size NxNxNxM

            for (i, f_i) in enumerate(self.coordinates):
                Dpxxf[np.ix_([i], self.stateIndexByCoordinate[i], self.stateIndexByCoordinate[i],
                             self.parameterIndexByCoordinate[i])] = f_i.dx2diff(stateByCoordinate[i],
                                                                                parameterByCoordinate[
                                                                                    i])  # insert derivative of this coordinate
            return Dpxxf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def dxdiff2(self, x, *parameter, diffIndex=None):
        """Return the third derivative of the HillModel vector field with respect to parameters (twice) and x (once) as a rank-4 tensor. The i-th
        rank-3 subtensor of this tensor is the associated third derivative of the i-th coordinate of f. NOTE: This function is not vectorized.
        It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dppxf = np.zeros(
                2 * [self.dimension] + 2 * [self.nParameter])  # initialize Derivative as 4-tensor of size NxNxMxM
            for (i, f_i) in enumerate(self.coordinates):
                Dppxf[np.ix_([i], self.stateIndexByCoordinate[i], self.parameterIndexByCoordinate[i],
                             self.parameterIndexByCoordinate[i])] = f_i.dxdiff2(
                    stateByCoordinate[i], parameterByCoordinate[i])  # insert derivative of this coordinate
            return Dppxf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    def radii_uniqueness_existence(self, equilibrium, *parameter):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        def F(x):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, *parameter)

        def DF(x):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, *parameter)

        DF_x = DF(equilibrium)
        D2F_x = self.dx2(equilibrium, *parameter)
        A = np.linalg.inv(DF_x)
        Y_bound = np.linalg.norm(A @ F(equilibrium))
        Z0_bound = np.linalg.norm(np.identity(len(equilibrium)) - A @ DF_x)

        def operator_norm(T):
            # takes a 3D tensor and returns the operator norm - any size
            norm_T = np.max(np.max(np.sum(np.abs(D2F_x), axis=2), axis=1), axis=0)
            return norm_T

        Z2_bound = np.linalg.norm(A) * operator_norm(D2F_x)
        if Z2_bound < 1e-16:
            Z2_bound = 1e-8  # in case the Z2 bound is too close to zero, we increase it a bit
        delta = 1 - 4 * (Z0_bound + Y_bound) * Z2_bound
        if delta < 0 or np.isnan(delta):
            return 0, 0
        max_rad = np.minimum((1 + np.sqrt(delta)) / (2 * Z2_bound),
                             0.1)  # approximations are too poor to extend further
        min_rad = (1 - np.sqrt(delta)) / (2 * Z2_bound)
        return max_rad, min_rad

    def find_equilibria(self, gridDensity, *parameter, uniqueRootDigits=5, eqBound=None):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        # TODO: Include root finding method as kwarg
        parameterByCoordinate = self.unpack_parameter(
            self.parse_parameter(*parameter))  # unpack variable parameters by component

        def F(x):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, *parameter)

        def DF(x):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, *parameter)

        def eq_is_positive(equilibrium):
            """Return true if and only if an equlibrium is positive"""
            return np.all(equilibrium > 0)

        # def radii_uniqueness_existence(equilibrium):
        #     DF_x = DF(equilibrium)
        #     D2F_x = self.dx2(equilibrium, *parameter)
        #     A = np.linalg.inv(DF_x)
        #     Y_bound = np.linalg.norm(A @ F(equilibrium))
        #     Z0_bound = np.linalg.norm(np.identity(len(equilibrium)) - A @ DF_x)
        #     Z2_bound = np.linalg.norm(A) * np.linalg.norm(D2F_x)
        #     if Z2_bound < 1e-16:
        #         Z2_bound = 1e-8  # in case the Z2 bound is too close to zero, we increase it a bit
        #     delta = 1 - 4 * (Z0_bound + Y_bound) * Z2_bound
        #     if delta < 0:
        #         return 0, 0
        #     max_rad = (1 + np.sqrt(delta)) / (2 * Z2_bound)
        #     min_rad = (1 - np.sqrt(delta)) / (2 * Z2_bound)
        #     return max_rad, min_rad

        # build a grid of initial data for Newton algorithm
        if eqBound is None:  # use the trivial equilibrium bounds
            eqBound = np.array(
                list(map(lambda f_i, parm: f_i.eq_interval(parm), self.coordinates, parameterByCoordinate)))
        coordinateIntervals = [np.linspace(*interval, num=gridDensity) for interval in eqBound]
        evalGrid = np.meshgrid(*coordinateIntervals)
        X = np.column_stack([G_i.flatten() for G_i in evalGrid])

        # Apply rootfinding algorithm to each initial condition
        solns = list(
            filter(lambda root: root.success and eq_is_positive(root.x), [find_root(F, DF, x, diagnose=True)
                                                                          for x in
                                                                          X]))  # return equilibria which converged
        if solns:
            equilibria = np.row_stack([root.x for root in solns])  # extra equilibria as vectors in R^n
            equilibria = np.unique(np.round(equilibria, uniqueRootDigits), axis=0)  # remove duplicates
            # equilibria = np.unique(np.round(equilibria/10**np.ceil(log(equilibria)),
            #                                uniqueRootDigits)*10**np.ceil(log(equilibria)), axis=0)

            if len(equilibria) > 1:
                all_equilibria = equilibria
                radii = np.zeros(len(all_equilibria))
                unique_equilibria = all_equilibria
                for i in range(len(all_equilibria)):
                    equilibrium = all_equilibria[i]
                    max_rad, min_rad = self.radii_uniqueness_existence(equilibrium, *parameter)
                    radii[i] = max_rad

                radii2 = radii
                for i in range(len(all_equilibria)):
                    equilibrium1 = all_equilibria[i, :]
                    radius1 = radii[i]
                    j = i + 1
                    while j < len(radii2):
                        equilibrium2 = unique_equilibria[j, :]
                        radius2 = radii2[j]
                        if np.linalg.norm(equilibrium1 - equilibrium2) < np.maximum(radius1, radius2):
                            # remove one of the two from
                            unique_equilibria = np.delete(unique_equilibria, j, 0)
                            radii2 = np.delete(radii2, j, 0)
                        else:
                            j = j + 1
                equilibria = unique_equilibria
            return np.row_stack([find_root(F, DF, x) for x in equilibria])  # Iterate Newton again to regain lost digits
        else:
            return None

    @verify_call
    def saddle_though_arc_length_cont(self, equilibrium, parameter, parameter_bound):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        def F(x, param):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, param)

        def DF(x, param):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, param)

        def D_lambda_F(x, param):
            return self.diff(x, param)

        def Jac(x, param):
            return np.array([DF(x, param), D_lambda_F(x, param)])

        def Newton_loc(x, param):
            iter = 0
            while np.linalg.norm(F(x, param)) < 10 ** -14 and iter < 20:
                iter = iter + 1
                step = np.linalg.solve(DF(x, param), F(x, param))
                x = x - step[:-1]
                param = param - step[-1]
            return x, param

        def arc_length_step(x, param, direction):
            step_size = 10 ** -6
            tangent = linalg.null_space(Jac(x, param))
            if tangent[-1] * direction < 0:
                tangent = -1 * tangent
            new_x = x + step_size * tangent[:-1]
            new_par = param + step_size * tangent[-1]
            [x, param] = Newton_loc(new_x, new_par)
            if np.abs(np.log(np.linalg.det(DF(x, param)))) > 10 and np.linalg.norm(D_lambda_F(x, param)) > 0.9:
                is_saddle = True
            else:
                is_saddle = False
            return x, param, is_saddle

        if parameter < parameter_bound:
            direction = +1
        else:
            direction = -1
        is_saddle = False
        while not is_saddle and (parameter - parameter_bound) * direction < 0:
            equilibrium, parameter, is_saddle = arc_length_step(equilibrium, parameter, direction)

        return equilibrium, parameter, is_saddle
