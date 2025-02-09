"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 4/3/2020 
"""

from ndma.coordinate import Coordinate
from ndma.activation.hill import HillActivation
from ndma.hill_model import ezcat
import numpy as np


def nan_array(m, n):
    """Return an m-by-n numpy array or vector of np.nan values"""

    # if m ==1:  # return a vector
    #     return np.array([np.nan for idx in range(n)])
    # else:  # return a matrix
    nanArray = np.empty((m, n))
    nanArray[:] = np.nan
    return nanArray


def check_symmetric(tensor):
    """Check if the tensor is symmetric or not"""

    if tensor.ndim == 2:
        return np.max(tensor - tensor.T)

    elif tensor.ndim == 3:
        perms = ['ikj', 'jik', 'jki', 'kji', 'kij']
        return [np.max(tensor - np.einsum('ijk->' + permString, tensor)) for permString in perms]


def check_equal(tensor1, tensor2):
    """check if two tensors are equal up to rounding errors"""
    return np.max(np.abs(tensor1 - tensor2))


# ============= Example 1 =============
# f(x, x0, x1, x2) = -gamma*x + H1(x1) + H2(x2) + H3(x3)
productionSign = [1, 1, 1]
productionType = [3]
nStateVariables = 4
pHill = np.array([1, 2, 4, 3])  # choose some Hill function parameters to use for all Hill functions.
parameter = nan_array(len(productionSign), 4)
productionVals = ezcat(*len(productionSign) * [pHill])
gammaVal = 10
p = ezcat(gammaVal, productionVals)
x = np.array([1, 4, 3, 2], dtype=float)
f = Coordinate(np.nan, parameter, productionSign, productionType, nStateVariables, HillActivation)

# set up callable copy of Hill function which makes up all production terms
# H = HillComponent(1, ell=pHill[0], delta=pHill[1], theta=pHill[2], hillCoefficient=pHill[3])
H = HillActivation(1)


# check f
y = f(x, p)
y_chk = -gammaVal * x[0] + H(x[1], pHill) + H(x[2], pHill) + H(x[3], pHill)


def test_Coordinate():
    x = np.array([1, 4, 3, 2], dtype=float)
    assert (abs(y-y_chk)< 10**-4)

def test_first_derivative():
    x = np.array([1, 4, 3, 2], dtype=float)
    # FIRST DERIVATIVES
    # check dx
    yx = f.dx(x, p)
    yx_chk = np.array([-gammaVal, H.dx(x[1], pHill), H.dx(x[2], pHill), H.dx(x[3], pHill)])
    assert(yx.shape[0] == 4)
    assert(check_equal(yx, yx_chk)< 10**-5)


    # check diff
    yp = f.diff(x, p)
    yp_chk = np.array([-x[0]] + [H.diff(x[1], pHill, j) for j in range(4)] + [H.diff(x[2], pHill, j) for j in range(4)] + [H.diff(x[3], pHill, j) for j in range(4)])
    assert(yp.shape[0] == 13)
    assert(check_equal(yp, yp_chk)< 10**-5)

def test_second_derivative():
    x = np.array([1, 4, 3, 2], dtype=float)
    # SECOND DERIVATIVES
    yxx = f.dx2(x, p)
    yxx_chk = np.zeros(2 * [nStateVariables])
    yxx_chk[np.ix_(range(1, nStateVariables), range(1, nStateVariables))] = np.diag([H.dx2(x[1], pHill), H.dx2(x[2], pHill), H.dx2(x[3], pHill)])
    assert np.all(yxx.shape[:] == np.array([4,4]))
    assert(check_equal(yxx, yxx_chk)< 10**-5)

    ypx = f.dxdiff(x, p)
    assert np.all(ypx.shape[:] == np.array([4, 13]))

    ypp = f.diff2(x, p)
    # print('Check diff2')
    # print(ypp.shape)
    assert np.all(ypp.shape[:] == np.array([13, 13]))
    # print('\n')


    # yxxx = f.dx3(x, p)
    # ypxx = f.dx2diff(x, p)
    # yppx = f.dxdiff2(x, p)

def test_all_other_ders():

    productionSign = [1, 1, 1]
    productionType = [3]
    x = np.array([1, 4, 3, 2], dtype=float)

    # ============= check derivatives defined by tensor contraction operations =============
    DP = f.diff_production(x, p, 1)  # 1-tensor
    D2P = f.diff_production(x, p, 2)  # 2-tensor
    D3P = f.diff_production(x, p, 3)  # 3-tensor
    DxH = f.diff_production_component(x, p, [1, 0])  # 2-tensor
    DpH = f.diff_production_component(x, p, [0, 1])  # 2-tensor
    DxxH = f.diff_production_component(x, p, [2, 0])  # 3-tensor
    DpxH = f.diff_production_component(x, p, [1, 1])  # 3-tensor
    DppH = f.diff_production_component(x, p, [0, 2])  # 3-tensor
    DppxH = f.diff_production_component(x, p, [1, 2])  # 4-tensor
    DpxxH = f.diff_production_component(x, p, [2, 1])  # 4-tensor
    DxxxH = f.diff_production_component(x, p, [3, 0])  # 4-tensor

    # ============= build all derivatives via tensor contraction operations =============
    yx2 = np.einsum('i,ij', DP, DxH)
    # doesn't work due to different dimensions, but computes the right thing
    # yx2[f.productionIndex] -= gammaVal  # equal to yx. So DP and DxH are correct
    yp2 = ezcat(-x[f.productionIndex], np.einsum('i,ij', DP, DpH))  # equal to yp. So DpH is correct
    yxx2 = np.einsum('ik,kl,ij', D2P, DxH, DxH) + np.einsum('i,ijk', DP, DxxH)  # equal to yxx. So D2P, DxxH are correct.

    # ============= I still don't think these are correct =============

    # yppx2 = np.zeros([4, 17, 17])
    # term1 = np.einsum('ikq,qr,kl,ij', D3P, DpH, DpH, DxH)
    # term2 = np.einsum('ik,kl,ijq', D2P, DpH, DpxH)
    # term3 = np.einsum('ik,ij,klq', D2P, DxH, DppH)
    # term4 = np.einsum('il,lq,ijk', D2P, DpH, DpxH)
    # term5 = np.einsum('i, ijkl', DP, DppxH)
    # yppx2[np.ix_(np.arange(4), np.arange(1,17), np.arange(1,17))] = term1 + term2 + term3 + term4 + term5
    # print(check_equal(yppx, yppx2))


    # # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
    # DxH = f.diff_component(x, p, [1, 0], fullTensor=True)
    # DxxH = f.diff_component(x, p, [2, 0], fullTensor=True)
    # DxxxH = f.diff_component(x, p, [3, 0], fullTensor=True)
    #
    # # get tensors for derivatives of p o H(x) (outer terms of chain rule)
    # Dp = f.diff_interaction(x, p, 1)  # 1-tensor
    # D2p = f.diff_interaction(x, p, 2)  # 2-tensor
    # D3p = f.diff_interaction(x, p, 3)  # 3-tensor


    parameter2 = np.copy(parameter)
    p2Vars = [[0, -1], [1, 0]]  # set n_1, and ell_2 as variable parameters
    parameter2[0, -1] = parameter[1, 0] = np.nan
    p2 = np.array([gammaVal, parameter[0, -1], parameter[1, 0]], dtype=float)
    f2 = Coordinate(np.nan, parameter2, productionSign, productionType, 3, HillActivation)  # gamma is a variable parameter too
    x = np.array([1, 2, 3])
    p2 = np.array([7,8,9,0,2,3,5,3,2,1,4,6,7])
    assert np.abs(f2(x, p2))<16
    assert np.shape(f2.dx(x, p2))[0]==3
    assert(f2.diff(x, p2, 1)==1.)
    assert np.all(np.shape(f2.dx2(x, p2))== np.array(([3,3])))

    # check that diff and dn produce equivalent derivatives
    parameter3 = np.copy(parameter)
    p3Vars = [[0, -1], [1, -1]]  # set n_1, and n_2 as variable parameters
    parameter3[0, -1] = parameter3[1, -1] = np.nan
    p3 = np.array([6,34,2,3,5,7,2,1,4,7,2,1], dtype=float)
    f3 = Coordinate(gammaVal, parameter3, productionSign, productionType, 3,
                        HillActivation)  # gamma is a variable parameter too
    # print([f3.diff(x, p3, j) for j in range(f3.nParameter)])

    # check summand evaluations
    parameter4 = np.repeat(np.nan, 12).reshape(3, 4)
    productionType = [2, 1]
    productionSign = [1, 1, -1]
    p4 = np.arange(12)
    x4 = np.array([1, 2, 3, 4])
    f4 = Coordinate(gammaVal, parameter4, productionSign, productionType, 4, HillActivation)
    # print(f4.diff_production(x4, p4, 1))

    # f4.diff_production(x,p4,1) does NOT trigger a well defined error, because it is not meant to be used directly.
    # Use diff instead
    # print(f4.diff(x4, p4))
    # print(f4.dx2(x4, p4))

    assert True

