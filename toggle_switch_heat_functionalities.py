from hill_model import *
from scipy.interpolate import griddata


def sampler():
    """Sample parameters for the toggle switch other than the hill coefficient. This is a nondimensionalized sampler
     so it assumes that theta_1 = theta_2 = gamma_1 = 1 and returns a vector in R^5 of the form:
     (ell_1, delta_1, gamma_1, ell_2, delta_2).
     Takes a sample anywhere."""

    # pick ell_2, delta_2
    ell_1 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    delta_1 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)

    # pick gamma_2
    gammaScale = 1 + 9 * np.random.random_sample()  # gamma scale in [1, 10]
    g = lambda x: x if np.random.randint(2) else 1 / x
    gamma_2 = g(gammaScale)  # sample in (.1, 10) to preserve symmetry between genes

    # pick ell_2, delta_2
    ellByGamma_2 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    deltaByGamma_2 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    ell_2 = ellByGamma_2 * gamma_2
    delta_2 = deltaByGamma_2 * gamma_2
    return ezcat(ell_1, delta_1, gamma_2, ell_2, delta_2)


def fiber_sampler(u, v, alpha_bar=10):
    """Samples the fiber defined by (u,v) according to the formulas presented in the -temporary- Section 4.4.1"""

    # the first component: u, relates to gamma_2, ell_2, delta_2
    if u < 1:
        ell_2 = np.random.random_sample()
        delta_2 = np.random.random_sample()
        gamma_2 = (ell_2 + delta_2) / u
    elif u < 2:
        ell_2 = np.random.random_sample()
        gamma_2 = ell_2 + np.random.random_sample()
        delta_2 = (gamma_2 - ell_2) / (u - 1)
    else:
        gamma_2 = np.random.random_sample()
        delta_2 = np.random.random_sample()
        ell_2 = gamma_2 * ((u - 2) * (alpha_bar - 1) + 1)

    # the second component: v, relates to ell_1, delta_1
    if v < 1:
        ell_1 = v * np.random.random_sample()  # sample in (0, v)
        delta_1 = v - ell_1
    elif v < 2:
        ell_1 = np.random.random_sample()
        delta_1 = (1 - ell_1) / (v - 1)
    else:
        ell_1 = (v - 2) * (alpha_bar - 1) + 1
        delta_1 = np.random.random_sample()

    return ezcat(ell_1, delta_1, gamma_2, ell_2, delta_2)


def DSGRN_coordinate(alpha, beta, alphaMax):
    """Returns the DSGRN heat map coordinates For a parameter of the form (alpha, beta) where
    alpha = ell / gamma and beta = (ell + delta) / gamma"""

    if beta < 1:  # (ell + delta)/gamma < theta
        x = beta

    elif alpha > 1:  # theta < ell/gamma
        x = 2 + (alpha - 1) / (alphaMax - 1)

    else:  # ell/gamma < theta < (ell + delta)/gamma
        x = 1 + (1 - alpha) / (beta - alpha)
    return x


def DSGRN_coordinates(alpha1, beta1, alpha2, beta2, alphaMax):
    """ take vectors of 4D coordinates and return vectors of x-coordinates and y-coordinates"""
    x = np.array(
        [DSGRN_coordinate(alpha2[j], beta2[j], alphaMax) for j in range(len(alpha2))])
    y = np.array(
        [DSGRN_coordinate(alpha1[j], beta1[j], alphaMax) for j in range(len(alpha1))])
    return x, y


def parameter_to_alpha_beta(parameterArray):
    """Return alpha/beta coordinates for the toggle switch for given parameters in R^5"""
    if is_vector(parameterArray):
        alpha1 = parameterArray[0]
        beta1 = parameterArray[0] + parameterArray[1]
        alpha2 = parameterArray[3] / parameterArray[2]
        beta2 = (parameterArray[3] + parameterArray[4]) / parameterArray[2]
    else:
        alpha1 = parameterArray[:, 0]
        beta1 = parameterArray[:, 0] + parameterArray[:, 1]
        alpha2 = parameterArray[:, 3] / parameterArray[:, 2]
        beta2 = (parameterArray[:, 3] + parameterArray[:, 4]) / parameterArray[:, 2]
    return alpha1, beta1, alpha2, beta2


def parameter_to_DSGRN_coord(parameterArray, alphaMax=None):
    """ takes a 5D parameter and returns a 2D DSGRN parameter"""
    alpha1, beta1, alpha2, beta2 = parameter_to_alpha_beta(parameterArray)
    if alphaMax is None:
        alphaMax = np.max(np.max(alpha1), np.max(alpha2))

    return DSGRN_coordinates(alpha1, beta1, alpha2, beta2, alphaMax)


def grid_lines(ax=None):
    """Add grid lines to a dsgrn coordinate plot"""
    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()

    for i in range(1, 3):
        ax.plot([i, i], [0, 3], 'k')
        ax.plot([0, 3], [i, i], 'k')
    return


def dsgrn_plot(parameterData, alphaMax=None, ax=None, **pyPlotOpts):
    """A scatter plot in DSGRN coordinates of a M-by-5 dimensional array. These are nondimensional parameters with rows
    of the form: (ell_1, delta_1, gamma_2, ell_2, delta_2)."""

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
    x, y = parameter_to_DSGRN_coord(parameterData, alphaMax)
    ax.scatter(x, y, marker='o', s=4, **pyPlotOpts)
    grid_lines()


def dsgrn_heat_plot(parameterData, colorData, alphaMax=None, ax=None, gridLines=True):
    """Produce a heat map plot of a given choice of toggle switch parameters using the specified color map data.
    ParameterData is a N-by-5 matrix where each row is a parameter of the form (ell_1, delta_1, gamma_2, ell_2, delta_2)"""

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()

    x, y = parameter_to_DSGRN_coord(parameterData, alphaMax)
    z = colorData

    # define grid.
    xGrid = np.linspace(0, 3, 100)
    yGrid = np.linspace(0, 3, 100)
    # grid the data.
    zGrid = griddata((x, y), z, (xGrid[None, :], yGrid[:, None]), method='linear')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    zmin = 0
    zmax = np.nanmax(zGrid)
    palette = plt.matplotlib.colors.LinearSegmentedColormap('jet3', plt.cm.datad['jet'], 2048)
    palette.set_under(alpha=0.0)
    plt.imshow(zGrid, extent=(0, 3, 0, 3), cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='auto',
               interpolation='bilinear')

    plt.colorbar()  # draw colorbar
    if gridLines:
        grid_lines(ax)

"""
plt.close('all')
n_sample = 500
u = 0 + 0.5 * np.random.rand(n_sample)
v = 0 + 0.5 * np.random.rand(n_sample)
parameter_full = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample)])
solutions = np.random.uniform(1, 1.5, n_sample)
fig1 = plt.figure()
dsgrn_heat_plot(parameter_full, solutions, 10)
fig2 = plt.figure()

dsgrn_plot(parameter_full, 10)
"""
# dsgrn_plot(parameter_full, 10, **{'c': 'k'})
