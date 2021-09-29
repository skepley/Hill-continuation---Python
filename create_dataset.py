"""
Some code to create and manage a huge data set of stored parameters, created a priori, then accessed as needed


Author: Elena Queirolo
Created: 1st March 2021
Modified: 1st March 2021
"""
from toggle_switch_heat_functionalities import *
import random
from scipy.optimize import minimize
from datetime import datetime


def create_dataset(n_parameters: int, assign_region, n_parameter_region: int, size_dataset: int, file_name=None):
    """
    create_dataset uses the information concerning a Hill model and its number of parameter regions to create a Fisher
    distribution spanning the parameter space such that all parameter regions are similarly sampled.
    Once the Fisher distribution is found, a sample of the distribution is taken. All information is then stored in a
    npz file.

    At the moment it ony works for the Toggle Switch

    INPUT
    n_parametes         interger, number of parameters of the semi-algebraic set
    assign_region       function, takes as input a parameter of an array of parameters and returns (an array of) region(s)
    n_parameter_region  integer, how many parameter regions are associated to the model
    size_dataset        integer, size of the output dataset
    file_name           string, name of the saved file

    OUTPUT
    file_name           name of the saved file

    helper functions:
    region_sampler
    DSGRN_parameter_region
    generate_data_from_coefs
    """
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        file_name = f"{timestamp}"+'.npz'

    sampler_global = region_sampler()
    sampler_fisher = region_sampler_fisher()
    def sampler_score_fisher(fisher_coefficients):

        data_sample = sampler_fisher(fisher_coefficients[:n_parameters], fisher_coefficients[n_parameters:], 5*10**3)
        data_region = assign_region(data_sample)
        # TODO: link to DSGRN, this takes as input a matrix of parameters par[1:n_pars,1:size_sample], and returns a
        # vector data_region[1:size_sample], such that data_region[i] tells us which region par[:, i] belongs to
        # data_region goes from 0 to n_parameter_region -1
        counter = np.zeros(n_parameter_region)
        for iter_loc in range(n_parameter_region):
            counter[iter_loc] = np.count_nonzero(data_region == iter_loc)
        score = 1 - np.min(counter)/np.max(counter)
        # print(score) # lowest score is best score!
        return score  # score must be minimized

    def sampler_score(normal_coefficients):

        data_sample = sampler_global(normal_coefficients[:n_parameters], normal_coefficients[n_parameters:], 5*10**3)
        data_region = assign_region(data_sample)
        # TODO: link to DSGRN, this takes as input a matrix of parameters par[1:n_pars,1:size_sample], and returns a
        # vector data_region[1:size_sample], such that data_region[i] tells us which region par[:, i] belongs to
        # data_region goes from 0 to n_parameter_region -1
        counter = np.zeros(n_parameter_region)
        for iter_loc in range(n_parameter_region):
            counter[iter_loc] = np.count_nonzero(data_region == iter_loc)
        score = 1 - np.min(counter)/np.max(counter)
        # print(score) # lowest score is best score!
        return score  # score must be minimized

    size_coef = n_parameters*(1+n_parameters)
    # for fisher  size_coef = 2*n_parameters
    coefficients = np.abs(np.random.normal(size=size_coef))
    for i in range(100):
        other_random_coefs = np.abs(np.random.normal(size=size_coef))
        if sampler_score(other_random_coefs) < sampler_score(coefficients):
            coefficients = other_random_coefs
    print('Random initial condition chosen to the best of what random can give us')
    print('Initial score', -sampler_score(coefficients) + 1)
    optimal_coefs = minimize(sampler_score, coefficients, method='nelder-mead')
    print(optimal_coefs.message)
    if optimal_coefs.success is False:
        print('The convergence failed, but the ration between worst region and best region is', -optimal_coefs.fun+1,
              ', where this is 1 if they have the same number of samples')
    optimal_coef = optimal_coefs.x
    # data = sampler_global(optimal_coef[:n_parameters], optimal_coef[n_parameters:], size_dataset)
    # parameter_region = DSGRN_parameter_region(f, data)
    # np.savez(file_name, optimal_coef=optimal_coef, data=data, parameter_region=parameter_region)
    generate_data_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters)
    return file_name


def generate_data_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters):
    """
    Takes the optimal coefficients and create a dataset out of them

    INPUT
    file_name       name of output file
    optimal_coef    optimal coefficients for the Fisher distribution
    sampler_global  way to sample from the correct distribution given the optimal parameters
    size_dataset    integer, size of the wanted dataset
    """

    data = sampler_global(optimal_coef[:n_parameters], optimal_coef[n_parameters:], size_dataset)
    parameter_region = assign_region(data)
    np.savez(file_name, optimal_coef=optimal_coef, data=data, parameter_region=parameter_region)
    return file_name


def load_dataset(file_name):
    """
    Takes as input the name of the file with a parameter dataset and returns the infomration within

    OUTPUT
    data                parameter values
    parameter_region    number of the parameter region each parameter belongs to
    optimal_coef        coefficients of the appropriate distribution that have been used to create the dataset
    """
    dataset = np.load(file_name)
    return dataset.f.data, dataset.f.parameter_region, dataset.f.optimal_coef


def region_sampler_fisher():
    """
    Creates a sample from the appropriate distribution based on the coefficients given

    Returns a function that takes as input 2 coefficient vectors and the size of the requested sample and that has as
    output a sample
    """
    def fisher_distribution(c1, c2, size):
        return np.random.f(c1, c2, size)

    def many_fisher_distributions(c1_vec, c2_vec, size):
        par = np.zeros([len(c1_vec), size])
        for i in range(len(c1_vec)):
            par[i, :] = fisher_distribution(c1_vec[i], c2_vec[i], size)
        return par
    return many_fisher_distributions


def region_sampler():
    """
    Creates a sample from the appropriate normal multivariate distribution based on the coefficients given

    Returns a function that takes as input 2 coefficient vectors and the size of the requested sample and that has as
    output a sample
    """

    def multivariate_normal_distributions(c1_vec, c2_vec, size):
        # par = np.zeros([len(c1_vec), size])
        mean = c1_vec
        dim = len(mean)
        cov = np.reshape(c2_vec, (dim, dim))
        x = np.random.multivariate_normal(mean, cov, size)
        par = np.square(x).T
        # square ensures it's positive
        return par
    return multivariate_normal_distributions


def create_dataset_ToggleSwitch(size_dataset, namefile=None, boolAppend=False):
    alpha = np.random.uniform(0, 3, size_dataset)
    beta = np.random.uniform(0, 3, size_dataset)
    parameters = np.array([fiber_sampler(alpha[j], beta[j]) for j in range(size_dataset)])
    parameter_region = associate_parameter_regionTS(alpha, beta)
    if namefile is None:
        namefile = f"ToggleSwitchDataset"
    np.savez(namefile, alpha=alpha, beta=beta, parameters=parameters, parameter_region=parameter_region)
    return


"""def readTS(file_name=None):
    if file_name is None:
        file_name = f"ToggleSwitchDataset.npz"
    dataset = np.load(file_name)
    return dataset.f.alpha, dataset.f.beta, dataset.f.parameters, dataset.f.parameter_region"""


def subsample_data_by_region(n_sample, region, alpha, beta, parameters, parameter_region):
    idx = parameter_region.index(region)
    if len(idx) < n_sample:
        raise Exception("Not enough samples to go by")
    sample_idx = idx[random.sample(range(len(idx)), k=n_sample)]
    loc_alpha = alpha[sample_idx]
    loc_beta = beta[sample_idx]
    loc_parameters = parameters[sample_idx, :]
    loc_parameter_region = parameter_region[sample_idx]
    return loc_alpha, loc_beta, loc_parameters, loc_parameter_region


def subsample_data_by_bounds(n_sample, alpha_min, alpha_max, beta_min, beta_max, alpha, beta, parameters, parameter_region):
    idx = np.nonzero((alpha > alpha_min) * (alpha < alpha_max)*(beta > beta_min) * (beta < beta_max))
    if len(idx) < n_sample:
        raise Exception("Not enough samples to go by")
    sample_idx = idx[random.sample(range(len(idx)), k=n_sample)]
    loc_alpha = alpha[sample_idx]
    loc_beta = beta[sample_idx]
    loc_parameters = parameters[sample_idx, :]
    loc_parameter_region = parameter_region[sample_idx]
    return loc_alpha, loc_beta, loc_parameters, loc_parameter_region


def associate_parameter_regionTS(alpha, beta):
    axes_1 = np.zeros_like(alpha)
    axes_2 = np.zeros_like(alpha)

    axes_1[alpha < 1] = 0
    axes_1[np.logical_and(alpha >= 1, alpha < 2)] = 1
    axes_1[alpha >= 2] = 2

    axes_2[beta < 1] = 0
    axes_2[np.logical_and(beta >= 1, beta < 2)] = 1
    axes_2[beta >= 2] = 2

    matrix_region = axes_1 * 3 + axes_2
    return matrix_region


def DSGRN_parameter_regionTS(parameter):
    # warnings.warn("This function is ONLY CODED FOR THE TOGGLE SWITCH")
    alpha, beta = parameter_to_DSGRN_coord(parameter.T)
    return associate_parameter_regionTS(alpha, beta)


def subsample(file_name, size_subsample):
    data, regions, coefs = load_dataset(file_name)
    size_data = np.size(data, 1)
    if size_subsample > size_data:
        stopHere
    index_random = np.random.randint(0, size_data, size_subsample)
    data_subsample = data[:, index_random]
    region_Subsample = regions[index_random]
    return data_subsample, region_Subsample, coefs


def region_subsample(file_name, region_number, size_subsample):
    data, regions, coefs = load_dataset(file_name)
    subindex_selection, = np.where(regions == region_number)
    data = data[:, subindex_selection]
    size_data = np.size(data, 1)
    if size_subsample > size_data:
        stopHere
    index_random = np.random.randint(0, size_data, size_subsample)
    data_subsample = data[:, index_random]
    return data_subsample, coefs


def create_dataset_TS(size_dataset_TS: int, name_TS=None):
    if name_TS is None:
        name_TS = 'TS_data_' + str(size_dataset_TS)+ '.npz'
    n_parameters_TS = 5
    n_regions_TS = 9
    name_TS = create_dataset(n_parameters_TS, DSGRN_parameter_regionTS, n_regions_TS, size_dataset_TS, name_TS)
    return name_TS


