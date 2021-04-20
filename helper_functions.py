#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:14:51 2021

@author: noahlefrancois
"""

import numpy as np
import pypico
#import time as t
import matplotlib.pyplot as plt
#import astropy.units as u
from astropy.cosmology import LambdaCDM
import emcee
#import corner
import os
import imageio
#import h5py

#Load the pico training data
pico = pypico.load_pico("jcset_py3.dat")


def get_PICO_spectrum(pars):
    """
    Function to evaluate the CAMB model emulator for a given set of parameters. 
    Much faster than the full CAMB model
    
    Input:
        pars (arr)- model parameters
        
    Output:
        tt (arr)- power spectrum values as a function of multipole moment 
            according to a pico model with the given parameters
    """
    
    H0, ombh2, omch2, omk, tau, As, ns, alpha = pars #Unpack model parameters
    #Feed input parameters as dictionary to pico
    input_dict = {"As": As,"ns": ns,"tau": tau,"ombh2":ombh2,"omch2":omch2,"H0":H0,"omk":omk}
    output_dict = pico.get(**input_dict, force=True)
    #Unpack the pico power spectrum output
    tt = output_dict['dl_TT']
    
    return tt

def get_cov_model(err, alpha):
    """
    Evaluate the covariance matrix using a model where adjacent points have a correlation 
    scaled by the parameter alpha
    
    Runtime testing:
    0.16 seconds with vectorization, 0.36 seconds with for loops
    
    Input:
        err (arr)- error bars from each point in the WMAP data
        alpha (float)- parameter controlling the correlation strength
    Output:
        C (arr)- covariance matrix of size [len(err),len(err)] for the correlated error model
    """    
    #Compute each element in the covariance matrix
    err_shift_1 = np.roll(err,-1) #Get the error shifted by one, to compute k=1 correlation terms
    #Compute the diagonal and k=1 terms
    diag_terms = err**2
    diag_k1_terms = alpha*np.abs(err[0:-1]*err_shift_1[0:-1]) #Should I take the absolute value?
    
    #Cast the terms into matrix form and combine to get the final covariance matrix
    C_diag = np.diag(np.array(diag_terms))
    C_diag_k1 = np.diag(np.array(diag_k1_terms), k=1)
    C_diag_km1 = np.diag(np.array(diag_k1_terms), k=-1)
    C = C_diag + C_diag_k1 + C_diag_km1
    
    return C

def log_likelihood_WMAP(theta, multipole, p_data, err, covariance_model):
    """
    Evaluate the chi-sq metric of a PICO fit with a neighbour-correlation model or
    an uncorrelated error model, given a set of model parameters stored in the array theta
    
    Return the log likelihood probability, which is equal to -chi_sq
    
    Input:
        theta (arr)- model params
        multipole (arr)- multipole moment data from WMAP
        p_data (arr)- power spectrum data from WMAP
        err (arr)- error bars on WMAP data points
        covariance_model (str) - controls whether to calculate chi_sq using correlated or uncorrelated error model
    Output:
        chi_sq (float) - measure of goodness of fit, -chi_sq is proportional to log likelihood probability
    """
    
    #Get model predictions from given params using PICO
    pico_tt = get_PICO_spectrum(theta) #evaluate model
    p_model = pico_tt[2:len(multipole)+2] #cut off the extra model points that extrapolate past where our multipole data ends
    
    if covariance_model == 'correlated':
        alpha = theta[7] #Get the covariance scaling parameter
        
        #Get the components of the correlated chi-sq expression
        At = np.array([p_data - p_model])
        A = np.transpose(At)
        C_inv = np.linalg.inv(get_cov_model(err, alpha))

        chi_sq = np.dot(At, np.dot(C_inv,A))[0,0] #Evaluate the matrix multiplication of chi-squared terms

    elif covariance_model == 'uncorrelated':
        #Get the components of the uncorrelated chi-sq expression
        x = np.asarray(p_data)
        y = np.asarray(p_model)
        error = np.asarray(err)
        
        chi_sq = sum((x-y)**2/error**2) #Evaluate chi-sq

    return -chi_sq

def log_prior_WMAP(theta):
    """
    Evaluate the log prior probability function given model parameters
    
    Input:
        theta (arr)- model params
    Output:
        Return 0.0 if params fall within constraints, else return -np.inf
    """
    H0, ombh2, omch2, omk, tau, As, ns, alpha = theta #Unpack model parameters
    
    #Convert units of Omega params
    h = H0/100
    Omb = ombh2/(h**2)
    Omde = omch2/(h**2)
    
    #Check that the params are allowed by our physical constraints
    if 0. <= Omb <= 1. and 0. < Omde < 1. and -1.<=alpha<=1.:
        return 0.0 # the constant doesn't matter since MCMCs only care about *ratios* of probabilities
    return -np.inf # log(0) = -inf

def log_post_WMAP_correlated(theta, multipole, p_data, err):
    """
    Evaluate the log posterior probability function given WMAP data and 
    model parameters, using correlated error model
    
    Input:
        theta (arr)- model params
        multipole (arr)- multipole moment data from WMAP
        p_data (arr)- power spectrum data from WMAP
        err (arr)- error bars on WMAP data points
    Output:
        Return log likelihood probability if params fall within constraints, else return -np.inf
        
    """
    covariance_model = 'correlated'
    
    lp = log_prior_WMAP(theta) #Evaluate log prior
    if not np.isfinite(lp):
        return -np.inf #Return -np.inf if params outside of constraints
    
    #If params inside constraints, lp = 0.0 and we reutrn the log likelihood
    return lp + log_likelihood_WMAP(theta, multipole, p_data, err, covariance_model)

def log_post_WMAP_uncorrelated(theta, multipole, p_data, err):
    """
    Evaluate the log posterior probability function given WMAP data and 
    model parameters, using uncorrelated error model
    
    Input:
        theta (arr)- model params
        multipole (arr)- multipole moment data from WMAP
        p_data (arr)- power spectrum data from WMAP
        err (arr)- error bars on WMAP data points
    Output:
        Return log likelihood probability if params fall within constraints, else return -np.inf
    """
    covariance_model = 'uncorrelated'
    
    lp = log_prior_WMAP(theta) #Evaluate log prior
    if not np.isfinite(lp):
        return -np.inf #Return -np.inf if params outside of constraints
    
    #If params inside constraints, lp = 0.0 and we reutrn the log likelihood
    return lp + log_likelihood_WMAP(theta, multipole, p_data, err, covariance_model)

def plot_SNe_sample(x_data, flat_samples, ind):
    """
    Prepare plot of SNe MCMC fit for animation
    
    Input:
        x_data (arr)- x-axis values from data
        flat_samples (arr)- parameters for each MCMC sample
        ind (int) - index for selecting one sample
    Output:
        fig_name (str) - frame name that this figure was saved under, for use in reading into animation
    """
    #Get the parameters from the specified sample, create the corresponding cosmo model, and evaluate mu
    sample = flat_samples[ind]
    H0, Om0, Ode0 = sample
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0) 
    y_model = mu_func(x_data, cosmo)
    #Plot the fit for these parameters
    plt.plot(x_data, y_model, alpha=0.01, color='red',zorder=2)
    #Save the figure to add to the animation
    fig_name = 'frame'+str(ind)+'.png'
    plt.savefig(fig_name)
    
    return fig_name

def plot_WMAP_sample(x_data, flat_samples, ind):
    """
    Prepare plot of WMAP MCMC fit for animation
    
    Input:
        x_data (arr)- x-axis values from data
        flat_samples (arr)- parameters for each MCMC sample
        ind (int) - index for selecting one sample
    Output:
        fig_name (str) - frame name that this figure was saved under, for use in reading into animation
    """
    #Get the parameters and evaluate the corresponding PICO model
    sample = flat_samples[ind]
    y_model = get_PICO_spectrum(sample)
    y_model = y_model[2:len(x_data)+2]
    #Plot the fit for these parameters
    #plt.plot(multipole,power)
    plt.plot(x_data, y_model, alpha=0.01, color='red',zorder=2)
    #Save the figure to add to the animation
    fig_name = 'frame'+str(ind)+'.png'
    plt.savefig(fig_name)
    
    return fig_name
    
def write_animation(fig_name_list, filename):
    """
    Take a series of .png frames and animate them into a .gif. Save .gif to local working directory
    
    Input:
        fig_name_list (arr): contains all of the frame filenames
        filename (str): name under which the animation will be saved
    """
    #build gif from the frames in the directory
    with imageio.get_writer(filename, mode='I') as writer:
        for fig_name in fig_name_list:
            image = imageio.imread(fig_name)
            writer.append_data(image)

    #clear the files from the directory files
    for fig_name in set(fig_name_list):
        os.remove(fig_name)
        
    print('Animation saved as ', filename)

def MCMC_animation(flat_samples, x_data, y_data, y_err, dataset, filename, N_samples):
    """
    Create an animation of the first N_samples of the MCMC fit
    
    Input:
        sampler (obj): output of emcee with multiple walkers
        x_data (arr)
        y_data (arr)
        y_err (arr): error bars on y_data
        dataset (str): controls whether to animate SNe or WMAP data
        filename (str): name under which the animation will be saved
        N_samples (int): number of samples in single chain
    """
    #Plot the original data for either SNe or WMAP
    plt.figure(figsize=(7,7))
    if dataset == 'SNe':
        flat_samples = flat_samples.get_chain(flat=True)[:N_samples]
        plt.plot(x_data, y_data,'.k')
        #plt.errorbar(x_data, y_data, yerr=y_err, linestyle = 'None', fmt='.k',mec='black',mfc='black',ecolor='grey',zorder=1)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$m-M (Mag)$')
        plt.xscale('log')
        plt.title('SCP Union 2.1 SNe Ia Data')
    if dataset =='WMAP':
        plt.plot(x_data,y_data)
        #plt.errorbar(x_data,y_data,y_err,fmt='*')
        plt.xlabel('Multipole Moment')
        plt.ylabel('Power Spectrum')
        plt.title('WMAP Satellite 9-year CMB Data')
    
    #Plot each sample and save the plot frame as a .png
    fig_name_list = []
    for ind in range(N_samples):
        if dataset == 'SNe':
            fig_name = plot_SNe_sample(x_data, flat_samples, ind)
        if dataset == 'WMAP':
            fig_name = plot_WMAP_sample(x_data, flat_samples, ind)
        #Store the frame filename
        fig_name_list.append(fig_name)
        
    #Collect the .png frames and save them as a .gif animation
    write_animation(fig_name_list, filename)
    
def run_mcmc(log_posterior, args, ndim, nwalkers, initial_pos, backend_filename, do_burn_in, plot_convergence=True, num_iter=1000, burn_in=0,thin=0):
    """
    Function which will either run MCMC with unknown burn-in time just until convergence (do_burn_in=True) 
    OR with known burn-in, thinning, and number of iterations. 
    
    Input:
        log_posterior (func): log posterior probability function to evaluate
        args (arr): contains x_data, y_data, and y_err
        ndim (int): number of model parameters to fit
        nwalkers (int): number of emcee walkers to use
        intial_pos (list): initial position in parameter space for each walker
        backend_filename (str): name for backend file of results
        do_burn_in (bool): controls whether to monitor convergence or specify number of iterations
        plot_convergence (bool): controls whether to plot convergence-tracking figure
        num_iter (int): specified number of iterations if do_burn_in==True
        burn_in (int): number of steps to discard as burn-in
        thin (int): thinning rate for chain. If thin=n, keep only every nth sample
        
    """
    # Set up the backend to store chain results in case of crashing or infinite looping
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(nwalkers, ndim) #reset if it's already been created

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=args, backend=backend)
    
    if(do_burn_in):
        #run until converged , with the option of plotting convergence
        autocorr, tau = mcmc_burn_in(sampler,plot_convergence)
        
        #calculate the burn-in and thin parameters
        burn_in = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        
        print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(tau)))
        print("burn-in: {0}".format(burn_in))
        print("thin: {0}".format(thin))
    else:
        #run with specified number of iterations
        flat_samples = sampler.run_mcmc(initial_pos, num_iter, progress=True)
        
    flat_samples = sampler.get_chain(discard=burn_in, flat=True, thin=thin) 
    return sampler, flat_samples , burn_in , thin

def plot_convergence(autocorr,index):
    """
    Plot the convergence-tracking figure of autocorrelation time vs chain length
    
    Input:
        autocorr (arr)- autocorrelation times as a function of chain length
        index (int)- Number of autocorrelation time measurements to plot
    """
    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.show()
    
def mcmc_burn_in(sampler,plot,max_n=10000000):
    """
    Note: The following code is adapted from an emcee tutorial
    
    Run mcmc for maximum 100,000 steps, or until converged
    
    Input:
        sampler (obj)- output of emcee with multiple walkers
    Output:
        autocorr (arr)- autocorrelation times as a function of chain length
        tau (float) - autocorrelation time
    """

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(initial_pos, iterations=max_n, store = True , progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    
    tau = sampler.get_autocorr_time()
    
    if(plot):
        plot_convergence(autocorr,index)
    
    return autocorr, tau

def mu_func(z, cosmo):
    """
    Given a redshift value z and universe model cosmo, convert to luminosity distance and calculate 
    the distance modulus mu (aka m-M)
    
    Input:
        z (float): redshift value
        cosmo (obj): LambdaCDM cosmology model
    Output:
        mu (float): distance modulus, aka m-M
    """
    D_L = cosmo.luminosity_distance(z).value #convert z to luminosity distance
    mu = 5*np.log10(D_L)+25 #calculate mu
    return mu

def log_likelihood_sn(theta, z, mu_data, mu_err):
    """
    Evaluate the log likelihood of a SNe Ia fit with an uncorrelated error 
    model given a set of model parameters stored in the array theta
    
    Input:
        theta - array of model params
        z - array of redshift data from SNe Ia
        mu_data - array of mu data from SNe Ia
        mu_err - array of error bars on mu_data
    Output:
        Return the log likelihood probability, which is equal to -chi_sq
    """
    #Get the parameters and create the corresponding cosmo model
    H0, Om0, Ode0 = theta
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    #Evaluate the model at the data point z values
    mu_model = mu_func(z, cosmo)
    
    sigma2 = mu_err ** 2
    return -0.5 * np.sum((mu_data - mu_model) ** 2 / sigma2 + np.log(2*np.pi*sigma2)) # the 2pi factor doesn't affect the shape

def log_prior_sn(theta):
    """
    Evaluate the log prior probability function given model parameters
    
    Input:
        theta (arr)- model params
    Output:
        Return 0.0 if params fall within constraints, else return -np.inf
    """
    H0, Om0, Ode0 = theta
    #Check that the params are allowed by our physical constraints
    if 0. <= Om0 <= 1. and 0. < Ode0 < 1.:
        return 0.0 # the constant doesn't matter since MCMCs only care about *ratios* of probabilities
    return -np.inf # log(0) = -inf

def log_post_sn(theta, z, mu_data, mu_err):
    """
    Evaluate the log posterior probability function given SNe Ia data and model parameters
    
    Input:
        theta (arr)- model params
        z (arr)- redshift data from SNe Ia
        mu_data (arr)- mu data from SNe Ia
        mu_err (arr)- error bars on mu_data
    Output:
        Return log likelihood probability if params fall within constraints, else return -np.inf
        
    """
    lp = log_prior_sn(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sn(theta, z, mu_data, mu_err)
