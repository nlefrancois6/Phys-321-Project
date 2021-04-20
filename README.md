# Phys-321-Project

Description of the problem: 
We applied an ensemble Markov Chain Monte Carlo (MCMC) technique to Cosmic Microwave Background (CMB) and Type Ia supernovae (SNe Ia) data in order to sample the posterior probability distributions of cosmological parameters (including Hubble constant, cosmological curvature, dark energy and cold dark matter fractions). We compared the probability distributions obtained from the two different data sets, producing a figure that overlays the two distributions. We also implemented a correlated errors model for the CMB data, although due to slow run-time and our limited computing time for this project we were unable to run the MCMC with this model and investigate the results, choosing to instead focus on the investigation of the uncorrelated model results which had more reasonable run-time.

Contents:

Project_Main.ipynb: contains all of our data loading, fitting, and analysis figures.

helper_functions.py: contains all of the functions used for calculating posterior probabilities, correlated error model, MCMC fit analysis & plotting, animation of the fit, and overlaying the two posterior probability distributions.

SNe_MCMC_animation.gif: Animation showing how the fit moves over the first few steps
WMAP_MCMC_animation.gif: Animation showing how the fit moves over the first few steps


Contributions:

Ingrid: Helped trouble shoot the convergence of the MCMC, ran and generated all the plots for the lab. Wrote intro for LambdaCDM. Wrote the results, discussion as well as all the figure captions. Created table 1 of the final results. 

Noah: Wrote prior, likelihood, and posterior functions, covariance model using correlated & uncorrelated errors, functions for plotting WMAP and SNe fits as well as animation of the fit. Wrote & ran testing functions, wrote function I/O header comments. Wrote report sections 2.4, 3.1, bibliography, and helped edit the paper.

Katie:
