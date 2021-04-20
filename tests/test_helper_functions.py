import unittest
import nose.tools as nt
import numpy as np
import helper_functions as hf

import pypico

class test_tools():

    def setUp(self):
        # Create a random array
        #self.n = 4
        #self.rand_array = np.random.normal(size=(self.n,self.n))
        pass

    def tearDown(self):
        pass
    
    def test_WMAP_correlated_posterior(self):
        theta=np.asarray([70,0.02,0.1,0.0,0.05,2e-9,0.97,0.07])
        
        wmap = np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
        multipole = wmap[:,0] 
        power = wmap[:,1] 
        errPower = wmap[:,2]
        pico = pypico.load_pico("jcset_py3.dat")
        
        p_log_post = hf.log_post_WMAP_correlated(theta, multipole, power, errPower)
        
        assert type(p_log_post) == np.float64
        nt.assert_equal(round(p_log_post,1), -1412.6)
        
    def test_WMAP_uncorrelated_posterior(self):
        theta=np.asarray([70,0.02,0.1,0.0,0.05,2e-9,0.97,0.07])
        
        wmap = np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
        multipole = wmap[:,0] 
        power = wmap[:,1] 
        errPower = wmap[:,2]
        pico = pypico.load_pico("jcset_py3.dat")
        
        p_log_post = hf.log_post_WMAP_uncorrelated(theta, multipole, power, errPower)
        
        assert type(p_log_post) == np.float64
        nt.assert_equal(round(p_log_post,1), -1428.6)
    
    def test_WMAP_prior(self):
        H0 = 70
        h = H0/100
        Omb = np.asarray([-0.1, 0.5, 1.1])
        Ombh2 = Omb*h**2

        t1=np.asarray([H0,Ombh2[0],0.1,0.0,0.05,2e-9,0.97,0.07])
        t2=np.asarray([H0,Ombh2[1],0.1,0.0,0.05,2e-9,0.97,0.07])
        t3=np.asarray([H0,Ombh2[2],0.1,0.0,0.05,2e-9,0.97,0.07])
        
        p1 = hf.log_prior_WMAP(t1)
        p2 = hf.log_prior_WMAP(t2)
        p3 = hf.log_prior_WMAP(t3)
        
        nt.assert_equal(-np.inf, p1)
        nt.assert_equal(0.0, p2)
        nt.assert_equal(-np.inf, p3)
        
    
    def test_get_cov_model(self):
        wmap = np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
        errPower = wmap[:,2]
        
        alpha = 0.1
        
        C = hf.get_cov_model(errPower, alpha)
        
        N = len(errPower)
        
        nt.assert_equal(C.shape[0],N)
        nt.assert_equal(C.shape[1],N)
        
    def test_sn_posterior(self):
        sn_z,sn_dm,sn_dm_err = np.loadtxt("SCPUnion2.1_mu_vs_z.txt",delimiter="\t",skiprows=5, usecols = (1,2,3),unpack=True)
        
        theta = np.asarray([70,0.7,0.3])
        p_log_post = hf.log_post_sn(theta,sn_z,sn_dm,sn_dm_err)
        
        assert type(p_log_post) == np.float64
        nt.assert_equal(round(p_log_post,1), -215.2)
        
    def test_sn_prior(self):
        Om0 = np.asarray([-0.1, 0.5, 1.1])
        
        t1 = np.asarray([70,Om0[0],0.3])
        t2 = np.asarray([70,Om0[1],0.3])
        t3 = np.asarray([70,Om0[2],0.3])
        
        p1 = hf.log_prior_sn(t1)
        p2 = hf.log_prior_sn(t2)
        p3 = hf.log_prior_sn(t3)
        
        nt.assert_equal(-np.inf, p1)
        nt.assert_equal(0.0, p2)
        nt.assert_equal(-np.inf, p3)
