from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import h5py
import numpy as np
import scipy.stats as st
import scipy.special as sp
import scipy.linalg as lg
from emcee.mpi_pool import MPIPool,_close_pool_message,_function_wrapper

from functools import partial

ph_dim = 5   #photometric dimensions
cf_dim = 4
dm     = 0.75

from time import time
from Priors import Support
from Priors import logPriors
from Likelihood import logLikeIndependent
from Likelihood import logLikeIndependentInc



class LogPosteriorModule(object):
    """
    Chain for computing the likelihood 
    """
    def __init__(self,pathD,comm):
        """
        Constructor of the logposteriorModule
        """
        self.pathD=pathD
        print("Initialasing pool of workers ...")
        size = comm.Get_size()
        rank = comm.Get_rank()

        sys.stdout.write("Hello, World! I am process %d of %d.\n"% (rank, size))
        self.pool = MPIPool(comm=comm,loadbalance=False)
        print("Log Posterior Initialized")
    
    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        
        with h5py.File(self.pathD,'r') as hf:
            print( "Reading Constants")
            cons                = hf.get('Constants')
            self.alpha          = np.array(cons.get("alphas"))
            self.mu_pm_hyp      = np.array(cons.get("mean_pm_hyp"))
            self.sg_pm_hyp      = np.array(cons.get("sigma_pm_hyp"))
            self.mu_coefs       = np.array(cons.get("mu_coefs"))

            self.alpha_color    = np.array(cons.get("alpha_mag"))
            self.rg_color       = np.array(cons.get("rg_clr"))
            self.vr_coefs_hyp   = np.array(cons.get("vr_coefs_hyp"))
            self.A_phot         = np.array(cons.get("A_phot"))
            self.A_pm           = np.array(cons.get("A_pm"))

            self.K              = np.array(cons.get("K"))[0]
            self.stp            = np.array(cons.get("stp_int"))[0]
            self.nu             = np.array(cons.get("nu_hyp"))[0]
            self.vr_clr_hyp     = np.array(cons.get("vr_clr_hyp"))[0]
            self.mu_clr_hyp     = 1e-3

            print( "Reading Data")
            data      = hf.get('Data')
            index     = np.array(data.get("indicator"))
            obs_T     = np.array(data.get("observations"))
            idx_full  = np.array(data.get("index_full"))-1
            idx_miss  = np.array(data.get("index_miss"))-1


        N_full            = len(idx_full)
        N_miss            = len(idx_miss)
        N_obs             = N_full + N_miss

        self.neva              = 300
        self.delcol            = np.linspace(self.rg_color[0],self.rg_color[1],self.neva)
    
        self.tu_idx = np.triu_indices(ph_dim)
        #---------------- Missing and Non-missing -----------
        self.obs_c = np.array([obs_T[idx_full[i]] for i in range(N_full)])
        self.obs_i = np.array([np.hstack([obs_T[idx_miss[i]],index[i]]) for i in range(N_miss)])

        
        #Remove AFTER DEBUGGING
        # ###################################################
        Ns = 2
        self.obs_c = np.array([obs_T[idx_full[i]] for i in range(Ns)])
        self.obs_i = np.array([np.hstack([obs_T[idx_miss[i]],index[i]]) for i in range(Ns)])
        # # ###################################################

        print("LogPosteriorModule setup done")

    def computeLikelihood(self, ctx):
        # This function calculates the logarithm of the posterior distribution of the parameters
        # given the data
        #       Paramters
        #------------------------------
        pi       = ctx.get('pi')
        pi_color = ctx.get('pi_color')
        mu_color = ctx.get('mu_color')
        vr_color = ctx.get('vr_color')
        mu_pm_ps = ctx.get('mu_pm_ps')
        sg_pm_ps = ctx.get('sg_pm_ps')
        mu_pm_bs = ctx.get('mu_pm_bs')
        sg_pm_bs = ctx.get('sg_pm_bs')
        coefs    = ctx.get('coefs')
        tu_cov   = ctx.get('tu_cov')
        
        #alpha    = ctx.get('alpha')
        alpha    = np.array([0,0,0,0,0])
        #------- Covariance of photometry phot_cov is triangular upper ---
        phot_cov              = np.zeros((ph_dim,ph_dim))
        phot_cov[np.triu_indices(ph_dim)] = tu_cov
        phot_cov = phot_cov + np.triu(phot_cov,1).T
        #------------------------------
        # Generated Quantities in case of rejection
        ctx.getData()["PC1"] = str('NA')
        ctx.getData()["PC2"] = str('NA')

        # ctx.getData()["PhPs"] = str('NA')
        # ctx.getData()["PhBs"] = str('NA')

        #----- Checks if parameters' values are in the ranges
        supp = Support(pi,pi_color,mu_color,vr_color,
                    sg_pm_ps,sg_pm_bs,coefs,phot_cov,self.rg_color)


        if supp == -np.inf : 
            #print(":(")
            return -np.inf

        #------- Computes log of priors
        lpp  = logPriors(pi,pi_color,mu_color,vr_color,
                         mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
                         coefs,phot_cov,
                         self.alpha,self.rg_color,self.alpha_color,
                         self.mu_clr_hyp,self.vr_clr_hyp,
                         self.mu_pm_hyp,self.sg_pm_hyp,
                         self.mu_coefs,self.vr_coefs_hyp,
                         self.nu,self.A_phot,self.A_pm)


        #--------------------Laguerre polynomials ----------------------------
        evals =  np.vstack([np.polynomial.laguerre.lagval(self.delcol, coefs[0], tensor=True),
                            np.polynomial.laguerre.lagval(self.delcol, coefs[1], tensor=True),
                            np.polynomial.laguerre.lagval(self.delcol, coefs[2], tensor=True),
                            self.delcol,
                            np.polynomial.laguerre.lagval(self.delcol, coefs[3], tensor=True)]).T
        # ----- Computes Likelihoods ---------
        # Non-missing data
        #-----parallel function -----------
        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.


        # wrapplgLikIndepComp = _function_wrapper(logLikeIndependent,(pi,pi_color,mu_color,vr_color,
        #         mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
        #         self.stp,self.rg_color,coefs,phot_cov,alpha,evals),{})

        # wrapplgLikIndepMiss = _function_wrapper(logLikeIndependentInc,(pi,pi_color,mu_color,vr_color,
        #         mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
        #         self.stp,self.rg_color,coefs,phot_cov,alpha,evals),{})

        # lpsc   = self.pool.map(wrapplgLikIndepComp,self.obs_c.copy())

        # lpsi   = self.pool.map(wrapplgLikIndepMiss,self.obs_i.copy())

        lpsi =  self.pool.map(partial(logLikeIndependentInc,
                pi=pi,pi_color=pi_color,mu_color=mu_color,vr_clr=vr_color,
                mu_pm_ps=mu_pm_ps,sg_pm_ps=sg_pm_ps,mu_pm_bs=mu_pm_bs,sg_pm_bs=sg_pm_bs,
                stp=self.stp,rg_color=self.rg_color,coefs=coefs,phot_cov=phot_cov,alpha=alpha,evals=evals),self.obs_i)

        lpsc = self.pool.map(partial(logLikeIndependent,
                pi=pi,pi_color=pi_color,mu_color=mu_color,vr_clr=vr_color,
                mu_pm_ps=mu_pm_ps,sg_pm_ps=sg_pm_ps,mu_pm_bs=mu_pm_bs,sg_pm_bs=sg_pm_bs,
                stp=self.stp,rg_color=self.rg_color,coefs=coefs,phot_cov=phot_cov,alpha=alpha,evals=evals),self.obs_c)


        # lpsi =  map(lambda x:logLikeIndependentInc(x,
        #         pi,pi_color,mu_color,vr_color,
        #         mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
        #         self.stp,self.rg_color,coefs,phot_cov,alpha,evals),self.obs_i)

        # lpsc = map(lambda x:logLikeIndependent(x,
        #         pi,pi_color,mu_color,vr_color,
        #         mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
        #         self.stp,self.rg_color,coefs,phot_cov,alpha,evals),self.obs_c)

        # sys.exit()
        #--------- Blobs section ---------
        # loglikelihoods
        lps_c = [seq[0]  for seq in lpsc]
        lps_i = [seq[0]  for seq in lpsi]

        # Put generated quantities into strings to write them with DerivedParmterFileUtil
        PC1   = '\t'.join(sum([[str(seq[1])  for seq in lpsc],[str(seq[1])  for seq in lpsi]],[]))
        PC2   = '\t'.join(sum([[str(seq[2])  for seq in lpsc],[str(seq[2])  for seq in lpsi]],[]))

        ctx.getData()["PC1"] = str(PC1)
        ctx.getData()["PC2"] = str(PC2)

        # PhPs   = '\t'.join(sum([[str(seq[3])  for seq in lpsc],[str(seq[3])  for seq in lpsi]],[]))
        # PhBs   = '\t'.join(sum([[str(seq[4])  for seq in lpsc],[str(seq[4])  for seq in lpsi]],[]))

        # ctx.getData()["PhPs"] = str(PhPs)
        # ctx.getData()["PhBs"] = str(PhBs)  

        print(lpp,np.sum(lps_c),np.sum(lps_i))

        return lpp+np.sum(lps_i)+np.sum(lps_c)


# class _function_wrapper(object):
#     """
#     This is a hack to make the likelihood function pickleable when ``args``
#     or ``kwargs`` are also included.

#     """
#     def __init__(self, f, args, kwargs):
#         self.f = f
#         self.args = args
#         self.kwargs = kwargs

#     def __call__(self, x):
#         try:
#             return self.f(x, *self.args, **self.kwargs)
#         except:
#             import traceback
#             print("emcee: Exception while calling your likelihood function:")
#             print("  params:", x)
#             print("  args:", self.args)
#             print("  kwargs:", self.kwargs)
#             print("  exception:")
#             traceback.print_exc()
#             raise


    
