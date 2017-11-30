from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import h5py
import numpy as np
import scipy.stats as st
import scipy.special as sp
import scipy.linalg as lg
from scipy.interpolate import splev
import multiprocessing
from time import time

from functools import partial

ph_dim = 5   #photometric dimensions
cf_dim = 4
dm     = 0.75

id_mag   = np.array([1,2,3,4])

from time import time
from PriorsSpline import Support
from PriorsSpline import logPriors
from LikelihoodCuda_0 import logLikeIndependent
from LikelihoodCuda_0 import logLikeIndependentInc

class LogPosteriorModule(object):
    """
    Chain for computing the likelihood 
    """
    def __init__(self,pathD,threads):
        """
        Constructor of the logposteriorModule
        """
        self.pathD=pathD
        self.threads = threads
        print("Log Posterior Initialized")
    
    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        
        with h5py.File(self.pathD,'r') as hf:
            print( "Reading Constants")
            cons                = hf.get('Constants')
            self.theta_hyp      = np.array(cons.get("theta_hyp"))
            self.theta_Ps_hyp   = np.array(cons.get("alpha_Ps"))
            self.theta_Bs_hyp   = np.array(cons.get("alpha_Bs"))
            self.mu_pm_hyp      = np.array(cons.get("mean_pm_hyp"))
            self.sg_pm_hyp      = np.array(cons.get("sigma_pm_hyp"))
            self.mu_coefs       = np.array(cons.get("mu_coefs"))
            self.knots          = np.array(cons.get("knots"))

            self.theta_clr_hyp  = np.array(cons.get("alpha_mag"))
            self.rg_color       = np.array(cons.get("rg_clr"))
            self.vr_coefs_hyp   = np.array(cons.get("vr_coefs_hyp"))
            self.A_phot         = np.array(cons.get("A_phot"))
            self.A_pm           = np.array(cons.get("A_pm"))

            self.stp            = np.array(cons.get("stp_int"))[0]
            #self.stp_clr = 
            self.nu             = np.array(cons.get("nu_hyp"))[0]
            self.vr_clr_hyp     = np.array(cons.get("vr_clr_hyp"))[0]

            self.vr_alpha_hyp   = np.array(cons.get("vr_alpha_hyp"))
            self.degspl         = int(np.array(cons.get("spl_deg"))[0])
            

            print( "Reading Data")
            data      = hf.get('Data')
            obs_T     = np.array(data.get("observations"))
            index     = np.array(data.get("indicator"))
            idx_full  = np.array(data.get("index_full"))-1
            idx_miss  = np.array(data.get("index_miss"))-1


        N_full            = len(idx_full)
        N_miss            = len(idx_miss)


        self.neva              = 300
        self.delcol            = np.linspace(self.rg_color[0],self.rg_color[1],self.neva)
    
        self.tu_idx = np.triu_indices(ph_dim)
        
        #---------------- Missing and Non-missing -----------
        '''
        self.obs_c = np.array([obs_T[idx_full[i]] for i in range(N_full)])
        self.obs_i = np.array([np.hstack([obs_T[idx_miss[i]],index[i]]) for i in range(N_miss)])'''
        
        #Remove AFTER DEBUGGING
        ###################################################
        Ns = 20
        self.obs_c = np.array([obs_T[idx_full[i]] for i in range(Ns)])
        self.obs_i = np.array([np.hstack([obs_T[idx_miss[i]],index[i]]) for i in range(Ns)])
        # # ###################################################
        "Initialasing pool of workers ..."
        #self.pool = multiprocessing.Pool(self.threads)

        print("LogPosteriorModule setup done")
            
    def computeLikelihood(self, ctx):
        # This function calculates the logarithm of the posterior distribution of the parameters
        # given the data
        #       Paramters
        #------------------------------

        
        pi       = ctx.get('pi')
        pi_pm_ps = ctx.get('pi_pm_ps')
        pi_pm_bs = ctx.get('pi_pm_bs')
        pi_color = ctx.get('pi_color')
        mu_color = ctx.get('mu_color')
        vr_color = ctx.get('vr_color')
        mu_pm_ps = ctx.get('mu_pm_ps')
        sg_pm_ps = ctx.get('sg_pm_ps')
        mu_pm_bs = ctx.get('mu_pm_bs')
        sg_pm_bs = ctx.get('sg_pm_bs')
        coefp    = ctx.get('coefp')
        etu_cov  = ctx.get('etu_cov')
        alpha    = ctx.get('alpha')
        # knotsp   = ctx.get('knotsp')

        #print(pi)
        #print(pi_pm)
        #print(pi_color)
        #print(mu_color)
        #print(vr_color)
        #print(mu_pm_ps)
        #print(sg_pm_ps)
        #print(mu_pm_bs)
        #print(sg_pm_bs)
        #print(coefs)
        #print(etu_cov)
        #sys.exit(0)
        
        # Generated Quantities in case of rejection
        ctx.getData()["PC1"] = str('NA')
        ctx.getData()["PC2"] = str('NA')

        # ctx.getData()["PhPs"] = str('NA')
        # ctx.getData()["PhBs"] = str('NA')        
        
        #----- Checks if parameters' values are in the ranges
        supp = Support(pi,pi_pm_ps,pi_pm_bs,pi_color,mu_color,vr_color,
                    sg_pm_ps,sg_pm_bs,coefp,etu_cov,self.rg_color)


        if supp == -np.inf : 
            #print(":(")
            return -np.inf

        ######################## WRAPPING #########################################
        #------- Covariance of photometry phot_cov is triangular upper ---
        tu_cov   = np.zeros((ph_dim,ph_dim))
        tu_cov[np.triu_indices(ph_dim)] = etu_cov
        phot_cov = np.dot(tu_cov.T,tu_cov)
        #----------- knots ------------
        knots = np.hstack([np.repeat(self.rg_color[0],self.degspl),self.knots,np.repeat(self.rg_color[1],self.degspl)])
        #------- wrap knots, coefficients and degree into list for splev-------
        coefs =np.array([np.array([knots,np.append(coefp[0],[0]*(self.degspl+1))]),
                 np.array([knots,np.append(coefp[1],[0]*(self.degspl+1))]),
                 np.array([knots,np.append(coefp[2],[0]*(self.degspl+1))]),
                 np.array([knots,np.append(coefp[3],[0]*(self.degspl+1))])])
        ###############################################################################################################

        #------- Computes log of priors
        lpp  = logPriors(pi,pi_pm_ps,pi_pm_bs,pi_color,mu_color,vr_color,
                         mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
                         coefp,phot_cov,alpha,
                         self.theta_hyp,self.theta_Ps_hyp,self.theta_Bs_hyp,
                         self.rg_color,self.theta_clr_hyp,self.vr_clr_hyp,
                         self.mu_pm_hyp,self.sg_pm_hyp,
                         self.mu_coefs,self.vr_coefs_hyp,
                         self.nu,self.A_phot,self.A_pm,self.vr_alpha_hyp)
        # sys.exit()
        #--------------------Laguerre polynomials ----------------------------
        evals =  np.vstack([self.delcol,
                            splev(self.delcol,np.array([coefs[0][0],coefs[0][1],self.degspl])),
                            splev(self.delcol,np.array([coefs[1][0],coefs[1][1],self.degspl])),
                            splev(self.delcol,np.array([coefs[2][0],coefs[2][1],self.degspl])),
                            splev(self.delcol,np.array([coefs[3][0],coefs[3][1],self.degspl]))]).T
        # ----- Computes Likelihoods ---------
        #self.pool.map                          
              
        lpsi =  map(partial(logLikeIndependentInc,
                pi=pi,pi_pm_ps=pi_pm_ps,pi_pm_bs=pi_pm_bs,pi_color=pi_color,mu_color=mu_color,vr_clr=vr_color,
                mu_pm_ps=mu_pm_ps,sg_pm_ps=sg_pm_ps,mu_pm_bs=mu_pm_bs,sg_pm_bs=sg_pm_bs,
                coefs=coefs,phot_cov=phot_cov,alpha=alpha,
                stp=self.stp,rg_color=self.rg_color,
                evals=evals,degspl=self.degspl,id_mag=id_mag),self.obs_i)

        lpsc =  map(partial(logLikeIndependent,
                pi=pi,pi_pm_ps=pi_pm_ps,pi_pm_bs=pi_pm_bs,pi_color=pi_color,mu_color=mu_color,vr_clr=vr_color,
                mu_pm_ps=mu_pm_ps,sg_pm_ps=sg_pm_ps,mu_pm_bs=mu_pm_bs,sg_pm_bs=sg_pm_bs,
                coefs=coefs,phot_cov=phot_cov,alpha=alpha,
                stp=self.stp,rg_color=self.rg_color,
                evals=evals,degspl=self.degspl,id_mag=id_mag),self.obs_c)

        print(lpsi)
        # "Initialasing pool of workers ...
        # pool = multiprocessing.Pool(self.threads)
        # lpsi = map(partial(logLikeIndependentInc,
        #         pi=pi,pi_color=pi_color,mu_color=mu_color,vr_clr=vr_color,
        #         mu_pm_ps=mu_pm_ps,sg_pm_ps=sg_pm_ps,mu_pm_bs=mu_pm_bs,sg_pm_bs=sg_pm_bs,
        #         stp=self.stp,rg_color=self.rg_color,coefs=coefs,phot_cov=phot_cov,alpha=alpha,evals=evals),self.obs_i)

        # lpsc = map(partial(logLikeIndependent,
        #         pi=pi,pi_color=pi_color,mu_color=mu_color,vr_clr=vr_color,
        #         mu_pm_ps=mu_pm_ps,sg_pm_ps=sg_pm_ps,mu_pm_bs=mu_pm_bs,sg_pm_bs=sg_pm_bs,
        #         stp=self.stp,rg_color=self.rg_color,coefs=coefs,phot_cov=phot_cov,alpha=alpha,evals=evals),self.obs_c)
        # pool.close()
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
        
        # print(lpp,np.sum(lps_c),np.sum(lps_i))
        sys.exit(0)
        return lpp+np.sum(lps_i)+np.sum(lps_c)


    
