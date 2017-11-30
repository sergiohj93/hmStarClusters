from __future__ import print_function
from cosmoHammer import getLogger

import numpy as np
#import rpy2.robjects as robjects
ph_dim = 5

class CoreModule(object):
    """
    Core Module for calculating variables of model M17
    """

    def __init__(self,sizes):
        """
        Constructor of the CoreModule
        """
        self.ss = sizes

        if len(self.ss)!= 12: print("Discrepant number of parameters' groups in context")
        
    def __call__(self, ctx):
        """
        Computes something and stores it in the context
        """
        # Get the parameters from the context
        params     = ctx.getParams()
        idx        = 0
        idxf       = idx + self.ss[0,0]
        pi         = params[idx:idxf]
        idx        = idxf
        idxf       = idx + self.ss[1,0]
        pipm_ps    = params[idx:idxf]
        idx        = idxf
        idxf       = idx + self.ss[2,0]
        pipm_bs    = params[idx:idxf]
        idx        = idxf
        idxf       = idx + self.ss[3,0]
        pi_color   = params[idx:idxf]
        idx        = idxf
        idxf       = idx + self.ss[4,0]
        mu_color   = params[idx:idxf]
        idx        = idxf
        idxf       = idx + self.ss[5,0]
        vr_color   = params[idx:idxf]
        idx        = idxf
        idxf       = idx + np.prod(self.ss[6,:])
        mu_pm_ps   = params[idx:idxf]
        idx        = idxf
        idxf       = idx + np.prod(self.ss[7,:])
        tu_pm_ps   = params[idx:idxf]
        idx        = idxf
        idxf       = idx + np.prod(self.ss[8,:])
        mu_pm_bs   = params[idx:idxf]
        idx        = idxf
        idxf       = idx + np.prod(self.ss[9,:])
        tu_pm_bs   = params[idx:idxf]
        # idx        = idxf
        # idxf       = idx + np.prod(self.ss[10,:])
        # knotsp     = params[idx:idxf]
        idx        = idxf
        idxf       = idx + np.prod(self.ss[10,:])
        coefp      = params[idx:idxf]
        idx        = idxf
        idxf       = idx + self.ss[11,0]
        etu_cov    = params[idx:idxf]
        # idx        = idxf
        # idxf       = idx + self.ss[11,0]
        # alpha      = params[idx:idxf]
        alpha    = np.array([0,0,0,0,0])

        # Resahpe parameters

        # mu_pm_ps = np.reshape(mu_pm_ps,(self.ss[5,0],self.ss[5,1]))
        tu_pm_ps = np.reshape(tu_pm_ps,(self.ss[7,0],self.ss[7,1]))
        # mu_pm_bs = np.reshape(mu_pm_bs,(self.ss[7,0],self.ss[7,1]))
        tu_pm_bs = np.reshape(tu_pm_bs,(self.ss[9,0],self.ss[9,1]))
        coefp    = np.reshape(coefp,   (self.ss[10,0],self.ss[10,1]))

        #---- construct proper motion matrices
        sg_pm_ps = np.zeros((self.ss[7,0],2,2))
        sg_pm_bs = np.zeros((self.ss[9,0],2,2))

        sg_pm_ps[0,0,0] = tu_pm_ps[0,0]
        sg_pm_ps[0,0,1] = tu_pm_ps[0,1]
        sg_pm_ps[0,1,0] = tu_pm_ps[0,1]
        sg_pm_ps[0,1,1] = tu_pm_ps[0,2]

        sg_pm_ps[1,0,0] = tu_pm_ps[1,0]
        sg_pm_ps[1,0,1] = tu_pm_ps[1,1]
        sg_pm_ps[1,1,0] = tu_pm_ps[1,1]
        sg_pm_ps[1,1,1] = tu_pm_ps[1,2]

        sg_pm_ps[2,0,0] = tu_pm_ps[2,0]
        sg_pm_ps[2,0,1] = tu_pm_ps[2,1]
        sg_pm_ps[2,1,0] = tu_pm_ps[2,1]
        sg_pm_ps[2,1,1] = tu_pm_ps[2,2]

        sg_pm_ps[3,0,0] = tu_pm_ps[3,0]
        sg_pm_ps[3,0,1] = tu_pm_ps[3,1]
        sg_pm_ps[3,1,0] = tu_pm_ps[3,1]
        sg_pm_ps[3,1,1] = tu_pm_ps[3,2]

        #----------------------------

        sg_pm_bs[0,0,0] = tu_pm_bs[0,0]
        sg_pm_bs[0,0,1] = tu_pm_bs[0,1]
        sg_pm_bs[0,1,0] = tu_pm_bs[0,1]
        sg_pm_bs[0,1,1] = tu_pm_bs[0,2]

        sg_pm_bs[1,0,0] = tu_pm_bs[1,0]
        sg_pm_bs[1,0,1] = tu_pm_bs[1,1]
        sg_pm_bs[1,1,0] = tu_pm_bs[1,1]
        sg_pm_bs[1,1,1] = tu_pm_bs[1,2]

        # sg_pm_bs[2,0,0] = tu_pm_bs[1,0]
        # sg_pm_bs[2,0,1] = tu_pm_bs[1,1]
        # sg_pm_bs[2,1,0] = tu_pm_bs[1,1]
        # sg_pm_bs[2,1,1] = tu_pm_bs[1,2]


        #-----Add dependent parameters
        pi_color  = np.hstack([pi_color,1-sum(pi_color)])
        pi_pm_ps  = np.hstack([pipm_ps,1-sum(pipm_ps)])
        pi_pm_bs  = np.hstack([pipm_bs,1-sum(pipm_bs)])

        #------- order knots in increasing order ----
        # knotsp.sort()


        # Add the result to the context using a unique key
        ctx.add('pi', pi)
        ctx.add('pi_pm_ps',pi_pm_ps)
        ctx.add('pi_pm_bs',pi_pm_bs)
        ctx.add('pi_color',pi_color)
        ctx.add('mu_color',mu_color)
        ctx.add('vr_color',vr_color)
        ctx.add('mu_pm_ps',mu_pm_ps)
        ctx.add('sg_pm_ps',sg_pm_ps)
        ctx.add('mu_pm_bs',mu_pm_bs)
        ctx.add('sg_pm_bs',sg_pm_bs)
        # ctx.add('knotsp',knotsp)
        ctx.add('coefp',coefp)
        ctx.add('etu_cov',etu_cov)
        ctx.add('alpha',alpha)

    def setup(self):
        """
        Sets up the core module.
        Tasks that need to be executed once per run
        """
        
        print("CoreModule setup done")