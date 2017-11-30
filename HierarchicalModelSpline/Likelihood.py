from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import numpy as np
from numba import jit
import scipy.stats as st
import scipy.special as sp
import scipy.linalg as lg
from scipy.interpolate import splev
import math
from time import time

ph_dim = 5   #photometric dimensions
cf_dim = 4
id_clr = 0       # index of color parameter
id_mag = [1,2,3,4] # indeces of magnituds for Binaries
dm     = 0.75
####################### LIKELIHOOD FUNCTIONS ###########################
@jit
def mcsn(x,sigma_obs,sigma_cls,isgs,lden,alpha):
    gamma     = np.vstack(([0]*len(x),alpha.dot(sigma_cls.dot(isgs))))
    nu        = np.zeros(2)
    Delta     = np.identity(2)
    Delta[1,1]= (1.0+alpha.dot(sigma_cls.dot(alpha))
               -alpha.dot(np.dot(sigma_obs,np.dot(isgs,np.dot(sigma_cls,alpha)))))
    return csn_pdf(x,sigma_obs+sigma_cls,isgs,lden,gamma,nu,Delta)

@jit
def csn_pdf(x, sigma,isgs,lden, gamma, nu, delta):
    # Caso general Ver articulo para el caso especifico de dos csn.
    f1, i = st.mvn.mvnun(lower=[-1000]*len(nu), 
                        upper = [0]*len(nu), 
                        means = nu, 
                        covar = (delta + np.dot(gamma.dot(sigma),gamma.T)))
    f2, i = st.mvn.mvnun(lower=[-1000]*len(nu), 
                        upper = gamma.dot(x.T),
                        means = nu, 
                        covar = delta)
    f3 = np.exp(-0.5*(np.dot(x.T,np.dot(isgs,x))+lden))
    return f3*(f2/f1)


@jit
def p_color(color,pi,mu,vr,rg):
    # There are for normal distributions, they are truncated.
    n = len(pi)
    tps = np.zeros(len(color))
    for i in range(n):
        #-------- denominators without sqrt(pi) ------
        de = np.sqrt(2*vr[i])
        #-------- exponentials divided by denominators----
        ps = np.exp(-((color-mu[i])**2)/(2*vr[i]))/de
        #-------- truncation factors, wihtout 0.5------
        tr = math.erf((rg[1]-mu[i])/de)-math.erf((rg[0]-mu[i])/de)
        tps += pi[i]*(ps/tr)
    #return value with common factors 0.5 and sqrt(pi)
    tps  = (2/np.sqrt(np.pi))*tps
    #-------
    # ps  = 0.99*(1.0/(rg[1]-rg[0]))+0.01*ps
    return tps
    

@jit
def LikeMemberPMone(obs,uncert,mu,sg):
    #------------ ads uncertainty of data
    sg  = sg + np.diag(uncert)
    #------ computes determinant and inverse
    det = (sg[0,0]*sg[1,1])-(sg[0,1]*sg[1,0])
    isg = np.empty((2,2)) 
    isg[0,0]=  sg[1,1]
    isg[0,1]= -sg[0,1]
    isg[1,0]= -sg[1,0]
    isg[1,1]=  sg[0,0] 
    isg = isg/det
    #------- computes density -----
    den = 2*np.sqrt(det)*np.pi
    z   = np.dot((obs-mu).T,isg.dot(obs-mu))
    ps  = np.exp(-0.5*(z))/den
    # print(ps,st.multivariate_normal.pdf(obs, mean=mu, cov=sg))
    return ps

@jit
def LikeMemberPM(obs,uncert,pi,mu,sg):
    ll = 0
    for i in range(len(pi)):
        ll += pi[i]*LikeMemberPMone(obs,uncert,mu,sg[i])
    return  ll

@jit
def LikeMemberPH(m_obs,sg_obs,sg_cls,isg,alpha,color,colorb,coefs,wid):
    #----color and colorb must have same length
    n = len(color)
    #-------common variables----
    lden = np.linalg.slogdet(sg_obs+sg_cls)[1]+(len(m_obs)*np.log(2*np.pi))
    ps=np.empty(n)
    bs=np.empty(n)
    for i in range(n):  
        m_true    = np.empty(ph_dim)
        #-------------------- Spline evaluation ----------------------------
        m_true[0] = color[i]
        m_true[1] = splev(color[i],coefs[0])
        m_true[2] = splev(color[i],coefs[1])
        m_true[3] = splev(color[i],coefs[2])
        m_true[4] = splev(color[i],coefs[3])
        
        #------------------------------------------------------
        x     = m_obs - m_true
        # print(m_obs,m_true)
        #------------------------------------------------------

        # ps[i] = mcsn(x,sg_obs,sg_cls,isg,lden,alpha)
        ps[i] = np.exp(-0.5*(np.dot(x.T,isg.dot(x))+lden))
        # print(np.sqrt(np.dot(x.T,isg.dot(x))))
        # ps0 = st.multivariate_normal.pdf(m_obs,mean=m_true, cov=sg)
        #-------- Binaries -------
        #--------------------Spline evaluation ----------------------------
        m_true[0] = colorb[i]
        m_true[1] = splev(colorb[i],coefs[0])
        m_true[2] = splev(colorb[i],coefs[1])
        m_true[3] = splev(colorb[i],coefs[2])
        m_true[4] = splev(colorb[i],coefs[3])
        
        #------------------------------------------------------
        m_true[id_mag] -= dm

        x     = m_obs - m_true
        # print(m_obs,m_true)
        bs[i] = np.exp(-0.5*(np.dot(x.T,np.dot(isg,x))+lden))
        # bs[i] = st.multivariate_normal.pdf(m_obs,mean=m_true, cov=sg)
    return ps,bs

@jit
def LikeMemberPHInc(m_obs,sg_obs,sg_cls,isg,alpha,color,colorb,coefs,wid,idx):
    #----color and colorb must have same length
    n = len(color)
    lden = np.linalg.slogdet(sg_obs+sg_cls)[1]+(len(m_obs)*np.log(2*np.pi))
    ps=np.empty(n)
    bs=np.empty(n)
    for i in range(n): 
        m_true    = np.empty(ph_dim)
        #-------------------- Spline evaluation ----------------------------
        m_true[0] = color[i]
        m_true[1] = splev(color[i],coefs[0])
        m_true[2] = splev(color[i],coefs[1])
        m_true[3] = splev(color[i],coefs[2])
        m_true[4] = splev(color[i],coefs[3])
        #------------------------------------------------------
        x      = m_obs - m_true[idx]
        # print(m_obs,m_true)
        # ps[i]  = mcsn(x,sg_obs,sg_cls,isg,lden,alpha)
        ps[i]  = np.exp(-0.5*(np.dot(x.T,np.dot(isg,x))+lden))
        # print(np.sqrt(np.dot(x.T,isg.dot(x))))
        # ps0 = st.multivariate_normal.pdf(m_obs,mean=m_true[idx], cov=sg)
        # print(ps[i],x,lden)

        #-------- Binaries -------
        #--------------------Spline evaluation ----------------------------
        m_true[0] = colorb[i]
        m_true[1] = splev(colorb[i],coefs[0])
        m_true[2] = splev(colorb[i],coefs[1])
        m_true[3] = splev(colorb[i],coefs[2])
        m_true[4] = splev(colorb[i],coefs[3])
        #------------------------------------------------------
        m_true[id_mag] -= dm
        # print(m_obs,m_true)

        x     = m_obs - m_true[idx]
        bs[i] = np.exp(-0.5*(np.dot(x.T,np.dot(isg,x))+lden))
        # bs[i] = st.multivariate_normal.pdf(m_obs,mean=m_true[idx], cov=sg)
    return ps,bs

# @jit
def MarginalColor(phot,sg_obs,sg_cls,alpha,coefs,pi_color,mu_color,vr_clr,
                    rg_color,stp,evals):
    dclr  = 0.1 
    
    sg    = sg_obs + sg_cls
    isg   = np.linalg.inv(sg)

    ###################### SINGLE MASS STARS ####################################################
    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in evals-phot])
    ctr_i = np.argmin(mha)
    ctr   = evals[ctr_i].copy()
    bins  = evals[max(0,ctr_i-25):min(len(evals),ctr_i+25)].copy()
    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in bins-ctr])
    ball  = bins[np.where(mha <= 3.0),id_clr].copy()
    # dom   = np.array([min(ctr[id_clr]-0.05,np.min(ball)),max(ctr[id_clr]+0.05,np.max(ball))])
    dom   = np.array([np.min(ball),np.max(ball)+dclr])
    ###################### BINARIES #############################################################
    nzc   = int(1.0/((rg_color[1]-rg_color[0])/len(evals)))
    #---- true phot bewteen ctr_clr and ctr_clr + 0.5
    bbins = evals[max(0,ctr_i-int(0.5*nzc)):min(len(evals),ctr_i+nzc)].copy()
    # #---- true binaries phot
    bbins[:,id_mag] -= dm
    # #------ distance ---------------------------------
    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in bbins-phot])
    ctrb_i= np.argmin(mha)
    ctrb  = bbins[ctrb_i].copy()
    bbins = bbins[max(0,ctrb_i-25):min(len(bbins),ctrb_i+25)].copy()

    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in bbins-ctrb])
    ball  = bbins[np.where(mha <= 3.0),id_clr].copy()
    # domb  = np.array([min(ctrb[id_clr]-0.05,np.min(ball)),max(ctrb[id_clr]+0.05,np.max(ball))])
    domb   = np.array([np.min(ball),np.max(ball)+dclr])
    ############################################################################################
    # #----- ensures that the domain is inside the limits of color  
    dom   = np.array([max(dom[0],rg_color[0]),min(dom[1],rg_color[1])])
    domb  = np.array([max(domb[0],rg_color[0]),min(domb[1],rg_color[1])])
    
    dtau   = (dom[1]-dom[0])/(stp-1)
    dtaub  = (domb[1]-domb[0])/(stp-1)
    tau    = np.linspace(dom[0],dom[1],stp)
    taub   = np.linspace(domb[0],domb[1],stp)

    pclr   = p_color(tau,pi_color,mu_color,vr_clr,rg_color)
    pclrb  = p_color(taub,pi_color,mu_color,vr_clr,rg_color)
    ps,bs  = LikeMemberPH(phot,sg_obs,sg_cls,isg,alpha,tau,taub,coefs,rg_color)
    PhPs   = np.dot(pclr,ps)*dtau
    PhBs   = np.dot(pclrb,bs)*dtaub

    return PhPs,PhBs

#@jit
def MarginalColorInc(phot,sg_obs,sg_cls,alpha,coefs,pi_color,mu_color,vr_clr,
                    rg_color,stp,w,evals):
    dclr   = 0.1#0.05
    dmha   = 3.5
    nw     = 25

    #------- missing ---------
    idx      = np.ix_(w)
    alpha    = alpha[idx]
    phot     = phot[idx]
    sg_obs   = sg_obs[np.ix_(w,w)]
    sg_cls   = sg_cls[np.ix_(w,w)]
    sg       = sg_obs + sg_cls
    isg      = np.linalg.inv(sg)
    ###################### SINGLE MASS STARS ####################################################
    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in evals[:,idx[0]]-phot])
    ctr_i = np.argmin(mha)
    ctr   = evals[ctr_i].copy()
    bins  = evals[max(0,ctr_i-nw):min(len(evals),ctr_i+nw)].copy()
    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in bins[:,idx[0]]-ctr[idx]])
    ball  = bins[np.where(mha <= dmha),id_clr].copy()
    # dom   = np.array([min(ctr[id_clr]-dclr,np.min(ball)),max(ctr[id_clr]+dclr,np.max(ball))])
    dom   = np.array([np.min(ball),np.max(ball)+dclr])
    ###################### BINARIES #############################################################
    nzc   = int(1.0/((rg_color[1]-rg_color[0])/len(evals)))
    # #---- true phot bewteen ctr_clr and ctr_clr + 0.5
    bbins = evals[max(0,ctr_i-int(0.5*nzc)):min(len(evals),ctr_i+nzc)].copy()
    #---- true binaries phot
    bbins[:,id_mag] -= dm
    # #------ distance ---------
    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in bbins[:,idx[0]]-phot])
    ctrb_i= np.argmin(mha)
    ctrb  = bbins[ctrb_i].copy()
    bbins = bbins[max(0,ctrb_i-nw):min(len(bbins),ctrb_i+nw)].copy()

    mha   = np.array([np.sqrt(x.T.dot(isg).dot(x)) for x in bbins[:,idx[0]]-ctrb[idx]])
    ball  = bbins[np.where(mha <= dmha),id_clr].copy()
    # domb  = np.array([min(ctrb[id_clr]-dclr,np.min(ball)),max(ctrb[id_clr]+dclr,np.max(ball))])
    domb   = np.array([np.min(ball),np.max(ball)+dclr])
    ############################################################################################
    #----- ensures that the domain is inside the limits of color  
    dom   = np.array([max( dom[0],rg_color[0]),min( dom[1],rg_color[1])])
    domb  = np.array([max(domb[0],rg_color[0]),min(domb[1],rg_color[1])])
    
    dtau  = (dom[1]-dom[0])/(stp-1)
    dtaub = (domb[1]-domb[0])/(stp-1)

    tau   = np.linspace(dom[0],dom[1],stp)
    taub  = np.linspace(domb[0],domb[1],stp)

    pclr  = p_color(tau,pi_color,mu_color,vr_clr,rg_color)
    pclrb = p_color(taub,pi_color,mu_color,vr_clr,rg_color)

    ps,bs = LikeMemberPHInc(phot,sg_obs,sg_cls,isg,alpha,tau,taub,coefs,rg_color,idx)

    PhPs  = np.dot(pclr,ps)*dtau
    PhBs  = np.dot(pclrb,bs)*dtaub
    return PhPs,PhBs


def logLikeIndependent(datum,pi,pi_pm_ps,pi_pm_bs,
    pi_color,mu_color,vr_clr,
    mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
    coefs,phot_cov,alpha,
    stp,rg_color,evals):
    #------- Non-Members ----------
    lnm      = datum[0]
    #--------------- Datum -----------
    obs      = datum[1:(2+ph_dim+1)]
    uncert   = datum[(2+ph_dim+1):]
    pm       = obs[0:2]
    phot     = obs[2:]
    u_pm     = uncert[0:2]
    u_phot   = np.zeros((ph_dim,ph_dim))
    u_phot[np.triu_indices(ph_dim)]   = uncert[2:]
    u_phot   = u_phot +np.triu(u_phot,1).T

    #---------------  Marginalization of color -----------------------
    PhPs,PhBs= MarginalColor(phot,u_phot,phot_cov,
                 alpha,coefs,pi_color,mu_color,vr_clr,rg_color,stp,evals)
    #--------------- to avoid log(0)-----
    PhPs  += 1e-315 
    PhBs  += 1e-315

    #----------- Proper Motions  -----------
    PmPs   = LikeMemberPM(pm,u_pm,pi_pm_ps,mu_pm_ps,sg_pm_ps)
    PmBs   = LikeMemberPM(pm,u_pm,pi_pm_bs,mu_pm_bs,sg_pm_bs)
    #---------------------------------------------------
    ps_c  = PhPs*PmPs
    ps_b  = PhBs*PmBs

    lmm   = (ps_c*pi[1]) + (ps_b*(1-pi[1])) 
    
    #--------------- to avoid log(0)-----
    lmm  += 1e-315 
    lnm  += 1e-315
    #--------------------------------------
    lp    = np.log(((pi[0])*lnm)+((1-pi[0])*lmm))
    #------- Blobs ---------------
    PC1   = np.log(((1-pi[0])*lmm))-lp
    PC2   = np.log(pi[1]*ps_c) - np.log(lmm)
    
    # print(np.log(PmPs),np.log(PmBs),np.log(PhPs),np.log(PhBs),np.log(ps_c),np.log(ps_b),np.exp(PC1))
    return [lp,PC1,PC2]


def logLikeIndependentInc(datum,pi,pi_pm_ps,pi_pm_bs,
    pi_color,mu_color,vr_clr,
    mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
    coefs,phot_cov,alpha,
    stp,rg_color,evals):
    #------- Non-Members ----------
    lnm      = datum[0]
    #--------------- Datum -----------
    obs      = datum[1:(3+ph_dim)]
    uncert   = datum[(3+ph_dim):-(ph_dim+2)]
    index    = datum[-(ph_dim+2):]

    pm       = obs[0:2]
    phot     = obs[2:]
    u_pm     = uncert[0:2]
    u_phot   = np.zeros((ph_dim,ph_dim))
    u_phot[np.triu_indices(ph_dim)]   = uncert[2:]
    u_phot   = u_phot +np.triu(u_phot,1).T

    #--------------- Missing ---------
    w        = np.where(index[2:2+ph_dim] == 0)[0]
    #--------------------------------

    #---------------- Marginalization of color --------------
    PhPs,PhBs= MarginalColorInc(phot,u_phot,phot_cov,
                 alpha,coefs,pi_color,mu_color,vr_clr,rg_color,stp,w,evals)
    #--------------- to avoid log(0)-----
    PhPs  += 1e-315 
    PhBs  += 1e-315
    # #----------- Proper Motions  -----------
    PmPs   = LikeMemberPM(pm,u_pm,pi_pm_ps,mu_pm_ps,sg_pm_ps)
    PmBs   = LikeMemberPM(pm,u_pm,pi_pm_bs,mu_pm_bs,sg_pm_bs)
    #---------------------------------------------------
    ps_c  = PhPs*PmPs
    ps_b  = PhBs*PmBs

    lmm   = (ps_c*pi[1]) + (ps_b*(1-pi[1])) 
    #--------------- to avoid log(0)-----
    lmm  += 1e-315
    lnm  += 1e-315
    #--------------------------------------
    lp    = np.log(((pi[0])*lnm)+((1-pi[0])*lmm))
    #------- Blobs ---------------
    PC1   = np.log(((1-pi[0])*lmm))-lp
    PC2   = np.log(pi[1]*ps_c) - np.log(lmm)
    # print(np.log(pi[1]),PhPs,PmPs)
    # print(phot[0],np.log(PmPs),np.log(PmBs),np.log(PhPs),np.log(PhBs),np.log(ps_c),np.log(ps_b),np.exp(PC1))
    return [lp,PC1,PC2]