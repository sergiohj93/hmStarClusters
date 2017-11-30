from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from numba import jit
import scipy.stats as st
import scipy.special as sp
import scipy.linalg as lg
import math
from time import time

ph_dim = 5   #photometric dimensions
cf_dim = 4
dm     = 0.75
vr_clr_ll = 1e-6
vr_ph_ll  = 1e-10
id_etu    = [0,5,9,12,14]

# @jit
# def Curvature(coefs,x):
#     # This function computes the extrinsic curvature of each parametrized CMD.
#     # K = d2Y/(1+(dY**2))**3/2
#     curv = np.zeros(cf_dim)
#     for i in range(cf_dim):
#         d1 = np.polynomial.laguerre.lagder(coefs[i],m=1)
#         d2 = np.polynomial.laguerre.lagder(coefs[i],m=2)
#         y1 = np.polynomial.laguerre.lagval(x,d1)
#         y2 = np.polynomial.laguerre.lagval(x,d2)
#         curv[i] = y2/(np.sqrt(1.+(y1**2))**3)
#         print(curv[i])
#     return curv
# @jit
# def ArcLength(coefs,dom,stp=50):
#     # This function computes the arc length of each parametrized CMD.
#     # L = int(sqrt(1+(dY**2)))
#     dtau  = (dom[1]-dom[0])/stp
#     tau   = np.linspace((dom[0]+(dtau/2)),(dom[1]-(dtau/2)),(stp-1))
#     length = np.zeros(cf_dim)
#     for i in range(cf_dim):
#         d1 = np.polynomial.laguerre.lagder(coefs[i],m=1)
#         y1 = np.polynomial.laguerre.lagval(tau,d1)
#         length[i] = np.sum(np.sqrt(1.0+(y1**2)))*dtau
#     return length


@jit
def logPriorColor(pi_clr,mu_clr,vr_clr,
                  beta,V_0,rg_clr):
    lp_frac = st.dirichlet.logpdf(pi_clr,beta)
    lp_mu   =  -1.0*len(mu_clr)*np.log((rg_clr[1]-rg_clr[0]))
    lp_sg   =  np.sum(st.halfcauchy.logpdf(vr_clr,loc=vr_clr_ll,scale=V_0))
    return lp_frac+lp_sg+lp_mu

@jit
def logPriorAlpha(alpha,vr_alpha_hyp):
    # lp_a   =  np.sum(st.cauchy.logpdf(alpha,loc=0,scale=vr_alpha_hyp))
    return np.sum(st.cauchy.logpdf(alpha,loc=0,scale=vr_alpha_hyp))

# @jit
# def logPriorKnots(knotsp,rg):
#     lp = -len(knotsp)*np.log(rg[1]-rg[0])
#     # for i in range(len(iknots)-1):
#     #     lp +=-np.log(rg[1]-iknots[i]) 
#     return lp

@jit
def logPriorCoefs(coefp,mu_coefs,vr_coefs_hyp,rg_clr):
    # this prior ensures the isochrone to be as rect as possible
    # the two first chebyshev coeffcients are position and rotation.
    m = len(coefp[0])
    lp = 0.0
    for i in range(cf_dim):
        for j in range(m):
            # print(i,j)
            # print(coefp[i,j])
            # print(mu_coefs[i,j])
            # print(vr_coefs_hyp[j])
            lp += st.norm.logpdf(coefp[i,j],
                            loc=mu_coefs[i,j],
                            scale=np.sqrt(vr_coefs_hyp[j]))
    return lp
# @jit
# def logPriorCoefs(coefs,mu_coefs,vr_coefs_hyp):
#     # this prior ensures the isochrone to be as rect as possible
#     # the two first chebyshev coeffcients are position and rotation.
#     m = len(coefs[0])
#     lp = 0.0
#     for i in range(cf_dim):
#         for j in range(m):
#             lp += st.cauchy.logpdf(coefs[i,j],loc=mu_coefs[i,j],scale=vr_coefs_hyp)

#     return lp[0]

# @jit
def logCovariancePhot(S,v,A):
    lpdf = np.sum(map(lambda x,y:st.halfcauchy.logpdf(x,loc=vr_ph_ll,scale=y),np.diag(S),A))
    # lpdf = logHuang(S,v,A)
    return lpdf

@jit
def logHuang(S,v,A):
    # Huang & Wand 2013 Bayesian Analysis v8 No. 2 p439
    # S must be +definite (tu is the cholesky decomposition)
    # A is a vector of lenght dim S
    # v is dof
    #----- computes determinant and inverse---
    ldS = np.linalg.slogdet(S)[1]
    iS = lg.inv(S)
    #--------------------
    p  = np.shape(iS)[0]
    c  = -0.5*(v+(2*p))*ldS
    pr =0.0
    for i in range(p):
        pr += (-0.5*(p+v))*np.log(v*iS[i,i]+(1.0/(A[i]**2)))
    lpdf= c+pr
    return lpdf

#------------ Fractions -----------
@jit
def logPriorFractions(theta,alpha):
    pi     = np.zeros((2,2))
    pi[0]  = [theta[0],1-theta[0]]    # Field
    pi[1]  = [theta[1],1-theta[1]]    # Isochrone 
    lp0    = st.dirichlet.logpdf(pi[0],alpha[0])
    lp1    = st.dirichlet.logpdf(pi[1],alpha[1])
    return lp0+lp1

# ------------------------ Proper Motions -----------------------
@jit
def logPriorProperMotion(pi_pm,mu_pm,sg_pm,alpha_pm,mean_pm_hyp,sg_pm_hyp,nu,A):
    #---- This works for binaries and non binaries
    lfr = st.dirichlet.logpdf(pi_pm,alpha_pm)
    lmu = st.multivariate_normal.logpdf(mu_pm,mean_pm_hyp,sg_pm_hyp)
    # lmu2 = st.multivariate_normal.logpdf(mu_pm,mean_pm_hyp,sg_pm_hyp)
    lsg = 0
    for i in range(len(pi_pm)):
        lsg += logHuang(sg_pm[i],nu,A)
        # print(sg_pm[i])
        # print(logHuang(sg_pm[i],nu,A))
    return lfr+lmu+lsg

# @jit
# def logInvWishartTU(A,sigma,n):
#     #------ turns cholesky into matrix form---
#     tu = np.zeros((2,2))
#     tu[0,0]=A[0]
#     tu[0,1]=A[1]
#     tu[1,1]=A[2]
#     A  = np.dot(tu.T,tu)
#     #----- computes determinant and inverse---
#     dA = (A[0,0]*A[1,1])-(A[0,1]*A[1,0])
#     iA = np.zeros((2,2)) 
#     iA[0,0]=  sg[1,1]
#     iA[0,1]= -sg[0,1]
#     iA[1,0]= -sg[1,0]
#     iA[1,1]=  sg[0,0] 
#     iA = iA/det
#     #-------------
#     gp = np.sqrt((np.pi))*sp.gamma(n)*sp.gamma(n-0.5)
#     dS = (sigma[0,0]*sigma[1,1])-(sigma[0,1]*sigma[1,0])
#     t  = np.trace(np.dot(sigma,iA))
#     pdf= (dA**(-(n+2+1)/2))*((dS**(n/2))*np.exp(-0.5*t))/((2**n)*gp)
#     return np.log(pdf)


# @jit
# def logHuangTU2X2(tu,v,A):
#     # Huang & Wand 2013 Bayesian Analysis v8 Nu. 2 p439
#     # S must be +definite
#     # A is a vector of lenght dim S
#     # v is dof
#     #------ turns cholesky into matrix form---
#     S = np.zeros((2,2))
#     S[0,0]= tu[0]
#     S[0,1]= tu[1]
#     S[1,1]= tu[2]
#     S[1,0]= tu[1]
#     #----- computes determinant and inverse---
#     dS = (S[0,0]*S[1,1])-(S[0,1]*S[1,0])
#     iS = np.zeros((2,2)) 
#     iS[0,0]=  S[1,1]
#     iS[0,1]= -S[0,1]
#     iS[1,0]= -S[1,0]
#     iS[1,1]=  S[0,0] 
#     iS = iS/dS
#     #--------------------
#     p  = np.shape(iS)[0]
#     c  = (dS)**(-1*(v+2*p)/2)
#     pr =1
#     for i in range(p):
#         pr = pr*(v*iS[i,i]+(1/(A[i]**2)))**(-(p+v)/2)
#     pdf= c*pr
#     return np.log(pdf)


# @jit
# def logHuang(S,v,A):
#     # Huang & Wand 2013 Bayesian Analysis v8 Nu. 2 p439
#     # S must be +definite
#     # A is a vector of lenght dim S
#     # v is dof
#     p  = np.shape(S)[0]
#     iS = np.linalg.inv(S)
#     dS = np.linalg.det(S)
#     c  = (dS)**(-1*(v+2*p)/2)
#     pr =1
#     for i in range(p):
#         pr = pr*(v*iS[i,i]+(A[i]**-2))**(-(p+v)/2)
#     pdf= c*pr
#     return np.log(pdf)

# @jit
# def isPosDefChol2x2(a):
#     # if matrix is positive definite, the its inverse is too.
#     #---Solve the inequality system zT(RR*)z>0----
#     #--- For R in 2x2 matrices----
#     #--Matrix is upper triangular with upper right cornen y,
#     #---upper left x and lower right z
#     #x=a[0]
#     #y=a[1]
#     #z=a[2]
#     #cond1 = (x < 0  and ((y < 0 and z != 0) or y == 0  or (y > 0 and z != 0 )))
#     #cond2 = (x == 0 and  (y < 0 or  (y == 0 and (z < 0 or z > 0)) or  y > 0))
#     #cond3 = (x > 0  and ((y < 0 and z != 0) or y == 0  or (y > 0 and z != 0 )))
#     # cond = (a[0] > 0  and a[2] > 0)
#     cond = (a[0]*a[2]-a[1]**2) > 0
#     return cond

# def isPosDefPhotCov(TU):
#     """
#     Since phot_cov is the cholesky composition of TU -> P = TU.T * TU,
#     then for phot_cov to be positive definite, TU must be real and non-singular.
#     It is by construction, real.
#     A square matrix is nonsingular iff its determinant is nonzero (Lipschutz 1991, p. 45) 
#     The determinant of the (upper) triangular matrix is the product of its entries on the main diagonal.
#     Therefore, phot_cov will be positive definite iff the entries of the diagonal of TU are all non-zero.
#     Also, the determinant must be positive so, since the determinant of the product of two matrices is
#     the product of their determinants, the determinant of TU could be negative.
#     However, in the cholesky decomposition, all elements in the diagonal must be positive.
#     """

#     if np.any(np.diag(TU)) < 1e-10: # This is the positive non-zero condition zero means 1e-10
#         # print("Values less than 1e-10")
#         return False

#     if np.prod(np.diag(TU)) < 1e-50 :   # The determinant of phot_cov > 1e-100
#         # print("Diagonal product less than 1e-50")
#         return False
#     # if np.linalg.cond(np.dot(TU.T,TU)) > 1e15:
#     #     # To have at least 3 precision digits in the solution (with float64) 
#     #     # print(A.dtype)
#     #     # print(np.linalg.cond(A))
#     #     # stop
#     #     return False
#     return True


@jit
def isPosDefChol(etu):
    if np.any(etu[id_etu] < vr_ph_ll):
        return False
    return True

# @jit
def isPosDef(A):
    if np.any(np.diag(A) < 1e-10):
        return False
    try:
        chol=lg.cholesky(A)
    except lg.LinAlgError:
        # print("Not PSD")
        return False
    else:
        # To have at least 3 precision digits in the solution (with float64) 
        # print(A.dtype)
        # print(np.linalg.cond(A))
        # stop
        if (np.linalg.slogdet(A)[1] > -200) :
            return True
        else:
            return False

    # return np.linalg.slogdet(A)[0]==1 #and np.linalg.slogdet(A)[1]> -10
@jit
def valid_erf(mu,vr,rg):
    tr = np.empty(len(mu))
    for i in range(len(mu)):
        de = np.sqrt(2*vr[i])
        tr[i] = math.erf((rg[1]-mu[i])/de)-math.erf((rg[0]-mu[i])/de)
    if np.any(tr == 0):
        return False
    else:
        return True


@jit
def Support(pi,pi_pm_ps,pi_pm_bs,pi_clr,mu_clr,vr_clr,sg_pm_ps,sg_pm_bs,coefp,etu_cov,rg_clr):
    # #------- Debugging --------------
    # if ((pi[0] > 1) or (pi[0] < 0)): print("ERROR Pi0")
    # if ((pi[1] > 1) or (pi[1] < 0)): print("ERROR Pi1")
    # if ((pi[2] > 1) or (pi[2] < 0)): print("ERROR Pi2")
    # if ((pi[3] > 1) or (pi[3] < 0)): print("ERROR Pi3")
    # if ((pi_clr[0] > 1) or (pi_clr[0] < 0)): print("ERROR PiClr0")
    # if ((pi_clr[1] > 1) or (pi_clr[1] < 0)): print("ERROR Piclr1")
    # if ((pi_clr[2] > 1) or (pi_clr[2] < 0)): print("ERROR Piclr2")
    # if (mu_clr[0] < rg_clr[0] or mu_clr[0] > rg_clr[1]): print("ERROR Color mu0")
    # if (mu_clr[1] < rg_clr[0] or mu_clr[1] > rg_clr[1]): print("ERROR Color mu1")
    # if (mu_clr[2] < rg_clr[0] or mu_clr[2] > rg_clr[1]): print("ERROR Color mu2")
    # if (vr_clr[0] < 0): print("ERROR Color vr0")
    # if (vr_clr[1] < 0): print("ERROR Color vr1")
    # if (vr_clr[2] < 0): print("ERROR Color vr2")
    # if not isPosDef(sg_pm_ps[0]): print("ERROR SG 00")
    # if not isPosDef(sg_pm_ps[1]): print("ERROR SG 01")
    # if not isPosDef(sg_pm_bs[0]): print("ERROR SG 10")
    # if not isPosDef(sg_pm_bs[1]): print("ERROR SG 11")
    # if not isPosDef(phot_cov): print("ERROR phot_cov")
    # if (np.any(pi > 1) or np.any(pi <= 0)): print("ERROR Pi")
    # if (np.any(pi_pm_ps > 1) or np.any(pi_pm_ps <= 0)): print("ERROR Pi Ps")
    # if (np.any(pi_pm_bs > 1) or np.any(pi_pm_bs <= 0)): print("ERROR Pi Bs")
    # if (np.any(pi_clr > 1) or np.any(pi_clr <= 0)): print("ERROR Pi Color")
    # if not valid_erf(mu_clr,vr_clr,rg_clr): print("ERROR mu_clr")
    # if np.any(vr_clr <= vr_clr_ll): print("ERROR Variance color")
    # if not isPosDef(sg_pm_ps[0]): print("ERROR pm ps 1")
    # if not isPosDef(sg_pm_ps[1]): print("ERROR pm ps 2")
    # if not isPosDef(sg_pm_ps[2]): print("ERROR pm ps 3")
    # if not isPosDef(sg_pm_bs[0]): print("ERROR pm bs 1")
    # if not isPosDef(sg_pm_bs[1]): print("ERROR pm bs 2")
    # if not isPosDefChol(tu_cov): print("ERROR phot_cov")

# #------- Support of Parameters --------------
    if (np.any(pi > 1) or np.any(pi <= 0)): return -np.inf
    if (np.any(pi_pm_ps > 1) or np.any(pi_pm_ps <= 0)): return -np.inf
    if (np.any(pi_pm_bs > 1) or np.any(pi_pm_bs <= 0)): return -np.inf
    if (np.any(pi_clr > 1) or np.any(pi_clr <= 0)): return -np.inf
    if not valid_erf(mu_clr,vr_clr,rg_clr): return -np.inf
    if np.any(vr_clr <= vr_clr_ll): return -np.inf
    if not isPosDef(sg_pm_ps[0]): return -np.inf
    if not isPosDef(sg_pm_ps[1]): return -np.inf
    if not isPosDef(sg_pm_ps[2]): return -np.inf
    if not isPosDef(sg_pm_ps[3]): return -np.inf
    if not isPosDef(sg_pm_bs[0]): return -np.inf
    if not isPosDef(sg_pm_bs[1]): return -np.inf
    # if not isPosDef(sg_pm_ps[2]): return -np.inf
    # if (np.any(knotsp > rg_clr[1]) or np.any(knotsp < rg_clr[0])): return -np.inf
    # if np.any(np.diff(knotsp) < 0.1): return -np.inf
    if not isPosDefChol(etu_cov): return -np.inf
    pass

@jit
def logPriors(pi,pi_pm_ps,pi_pm_bs,pi_color,mu_color,vr_color,
    mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
    coefp,phot_cov,alpha,
    theta_hyp,theta_Ps_hyp,theta_Bs_hyp,
    rg_clr,theta_clr_hyp,vr_clr_hyp,
    mu_pm_hyp,sg_pm_hyp,
    mu_coefs,vr_coefs_hyp,
    nu,A_phot,A_pm,vr_alpha_hyp):
    
#----- Compute priors ----------
    lp_fr      = logPriorFractions(pi,theta_hyp)
    # print(lp_fr)
    lp_clr     = logPriorColor(pi_color,mu_color,vr_color,theta_clr_hyp,vr_clr_hyp,rg_clr) 
    # print(lp_clr)
    # lp_knots   = logPriorKnots(knotsp,rg_clr)
    # print (lp_knots)
    lp_coefs   = logPriorCoefs(coefp,mu_coefs,vr_coefs_hyp,rg_clr)
    # print (lp_coefs)
    lp_photcov = logCovariancePhot(phot_cov,nu,A_phot)
    # print (lp_photcov)
    lp_pmPs    = logPriorProperMotion(pi_pm_ps,mu_pm_ps,sg_pm_ps,theta_Ps_hyp,mu_pm_hyp,sg_pm_hyp,nu,A_pm)
    # print(lp_pmPs)
    lp_pmBs    = logPriorProperMotion(pi_pm_bs,mu_pm_bs,sg_pm_bs,theta_Bs_hyp,mu_pm_hyp,sg_pm_hyp,nu,A_pm)
    # print(lp_pmBs)
    lp_alpha   = logPriorAlpha(alpha,vr_alpha_hyp)
    # print(lp_alpha)
    lp_prior   = lp_fr+lp_clr+lp_coefs+lp_photcov+lp_pmPs+lp_pmBs+lp_alpha#+lp_knots
    # print(lp_fr,lp_clr,lp_coefs,lp_photcov,lp_pmPs1,lp_pmPs2,lp_pmBs1,lp_pmBs2)
    return lp_prior

