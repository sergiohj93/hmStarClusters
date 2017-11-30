from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import numpy as np
from numba import cuda,float32,float64,int8,int16
import scipy.stats as st
import scipy.special as sp
import scipy.linalg as lg
from scipy.interpolate import splev
import math
from time import time
from CudaFunctions import *

ph_dim = 5   #photometric dimensions
cf_dim = 4
id_clr = 0       # index of color parameter
dm     = 0.75
vPi    = 3.14159265359
e      = 2.718281828
'''max1d  = (99)
max2d  = (99,99)
max2dEvals = (300,ph_dim)'''
max1d  = (110)
max2d  = (61,ph_dim)
max1dEvals = (300)
max2dEvals = (300,ph_dim)
'''max1d  = (99)
max2d  = (70,25)
max2dEvals = (300,10)'''

####################### LIKELIHOOD FUNCTIONS ###########################

#IMPORTANT: The functions that have to return new arrays don't return anything
#and write their results in the last parameter (that contain the array created before).

@cuda.jit(device=True)
def CreateU_phot(uncert, u_phot2):
    u_phot   = cuda.local.array(shape=(ph_dim, ph_dim), dtype=float32)
    zeros2d(u_phot)
    k = 0
    for i in range(ph_dim):
        for j in range(i, ph_dim):
            u_phot[i][j] = uncert[k]
            k += 1
             
    for i in range(ph_dim):
        for j in range(i+1, ph_dim):
            u_phot2[j][i] = u_phot[i][j]
    
    for i in range(ph_dim):
        for j in range(ph_dim):
            u_phot2[i][j] = u_phot2[i][j] + u_phot[i][j]


@cuda.jit(device=True)            
def CalculateCtr_i(isg, A):
    #Realizes the next calcule for MarginalColor and MarginalColorInc:
    '''mha   = np.array([np.sqrt(x.dot(isg).dot(x.T)) for x in A])
    ctr_i = np.argmin(mha)'''

    dot = cuda.local.array(shape=max1d, dtype=float32)
    dot = dot[0:len(isg[0])]
    minMha = 99999999
    ctr_i = -1
    for i in range(len(A)):
        x = A[i]
        dotVecMat(x,isg,dot)
        mha = math.sqrt(dotVec(dot,x.T))
        if mha < minMha:
            minMha = mha
            ctr_i = i
    return ctr_i

@cuda.jit(device=True)
def CalculateBalls(isg, bins, bound, A):
    #Realizes the next calcule for MarginalColor and MarginalColorInc:
    '''mha   = np.array([np.sqrt(x.dot(isg).dot(x.T)) for x in sub2])
    ball  = bins[np.where(mha <= bound),id_clr].copy()'''
    
    dot = cuda.local.array(shape=max1d, dtype=float32)
    dot = dot[0:len(isg[0])]
    minBall = 99999999
    maxBall = -99999999
    for i in range(len(A)):
        x = A[i]
        dotVecMat(x,isg,dot)
        mha = math.sqrt(dotVec(dot,x.T))
        if mha <= bound:
            binsItem = bins[i,id_clr]
            if binsItem < minBall:
                minBall = binsItem
            if binsItem > maxBall:
                maxBall = binsItem
    return minBall,maxBall

def mcsn(x,sigma_obs,sigma_cls,isgs,lden,alpha):
    gamma     = np.vstack(([0]*len(x),alpha.dot(sigma_cls.dot(isgs))))
    nu        = np.zeros(2)
    Delta     = np.identity(2)
    Delta[1,1]= (1.0+alpha.dot(sigma_cls.dot(alpha))
               -alpha.dot(np.dot(sigma_obs,np.dot(isgs,np.dot(sigma_cls,alpha)))))
    return csn_pdf(x,sigma_obs+sigma_cls,isgs,lden,gamma,nu,Delta)


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


@cuda.jit(device=True)
def p_color(color,pi,mu,vr,rg,ps):
    # There are for normal distributions, they are truncated.
    #-------- denominators without sqrt(pi) ------
    de0 = math.sqrt(2*vr[0])
    de1 = math.sqrt(2*vr[1])
    de2 = math.sqrt(2*vr[2])
    de3 = math.sqrt(2*vr[3])
    de4 = math.sqrt(2*vr[4])
    #-------- exponentials divided by denominators----
    #-------------- creation of variables-------------
    length = len(color)
    sub = cuda.local.array(shape=max1d, dtype=float32)
    sub = sub[0:length]
    
    powS = cuda.local.array(shape=max1d, dtype=float32)
    powS = powS[0:length]
    
    div = cuda.local.array(shape=max1d, dtype=float32)
    div = div[0:length]
    
    neg = cuda.local.array(shape=max1d, dtype=float32)
    neg = neg[0:length]
    
    exp = cuda.local.array(shape=max1d, dtype=float32)
    exp = exp[0:length]
    
    ps0 = cuda.local.array(shape=max1d, dtype=float32)
    ps0 = ps0[0:length]
    
    ps1 = cuda.local.array(shape=max1d, dtype=float32)
    ps1 = ps1[0:length]
    
    ps2 = cuda.local.array(shape=max1d, dtype=float32)
    ps2 = ps2[0:length]
    
    ps3 = cuda.local.array(shape=max1d, dtype=float32)
    ps3 = ps3[0:length]
    
    ps4 = cuda.local.array(shape=max1d, dtype=float32)
    ps4 = ps4[0:length]
    
    #-------------- operations----------------------
    subValue(color, mu[0], sub)
    powValue(sub, 2, powS)
    divValue(powS, 2*vr[0], div)
    negVec(div, neg)
    expVec(neg, exp)
    divValue(exp, de0, ps0)
    
    subValue(color, mu[1], sub)
    powValue(sub, 2, powS)
    divValue(powS, 2*vr[1], div)
    negVec(div, neg)
    expVec(neg, exp)
    divValue(exp, de1, ps1)
    
    subValue(color, mu[2], sub)
    powValue(sub, 2, powS)
    divValue(powS, 2*vr[2], div)
    negVec(div, neg)
    expVec(neg, exp)
    divValue(exp, de2, ps2)

    subValue(color, mu[3], sub)
    powValue(sub, 2, powS)
    divValue(powS, 2*vr[3], div)
    negVec(div, neg)
    expVec(neg, exp)
    divValue(exp, de3, ps3)

    subValue(color, mu[4], sub)
    powValue(sub, 2, powS)
    divValue(powS, 2*vr[4], div)
    negVec(div, neg)
    expVec(neg, exp)
    divValue(exp, de4, ps4)    
    
    #-------- truncation factors, wihtout 0.5------
    tr0 = math.erf((rg[1]-mu[0])/de0)-math.erf((rg[0]-mu[0])/de0)
    tr1 = math.erf((rg[1]-mu[1])/de1)-math.erf((rg[0]-mu[1])/de1)
    tr2 = math.erf((rg[1]-mu[2])/de2)-math.erf((rg[0]-mu[2])/de2)
    tr3 = math.erf((rg[1]-mu[3])/de3)-math.erf((rg[0]-mu[3])/de3)
    tr4 = math.erf((rg[1]-mu[4])/de4)-math.erf((rg[0]-mu[4])/de4)
    #-------------- creation of variables-------------
    ps0op = cuda.local.array(shape=max1d, dtype=float32)
    ps0op = ps0op[0:length]
    
    ps1op = cuda.local.array(shape=max1d, dtype=float32)
    ps1op = ps1op[0:length]
    
    ps2op = cuda.local.array(shape=max1d, dtype=float32)
    ps2op = ps2op[0:length]
    
    ps3op = cuda.local.array(shape=max1d, dtype=float32)
    ps3op = ps3op[0:length]
    
    ps4op = cuda.local.array(shape=max1d, dtype=float32)
    ps4op = ps4op[0:length]
    
    #-------------- operations----------------------
    divValue(ps0, tr0, ps0op)
    multValue(ps0op, pi[0], ps0op)
    
    divValue(ps1, tr1, ps1op)
    multValue(ps1op, pi[1], ps1op)
    
    divValue(ps2, tr2, ps2op)
    multValue(ps2op, pi[2], ps2op)
    
    divValue(ps3, tr3, ps3op)
    multValue(ps3op, pi[3], ps3op)
    
    divValue(ps4, tr4, ps4op)
    multValue(ps4op, pi[4], ps4op)
    
    add1d(ps0op, ps1op, ps)
    add1d(ps, ps2op, ps)
    add1d(ps, ps3op, ps)
    add1d(ps, ps4op, ps)   
    
    #return value with common factors 0.5 and sqrt(pi)
    multValue(ps, 2/math.sqrt(vPi), ps)
    #-------
    # ps  = 0.99*(1.0/(rg[1]-rg[0]))+0.01*ps
    

@cuda.jit(device=True)
def LikeMemberPMone(obs,uncert,mu,sg_prev):
    #------------ ads uncertainty of data
    uncertMat = cuda.local.array(shape=max2d, dtype=float32)
    uncertMat = uncertMat[0:len(uncert),0:len(uncert)]
    zeros2d(uncertMat)
    diag(uncert, uncertMat)
    
    sg = cuda.local.array(shape=max2d, dtype=float32)
    sg = sg[0:len(sg_prev),0:len(sg_prev[0])]
    add2d(sg_prev, uncertMat, sg)
    
    #------ computes determinant and inverse
    det = Determinant2x2(sg)
    isg = cuda.local.array(shape=(2,2), dtype=float32)
    Inverse2x2(sg,isg)   
    
    #------- computes density -----
    den = 2*math.sqrt(det)*vPi
    #print(isg,obs-mu,np.dot(isg,obs-mu))
    sub = cuda.local.array(shape=max1d, dtype=float32)
    sub = sub[0:len(obs)]
    sub1d(obs,mu,sub)
    dot = cuda.local.array(shape=max1d, dtype=float32)
    dot = dot[0:len(isg)]
    dotMatVec(isg,sub,dot)
    z = dotVec(sub,dot)
    #print(isg,obs-mu,np.dot(isg,obs-mu)) 
    ps  = math.exp(-0.5*(z))/den
    # print(ps,st.multivariate_normal.pdf(obs, mean=mu, cov=sg))
    return ps

@cuda.jit(device=True)
def LikeMemberPM(obs,uncert,pi,mu,sg):
    ll = 0
    for i in range(len(pi)):
        ll += pi[i]*LikeMemberPMone(obs,uncert,mu,sg[i])
    return ll

#Return ps and bs
@cuda.jit(device=True)
def LikeMemberPH(m_obs,sg_obs,sg_cls,isg,alpha,color,colorb,coefs,wid,degspl,id_mag,ps,bs):
    #----color and colorb must have same length
    n = len(color)
    #-------common variables----
    
    
    sg = cuda.local.array(shape=max2d, dtype=float32)
    sg = sg[0:len(sg_obs),0:len(sg_obs[0])]
    add2d(sg_obs, sg_cls, sg)
    lden = math.log(Determinant5x5(sg)) + (len(m_obs)*math.log(2*vPi))
    
    x = cuda.local.array(shape=max1d, dtype=float32)
    x = x[0:len(m_obs)]
    
    dot = cuda.local.array(shape=max1d, dtype=float32)
    dot = dot[0:len(isg)]
    for i in range(n):  
        m_true = cuda.local.array(shape=ph_dim, dtype=float32)
        #-------------------- Spline evaluation ----------------------------
        m_true[0] = color[i]
        m_true[1] = SplEv(color[i],coefs[0][0],coefs[0][1],degspl)
        m_true[2] = SplEv(color[i],coefs[1][0],coefs[1][1],degspl)
        m_true[3] = SplEv(color[i],coefs[2][0],coefs[2][1],degspl)
        m_true[4] = SplEv(color[i],coefs[3][0],coefs[3][1],degspl)

        #------------------------------------------------------
        sub1d(m_obs, m_true, x)
        #------------------------------------------------------
        # ps[i] = mcsn(x,sg_obs,sg_cls,isg,lden,alpha)
        dotMatVec(isg, x, dot)
        ps[i] = math.exp(-0.5*(dotVec(x,dot)+lden))
        # print(np.sqrt(np.dot(x.T,isg.dot(x))))
        # ps0 = st.multivariate_normal.pdf(m_obs,mean=m_true, cov=sg)
        #-------- Binaries -------
        #--------------------Spline evaluation ----------------------------
        m_true[0] = colorb[i]
        m_true[1] = SplEv(colorb[i],coefs[0][0],coefs[0][1],degspl)
        m_true[2] = SplEv(colorb[i],coefs[1][0],coefs[1][1],degspl)
        m_true[3] = SplEv(colorb[i],coefs[2][0],coefs[2][1],degspl)
        m_true[4] = SplEv(colorb[i],coefs[3][0],coefs[3][1],degspl)
        #------------------------------------------------------       
        subItems1d(m_true,id_mag,dm)
        
        sub1d(m_obs, m_true, x)
        dotMatVec(isg, x, dot)
        bs[i] = math.exp(-0.5*(dotVec(x,dot)+lden))
        # bs[i] = st.multivariate_normal.pdf(m_obs,mean=m_true, cov=sg)

#Return ps and bs
@cuda.jit(device=True)
def LikeMemberPHInc(m_obs,sg_obs,sg_cls,isg,alpha,color,colorb,coefs,wid,idx,degspl,id_mag,ps,bs):
    #----color and colorb must have same length
    n = len(color)
    sg = cuda.local.array(shape=max2d, dtype=float32)
    sg = sg[0:len(sg_obs),0:len(sg_obs[0])]
    add2d(sg_obs, sg_cls, sg)
    
    if len(sg) == 4:
        lden = math.log(Determinant4x4(sg)) + (len(m_obs)*math.log(2*vPi))
    elif len(sg) == 3:
        lden = math.log(Determinant3x3(sg)) + (len(m_obs)*math.log(2*vPi))
    elif len(sg) == 2:
        lden = math.log(Determinant2x2(sg)) + (len(m_obs)*math.log(2*vPi))

    
    x = cuda.local.array(shape=max1d, dtype=float32)
    x = x[0:len(m_obs)]
    
    dot = cuda.local.array(shape=max1d, dtype=float32)
    dot = dot[0:len(isg)]
    for i in range(n): 
        m_true = cuda.local.array(shape=ph_dim, dtype=float32)
        #-------------------- Spline evaluation ----------------------------
        m_true[0] = color[i]
        m_true[1] = SplEv(color[i],coefs[0][0],coefs[0][1],degspl)
        m_true[2] = SplEv(color[i],coefs[1][0],coefs[1][1],degspl)
        m_true[3] = SplEv(color[i],coefs[2][0],coefs[2][1],degspl)
        m_true[4] = SplEv(color[i],coefs[3][0],coefs[3][1],degspl)
        #------------------------------------------------------
        getItems1d(m_true, idx, x)
        sub1d(m_obs, x, x)
        # ps[i]  = mcsn(x,sg_obs,sg_cls,isg,lden,alpha)
        dotMatVec(isg, x, dot)
        ps[i] = math.exp(-0.5*(dotVec(x,dot)+lden))
        # print(np.sqrt(np.dot(x.T,isg.dot(x))))
        # ps0 = st.multivariate_normal.pdf(m_obs,mean=m_true[idx], cov=sg)

        #-------- Binaries -------

        #--------------------Spline evaluation ----------------------------
        m_true[0] = colorb[i]
        m_true[1] = SplEv(colorb[i],coefs[0][0],coefs[0][1],degspl)
        m_true[2] = SplEv(colorb[i],coefs[1][0],coefs[1][1],degspl)
        m_true[3] = SplEv(colorb[i],coefs[2][0],coefs[2][1],degspl)
        m_true[4] = SplEv(colorb[i],coefs[3][0],coefs[3][1],degspl)
        #------------------------------------------------------
        #print(m_true)
        subItems1d(m_true,id_mag,dm)
        #print(m_true)
        getItems1d(m_true, idx, x)
        sub1d(m_obs, x, x)
        dotMatVec(isg, x, dot)
        bs[i] = math.exp(-0.5*(dotVec(x,dot)+lden))
        # bs[i] = st.multivariate_normal.pdf(m_obs,mean=m_true[idx], cov=sg)

@cuda.jit(device=True)
def MarginalColor(phot,sg_obs,sg_cls,alpha,coefs,pi_color,mu_color,vr_clr,
                    rg_color,stp,evals,degspl,id_mag):
    dclr = 0.1
    dmha = 3.0    
    
    sg = cuda.local.array(shape=max2d, dtype=float32)
    sg = sg[0:len(sg_obs),0:len(sg_obs[0])]
    add2d(sg_obs, sg_cls, sg)
    isg = cuda.local.array(shape=max2d, dtype=float32)
    isg = isg[0:len(sg_obs),0:len(sg_obs[0])]
    Inverse5x5(sg,isg)
    ###################### SINGLE MASS STARS ####################################################
    sub1 = cuda.local.array(shape=max2dEvals, dtype=float32)
    sub1 = sub1[0:len(evals),0:len(evals[0])]
    subMatVec(evals,phot,sub1)
    ctr_i = CalculateCtr_i(isg, sub1)
    
    ctr = cuda.local.array(shape=max1d, dtype=float32)
    ctr = ctr[0:len(evals[0])]
    copy1d(evals[ctr_i],ctr)
    minRange = max(0,ctr_i-25)
    maxRange = min(len(evals),ctr_i+25)
    
    bins = cuda.local.array(shape=max2d, dtype=float32)
    bins = bins[0:(maxRange-minRange),0:len(evals[0])]    
    copy2d(evals[minRange:maxRange],bins)
    
    sub2 = cuda.local.array(shape=max2d, dtype=float32)
    sub2 = sub2[0:len(bins),0:len(bins[0])]
    subMatVec(bins,ctr,sub2)
    
    minBall,maxBall = CalculateBalls(isg, bins, dmha, sub2)
    dom = cuda.local.array(shape=(2), dtype=float32)
    #dom[0] = min(ctr[id_clr]-0.05,minBall)
    #dom[1] = max(ctr[id_clr]+0.05,maxBall)
    dom[0] = minBall
    dom[1] = maxBall + dclr

    ###################### BINARIES #############################################################

    nzc   = int(1.0/((rg_color[1]-rg_color[0])/len(evals)))
    #---- true phot bewteen ctr_clr and ctr_clr + 0.5
    minRange = max(0,ctr_i-int(0.5*nzc))
    maxRange = min(len(evals),ctr_i+nzc)
    bbins = cuda.local.array(shape=max2d, dtype=float32)
    bbins = bbins[0:(maxRange-minRange),0:len(evals[0])]     
    
    copy2d(evals[minRange:maxRange],bbins)
    # bbins = evals.copy()
    # #---- true binaries phot
    idxBbins = cuda.local.array(shape=max1dEvals, dtype=int16)
    idxBbins = idxBbins[0:len(bbins)]
    idxFull(len(bbins),idxBbins)
    subItems2d(bbins,idxBbins,id_mag,dm)
    
    
    # #------ distance ---------------------------------
    sub3 = cuda.local.array(shape=max2d, dtype=float32)
    sub3 = sub3[0:len(bbins),0:len(bbins[0])]
    subMatVec(bbins,phot,sub3)
    ctrb_i = CalculateCtr_i(isg, sub3)
    
    ctrb = cuda.local.array(shape=max1d, dtype=float32)
    ctrb = ctrb[0:len(bbins[0])]
    copy1d(bbins[ctrb_i],ctrb)
    minRange = max(0,ctrb_i-25)
    maxRange = min(len(bbins),ctrb_i+25)

    bbinsR = cuda.local.array(shape=max2d, dtype=float32)
    bbinsR = bbinsR[0:(maxRange-minRange),0:len(bbins[0])] 
    copy2d(bbins[minRange:maxRange],bbinsR)
    

    sub4 = cuda.local.array(shape=max2d, dtype=float32)
    sub4 = sub4[0:len(bbinsR),0:len(bbinsR[0])]
    subMatVec(bbinsR,ctrb,sub4)

    minBall,maxBall = CalculateBalls(isg, bbinsR, dmha, sub4)
    
    domb = cuda.local.array(shape=(2), dtype=float32)
    #domb[0] = min(ctrb[id_clr]-0.05,minBall)
    #domb[1] = max(ctrb[id_clr]+0.05,maxBall)
    domb[0] = minBall
    domb[1] = maxBall + dclr
    ############################################################################################

    # #----- ensures that the domain is inside the limits of color  
    dom[0] = max( dom[0],rg_color[0])
    dom[1] = min( dom[1],rg_color[1])
    domb[0] = max( domb[0],rg_color[0])
    domb[1] = min( domb[1],rg_color[1])    
    
    
    dtau   = (dom[1]-dom[0])/(stp-1)
    dtaub  = (domb[1]-domb[0])/(stp-1)
    
    tau = cuda.local.array(shape=max1d, dtype=float32)
    tau = tau[0:int(stp)]
    linspace(dom[0],dom[1],stp,tau)
    taub = cuda.local.array(shape=max1d, dtype=float32)
    taub = taub[0:int(stp)]
    linspace(domb[0],domb[1],stp,taub)

    pclr = cuda.local.array(shape=max1d, dtype=float32)
    pclr = pclr[0:len(tau)]
    pclrb = cuda.local.array(shape=max1d, dtype=float32)
    pclrb = pclrb[0:len(taub)]
    p_color(tau,pi_color,mu_color,vr_clr,rg_color,pclr)
    p_color(taub,pi_color,mu_color,vr_clr,rg_color,pclrb)
    
    ps = cuda.local.array(shape=max1d, dtype=float64)
    ps = ps[0:len(tau)]
    bs = cuda.local.array(shape=max1d, dtype=float64)
    bs = bs[0:len(taub)]
    

    LikeMemberPH(phot,sg_obs,sg_cls,isg,alpha,tau,taub,coefs,rg_color,degspl,id_mag,ps,bs)

    PhPs   = dotVec(pclr,ps)*dtau
    PhBs   = dotVec(pclrb,bs)*dtaub


    return PhPs,PhBs

@cuda.jit(device=True)
def MarginalColorInc(phot_full,sg_obs_full,sg_cls_full,alpha_full,coefs,pi_color,mu_color,vr_clr,
                    rg_color,stp,idx,evals,degspl,id_mag):
    dclr = 0.1
    dmha = 3.5
    
    #------- missing ---------
    alpha = cuda.local.array(shape=max1d, dtype=float32)
    alpha = alpha[0:len(idx)] 
    getItems1d(alpha_full,idx,alpha)
    
    phot = cuda.local.array(shape=max1d, dtype=float32)
    phot = phot[0:len(idx)] 
    getItems1d(phot_full,idx,phot)
    
    sg_obs = cuda.local.array(shape=max2d, dtype=float32)
    sg_obs = sg_obs[0:len(idx),0:len(idx)]
    getItems2d(sg_obs_full,idx,idx,sg_obs)
    
    sg_cls = cuda.local.array(shape=max2d, dtype=float32)
    sg_cls = sg_cls[0:len(idx),0:len(idx)]
    getItems2d(sg_cls_full,idx,idx,sg_cls)
    
    sg = cuda.local.array(shape=max2d, dtype=float32)
    sg = sg[0:len(sg_obs),0:len(sg_obs[0])]
    add2d(sg_obs, sg_cls, sg)
    isg = cuda.local.array(shape=max2d, dtype=float32)
    isg = isg[0:len(sg_obs),0:len(sg_obs[0])]
    if len(sg)==4:
        Inverse4x4(sg,isg)
    elif len(sg)==3:
        Inverse3x3(sg,isg)
    elif len(sg)==2:
        Inverse2x2(sg,isg)

    ###################### SINGLE MASS STARS ####################################################
    sub1 = cuda.local.array(shape=max2dEvals, dtype=float32)
    sub1 = sub1[0:len(evals),0:len(idx)]
    idxEvals = cuda.local.array(shape=max1dEvals, dtype=int16)
    idxEvals = idxEvals[0:len(evals)]
    idxFull(len(evals),idxEvals)
    getItems2d(evals,idxEvals,idx,sub1)
    subMatVec(sub1,phot,sub1)
    
    ctr_i = CalculateCtr_i(isg, sub1)
    ctr = cuda.local.array(shape=max1d, dtype=float32)
    ctr = ctr[0:len(evals[0])]
    copy1d(evals[ctr_i],ctr)
    
    minRange = max(0,ctr_i-25)
    maxRange = min(len(evals),ctr_i+25)
    bins = cuda.local.array(shape=max2d, dtype=float32)
    bins = bins[0:(maxRange-minRange),0:len(evals[0])]    
    copy2d(evals[minRange:maxRange],bins)
    
    ctr_idx = cuda.local.array(shape=max1d, dtype=float32)
    ctr_idx = ctr_idx[0:len(idx)]
    getItems1d(ctr,idx,ctr_idx)
    
    sub2 = cuda.local.array(shape=max2d, dtype=float32)
    sub2 = sub2[0:len(bins),0:len(idx)]
    idxBins = cuda.local.array(shape=max1dEvals, dtype=int16)
    idxBins = idxBins[0:len(bins)]
    idxFull(len(bins),idxBins)
    getItems2d(bins,idxBins,idx,sub2)
    subMatVec(sub2,ctr_idx,sub2)
    
    minBall,maxBall = CalculateBalls(isg, bins, dmha, sub2)
    dom = cuda.local.array(shape=(2), dtype=float32)
    #dom[0] = min(ctr[id_clr]-0.05,minBall)
    #dom[1] = max(ctr[id_clr]+0.05,maxBall)
    dom[0] = minBall
    dom[1] = maxBall + dclr

    ###################### BINARIES #############################################################
    nzc   = int(1.0/((rg_color[1]-rg_color[0])/len(evals)))
    # #---- true phot bewteen ctr_clr and ctr_clr + 0.5
    minRange = max(0,ctr_i-int(0.5*nzc))
    maxRange = min(len(evals),ctr_i+nzc)
    bbins = cuda.local.array(shape=max2d, dtype=float32)
    bbins = bbins[0:(maxRange-minRange),0:len(evals[0])]    
    copy2d(evals[minRange:maxRange],bbins)
    # bbins = evals.copy()
    # #---- true binaries phot
    idxBbins = cuda.local.array(shape=max1dEvals, dtype=int16)
    idxBbins = idxBbins[0:len(bbins)]
    idxFull(len(bbins),idxBbins)
    subItems2d(bbins,idxBbins,id_mag,dm)
    
    

    # #------ distance ---------
    sub3 = cuda.local.array(shape=max2d, dtype=float32)
    sub3 = sub3[0:len(bbins),0:len(idx)]
    getItems2d(bbins,idxBbins,idx,sub3)
    subMatVec(sub3,phot,sub3)
    ctrb_i = CalculateCtr_i(isg, sub3)
    
    ctrb = cuda.local.array(shape=max1d, dtype=float32)
    ctrb = ctrb[0:len(bbins[0])]
    copy1d(bbins[ctrb_i],ctrb)
    minRange = max(0,ctrb_i-25)
    maxRange = min(len(bbins),ctrb_i+25)
    
    bbinsR = cuda.local.array(shape=max2d, dtype=float32)
    bbinsR = bbinsR[0:(maxRange-minRange),0:len(bbins[0])] 
    copy2d(bbins[minRange:maxRange],bbinsR)
    
    ctrb_idx = cuda.local.array(shape=max1d, dtype=float32)
    ctrb_idx = ctrb_idx[0:len(idx)]
    getItems1d(ctrb,idx,ctrb_idx)
    
    sub4 = cuda.local.array(shape=max2d, dtype=float32)
    sub4 = sub4[0:len(bbinsR),0:len(idx)]
	
    idxBbinsR = cuda.local.array(shape=max1d, dtype=int8)
    idxBbinsR = idxBbinsR[0:len(bbinsR)]
    idxFull(len(bbinsR),idxBbinsR)
    getItems2d(bbinsR,idxBbinsR,idx,sub4)
	
    subMatVec(sub4,ctrb_idx,sub4)
	
    minBall,maxBall = CalculateBalls(isg, bbinsR, dmha, sub4)
    domb = cuda.local.array(shape=(2), dtype=float32)
    #domb[0] = min(ctrb[id_clr]-0.05,minBall)
    #domb[1] = max(ctrb[id_clr]+0.05,maxBall)
    domb[0] = minBall
    domb[1] = maxBall + dclr

    ############################################################################################
    #----- ensures that the domain is inside the limits of color  
    dom[0] = max( dom[0],rg_color[0])
    dom[1] = min( dom[1],rg_color[1])
    domb[0] = max( domb[0],rg_color[0])
    domb[1] = min( domb[1],rg_color[1])

    
    dtau  = (dom[1]-dom[0])/(stp-1)
    dtaub = (domb[1]-domb[0])/(stp-1)
    
    tau = cuda.local.array(shape=max1d, dtype=float32)
    tau = tau[0:int(stp)]
    linspace(dom[0],dom[1],stp,tau)
    taub = cuda.local.array(shape=max1d, dtype=float32)
    taub = taub[0:int(stp)]
    linspace(domb[0],domb[1],stp,taub)

    pclr = cuda.local.array(shape=max1d, dtype=float32)
    pclr = pclr[0:len(tau)]
    pclrb = cuda.local.array(shape=max1d, dtype=float32)
    pclrb = pclrb[0:len(taub)]
    p_color(tau,pi_color,mu_color,vr_clr,rg_color,pclr)
    p_color(taub,pi_color,mu_color,vr_clr,rg_color,pclrb)

    ps = cuda.local.array(shape=max1d, dtype=float64)
    ps = ps[0:len(tau)]
    bs = cuda.local.array(shape=max1d, dtype=float64)
    bs = bs[0:len(taub)]
    LikeMemberPHInc(phot,sg_obs,sg_cls,isg,alpha,tau,taub,coefs,rg_color,idx,degspl,id_mag,ps,bs)
    PhPs  = dotVec(pclr,ps)*dtau
    PhBs  = dotVec(pclrb,bs)*dtaub
    return PhPs,PhBs

@cuda.jit(device=True)
def logLikeIndependent(datum,pi,pi_pm_ps,pi_pm_bs,
    pi_color,mu_color,vr_clr,
    mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
    coefs,phot_cov,alpha,
    stp,rg_color,evals,degspl,id_mag):
    
    #------- Non-Members ----------
    lnm      = datum[0]
    #--------------- Datum -----------
    obs      = datum[1:(2+ph_dim+1)]
    uncert   = datum[(2+ph_dim+1):]
    pm       = obs[0:2]
    phot     = obs[2:]
    u_pm     = uncert[0:2]
    u_phot   = cuda.local.array(shape=(ph_dim, ph_dim), dtype=float32)
    zeros2d(u_phot)
    CreateU_phot(uncert[2:], u_phot)
    
    
    #---------------  Marginalization of color -----------------------
    PhPs,PhBs = MarginalColor(phot,u_phot,phot_cov,
                 alpha,coefs,pi_color,mu_color,vr_clr,rg_color,stp,evals,degspl,id_mag)

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
    lp    = math.log(((pi[0])*lnm)+((1-pi[0])*lmm))
    #------- Blobs ---------------
    PC1   = math.log(((1-pi[0])*lmm))-lp
    PC2   = math.log(pi[1]*ps_c) - math.log(lmm)

    # print(phot,lnm,lmm,lp,np.exp(PC1))
    return lp,PC1,PC2


@cuda.jit(device=True)
def logLikeIndependentInc(datum,pi,pi_pm_ps,pi_pm_bs,
    pi_color,mu_color,vr_clr,
    mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
    coefs,phot_cov,alpha,
    stp,rg_color,evals,degspl,id_mag):
    #------- Non-Members ----------
    
    lnm      = datum[0]
    #--------------- Datum -----------
    obs      = datum[1:(3+ph_dim)]
    uncert   = datum[(3+ph_dim):-(ph_dim+2)]
    index    = datum[-(ph_dim+2):]

    pm       = obs[0:2]
    phot     = obs[2:]
    u_pm     = uncert[0:2]
    u_phot   = cuda.local.array(shape=(ph_dim, ph_dim), dtype=float32)
    
    zeros2d(u_phot)
    CreateU_phot(uncert[2:], u_phot)
    
    #--------------- Missing ---------
    indexPh_dim = cuda.local.array(shape=(ph_dim), dtype=float32)
    indexPh_dim = index[2:2+ph_dim]
    idx     = cuda.local.array(shape=(ph_dim), dtype=int8)
    count = 0
    for i in range(ph_dim):
        if (indexPh_dim[i] == 0.0):
            idx[count] = i
            count = count + 1 
    idx = idx[0:count]
      
       
    #--------------------------------
    #---------------- Marginalization of color --------------
    PhPs,PhBs= MarginalColorInc(phot,u_phot,phot_cov,
                 alpha,coefs,pi_color,mu_color,vr_clr,rg_color,stp,idx,evals,degspl,id_mag)
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
    lp    = math.log(((pi[0])*lnm)+((1-pi[0])*lmm))
    #------- Blobs ---------------
    PC1   = math.log(((1-pi[0])*lmm))-lp
    PC2   = math.log(pi[1]*ps_c) - math.log(lmm)
    
    # print(lnm,lmm,lp,np.exp(PC1))
    return lp,PC1,PC2

#Return lpsc and lpsi
@cuda.jit
def KernelLikelihood(obs_c,obs_i,pi,pi_pm_ps,pi_pm_bs,
    pi_color,mu_color,vr_clr,
    mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,
    coefs,phot_cov,alpha,
    stp,rg_color,evals,degspl,id_mag,lpsc,lpsi):
        
    start = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    lenObs_C = len(obs_c)    
    
    #Distribution of observations on GPU threads.
    for i in range(start, lenObs_C+len(obs_i), gridX):
        if i < lenObs_C:
            lp,PC1,PC2 = logLikeIndependent(obs_c[i],pi,pi_pm_ps,pi_pm_bs,pi_color,mu_color,vr_clr,
                           mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,coefs,phot_cov,alpha,
                           stp,rg_color,evals,degspl,id_mag)           
            lpsc[i,0] = lp
            lpsc[i,1] = PC1
            lpsc[i,2] = PC2
            
        else:
            lp,PC1,PC2 = logLikeIndependentInc(obs_i[i-lenObs_C],pi,pi_pm_ps,pi_pm_bs,pi_color,mu_color,vr_clr,
                           mu_pm_ps,sg_pm_ps,mu_pm_bs,sg_pm_bs,coefs,phot_cov,alpha,
                           stp,rg_color,evals,degspl,id_mag)
                           
            lpsi[i-lenObs_C,0] = lp
            lpsi[i-lenObs_C,1] = PC1
            lpsi[i-lenObs_C,2] = PC2
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
    