#!/usr/bin/env python
import sys
import numpy as np

from time import time

import h5py
from scipy import linalg

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl

import itertools

from gmm_missing import GMM_MISSING_MPI as gmmm
# from gmm_uniform import GMM_UNIFORM as gmmu
from gmm_uniform_uncert import GMM_U2 as gmmu2


try:
    from mpi4py import MPI
    MPI = MPI
except ImportError:
    MPI = None

comm = MPI.COMM_WORLD
rank,size = comm.Get_rank(),comm.Get_size()

#-------------- DATA ----------------------
doPM   = True
doPhot = True
tol    = 1e-4

range_pm   = [10]
range_phot = [15]
# Number of samples per component
dir_      = '/pcdisk/boneym5/jolivares/Data/Boneym_10G_0.75_1e+04/Field/'
# path_data = dir_+'Candidates-1e5.RData'
path_data = dir_+'Selection.h5'
#----- where to save parameters ------
path_pm   = dir_+'FieldParameters_PM.h5'
path_phot = dir_+'FieldParameters_Phot.h5'

with h5py.File(path_data,'r') as hf:
    nm   = np.array(hf.get("nonmem"))
    u_nm = np.array(hf.get("u_nonmem")) 
    A    = np.array(hf.get("A"))

id_pm= [0,1]
id_ph= [2,3,4,5,6] 

#------- put R nan into python NaN
nm[np.where(np.logical_not(np.isfinite(nm)))] = np.nan
u_nm[np.where(np.logical_not(np.isfinite(u_nm)))] = 1000
# #-----sample ----
X         = nm
# #------- finds missing data----
# ns     = 9990
# ismiss   = np.logical_not(np.isfinite(X))
# isobs    = np.isfinite(X)
# indx     = np.sum(ismiss*1,axis=1)
# ind_comp = np.where(indx == 0)[0]
# ns       = ns - len(ind_comp)
# X        = np.vstack([X[ind_comp,],X[np.random.randint(0,X.shape[0],ns),]])
#---------------------------------
X         = comm.bcast(X,root=0)
ns,D = X.shape
X_pm = X[:,id_pm]
X_ph = X[:,id_ph]
A_pm = A[np.ix_(id_pm,id_pm)]
A_ph = A[np.ix_(id_ph,id_ph)]
#-------------- covaraince matrices ------------
u_pm = np.zeros((len(X),len(id_pm),len(id_pm)))
u_ph = np.zeros((len(X),len(id_ph),len(id_ph)))
for i in range(len(X)):
    u_pm[i] = np.dot(A_pm,np.dot(np.diag(u_nm[i,id_pm]),A_pm.T))
    u_ph[i] = np.dot(A_ph,np.dot(np.diag(u_nm[i,id_ph]),A_ph.T))

#----------------- FIT PM-------------------------------
if rank ==0 and doPM:
    print "Computing kinematic parameters of field"
    lowest_bic = np.infty
    bic = []
    fit_pm = gmmu2(X_pm,u_pm,range_pm,tol=tol)
    best   = fit_pm.best
    bic    = fit_pm.bics


    M_pm      = len(best.frac)
    theta_pm  = best.frac
    mu_pm     = best.mean.T
    sg_pm     = best.cov.T
    mima      = fit_pm.mima
    cte       = fit_pm.cte

    with h5py.File(path_pm, 'w') as hf:
        hf.create_dataset('M_pm',        data=M_pm)
        hf.create_dataset('theta_nm_pm', data=theta_pm)
        hf.create_dataset('mu_nm_pm',    data=mu_pm)
        hf.create_dataset('sigma_nm_pm', data=sg_pm)
        hf.create_dataset('mima_nm_pm',  data=mima)
        hf.create_dataset('cte_nm_pm',   data=cte)

    
    #----------------------- PLOT ------------------------------------
    bic = np.array(bic)
    bars =[]

    print "Making Plots"

    with PdfPages(dir_+'GMM-BIC-PM-'+'{:1.0g}'.format(ns)+'-'+str(range_pm)+'.pdf') as pdf:
        fig = plt.subplot(1,1,1)
        xpos = np.array(range_pm)
        bars.append(plt.bar(xpos, bic,width=.2, color='blue'))
        plt.xticks(range_pm)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        plt.xlabel('Number of components')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    #-------- Plot parameters and data ------
        #------- Vector Point Diagram
        fig = plt.subplot(1,1,1)
        plt.scatter(X_pm[:,0],X_pm[:,1],0.5,color="black")
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)

        for i in range(M_pm-1):
            a,b = np.sqrt(sg_pm[0,0,i]),np.sqrt(sg_pm[1,1,i])
            # Plot an ellipse to show the Gaussian component
            angle = (np.pi/4)*sg_pm[0,1,i]/(a*b)
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mu_pm[:,i],a,b, 180 + angle,
                color="red",
                fill=False,linestyle="solid")
            fig.add_artist(ell)
        plt.title('Vector Point Diagram')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

if doPhot:
    t1 = time()
    fit_nm = gmmm(X_ph,range_phot,n_tri=20,tolCov=1e-4,tol=tol,rho=0.1)
    best   = fit_nm.best
    bic    = fit_nm.bics


    if rank ==0:
        M_ph      = len(best.frac)
        theta_ph  = best.frac
        mu_ph     = best.mean.T
        sg_ph     = best.cov.T
        print "Time needed was: ", time()-t1

        with h5py.File(path_phot, 'w') as hf:
            hf.create_dataset('M_phot',        data=M_ph)
            hf.create_dataset('theta_nm_phot', data=theta_ph)
            hf.create_dataset('mu_nm_phot',    data=mu_ph)
            hf.create_dataset('sigma_nm_phot', data=sg_ph)
        
        # ----------------------- PLOT ------------------------------------
        bic = np.array(bic)
        clf = best
        bars = []

        with PdfPages(dir_+'GMM-BIC-Missing-'+'{:1.0g}'.format(ns)+'-'+str(range_phot)+'.pdf') as pdf:
            plt.figure(figsize=(6, 6))
            xpos = np.array(range_phot)
            bars.append(plt.bar(xpos, bic,width=.2, color='blue'))
            plt.xticks(range_phot)
            plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
            plt.title('BIC score per model')
            plt.xlabel('Number of components')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            # ----- Color Magnitud Diagram ------

            fig = plt.subplot(1,1,1)
            plt.scatter(X_ph[:,0],X_ph[:,4],0.5,color="black")
            # plt.xlim(np.min(X_ph[:,4]),np.max(X_ph[:,4]))
            # plt.ylim(np.max(X_ph[:,1]),np.max(X_ph[:,1]))

            for i in range(M_ph):
                v, w = linalg.eigh(sg_ph[:,:,i][np.ix_([0,4],[0,4])])
                u = w[0] / linalg.norm(w[0])
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mu_ph[[0,4],i], v[0], v[1],angle,
                    color="green",fill=False,linestyle="solid")
                fig.add_artist(ell)

            plt.title('Color Magnitude Diagram')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


        
