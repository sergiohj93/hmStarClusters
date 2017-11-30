import sys
import numpy as np
from numpy.linalg import inv as inv
from scipy.stats import multivariate_normal as dmvnorm
from scipy.stats import invwishart
from scipy.misc  import logsumexp as logsumexp
from sklearn import mixture, cluster, covariance



"""
This code intends to reproduce the Gaussian Mixture Model 'Uncertainty training' algorithm of Ozerov 2012.
Project report 7862- January 2012- inria.

It also includes one uniform distribution. 
"""

# If mpi4py is installed, import it.
# try:
#     from mpi4py import MPI
#     MPI = MPI
# except ImportError:
#     MPI = None

# comm = MPI.COMM_WORLD
# rank,size = comm.Get_rank(),comm.Get_size()

class GMM_U2(object):
    """
    This class fits a Gaussian Mixture Model together with a uniform density to the data
    """
    def __init__(self,data,uncert,nComp=2,tol=1e-4,maxIter=1000,n_tri=10):
        #rho     Value of increasing step in gradient descent rho<2

        nComp = np.array(nComp)
        data = np.array(data)
        u_sg = np.array(uncert)

        if (np.any(nComp<=1)):
            print("Cluster number must be grater than one.\n Remember it fits also a uniform distribution.")
        N_g  = len(nComp)
        hbic = np.infty
        self.bics   = []
        N,D         = data.shape
        Nsg,Dsg,_   = uncert.shape
        if Nsg != N or D != Dsg:
            sys.exit("Uncertainty with different shape!")
        self.mima   = np.zeros((D,2))
        dif         = np.zeros(D)
        for i in range(D):
            self.mima[i]   = [np.min(data[:,i]),np.max(data[:,i])]
            dif[i]         = np.max(data[:,i])-np.min(data[:,i])
        intbd    = np.prod(dif)
        lunden   = np.log(1.0/intbd)
        self.cte = lunden
        #------------------

        #--------- Loop over different Models -----
        for G in nComp :
            # ----- initial set of parameters and Gaussians--------
            veritas = True
            cTri    = 0
            while veritas and cTri <= n_tri :
                #-------- GMM ---------
                gmm    = mixture.GMM(n_components=G, covariance_type='full',tol=1e1*tol,
                        random_state=0)
                gmm.fit(data)
                pro    = gmm.weights_
                mu     = gmm.means_
                sigma  = gmm.covars_
                if not gmm.converged_:
                        sys.exit("mixture.GMM did not converged!")

                det  = np.zeros(G)

                for i in range(G):
                    det[i] = np.linalg.det(sigma[i])

                idmx  = np.argsort(det)[::-1]

                pro   = pro[idmx]
                mu    = mu[idmx][1:]
                sigma = sigma[idmx][1:]

                z     = np.empty((N,G))
                lden  = np.zeros(G)

            
                for r in range(N):
                    lden[0]     = np.log(pro[0])+lunden
                    for i in range(G-1):
                        sg      = sigma[i]+u_sg[r]
                        div     = 0.5*np.log(((2*np.pi)**(D))*np.linalg.det(sg))
                        mvn     = -0.5*np.dot((data[r]-mu[i]),np.dot(inv(sg),(data[r]-mu[i]).T))
                        lden[i+1] = np.log(pro[i+1])+mvn-div
                    z[r,]    = np.exp(lden)/np.sum(np.exp(lden))

                veritas = np.isnan(z).any()

                #--- count trials ---
                cTri +=1

            if veritas :
                raise RuntimeError("After n_tri trials Z remained NaN")

           #----- DO fit ---------- 

            fit    = OneModel(data,u_sg,G,z,pro,mu,sigma,lunden,tol,maxIter)

            #------ BIC comparison ------
            print "BIC of model with ",G," gaussians: ",fit.bic
            self.bics.append(fit.bic)
            if fit.bic < hbic:
                hbic      = fit.bic
                self.best = fit

class OneModel(object):
        """
        Fit one model
        data
        Number of Gaussians
        membership probabilities
        fractions
        means
        covariances
        tolerance in absolute LogLikelihood
        iterations
        index of observed entries
        """

        def __init__(self,data,u_sg,G,z,pro,mu,sigma,lunden,tol,maxIter):
            N,D    = data.shape
            ollik  = -1
            rates  = np.array([1.0,1.0]) 
            rate   = 1.0
            count  = 1
            cond   = True

            while cond:
                #-------------------- E step ----------------
                lden  = np.zeros(G)
                lliks = np.zeros(N)

                for r in range(N):
                    lden[0]     = np.log(pro[0])+lunden
                    for i in range(G-1):
                        sg      = sigma[i] + u_sg[r]
                        div     = 0.5*np.log(((2*np.pi)**(D))*np.linalg.det(sg))
                        mvn     = -0.5*np.dot((data[r]-mu[i]),np.dot(inv(sg),(data[r]-mu[i]).T))
                        lden[i+1] = np.log(pro[i+1])+mvn-div
                    z[r,]    = np.exp(lden)/np.sum(np.exp(lden))
                    lliks[r] = lden[np.argmax(z[r,])]

                if np.isnan(z).any():
                    raise RuntimeError(
                        "Probability returned NaN")

                #---- likelihood ---
                llik = np.sum(lliks)
                print llik
                # ------------------ M step ------------------
                # compute parameters
                # ----- fractions
                pro = np.sum(z,axis=0)/N
                #--------------------
                for i in range(G-1):
                    
                #------- means and covariances ----------
                    # sgs = np.zeros((N,D,D))
                    # mus = np.zeros((N,D))

                    # for r in range(N):
                    #     mus[r]  = z[r,i+1]*data[r]

                    # mu[i]    = np.sum(mus,axis=0)/np.sum(z[:,i+1])

                    # for r in range(N):
                    #     sgs[r]  = z[r,i+1]*np.outer(data[r]-mu[i],data[r]-mu[i])

                    # sigma[i] = np.sum(sgs,axis=0)/np.sum(z[:,i+1])

                #------- means and covariances ----------
                    esgs = np.zeros((N,D,D))
                    emus = np.zeros((N,D))

                    for r in range(N):
                        sg      = sigma[i] + u_sg[r]
                    #---------- Wiener filter -----------------
                        W       = np.dot(sigma[i],inv(sg))
                    #---------- expected mean ---------------
                        # emus[r] = data[r]
                        emus[r] = np.dot(W,(data[r]-mu[i]))+mu[i]
                    #---------- expected sigma ------------- 
                        # esgs[r] = np.outer(data[r],data[r])
                        esgs[r] = np.outer(emus[r],emus[r])+ np.dot((np.eye(D)-W),sigma[i])
                    #---------- 
                        emus[r] = z[r,i+1]*emus[r]
                        esgs[r] = z[r,i+1]*esgs[r]

                    mu[i]    = np.sum(emus,axis=0)/np.sum(z[:,i+1])
                    sigma[i] = (np.sum(esgs,axis=0)/np.sum(z[:,i+1]))-np.outer(mu[i],mu[i])
                    
                    
                #---------- Comparison------------
                rate = np.abs(llik-ollik)/np.abs(ollik)
                ollik= llik
                rates[1] = rates[0]
                rates[0] = rate
                count += 1
                cond = ((np.any(rates > tol) and (count < maxIter)))
            if count == maxIter: print "Max iteration reached"
            print "Done in ",count," iterations."
            #---- BIC ------
            kmod = (G-1)+((G-1)*D)+((G-1)*(D*(D+1)*0.5)) # parameters of fractions,means and covariance matrices
            bic  = -2*llik+(kmod)*np.log(N)

            self.frac  = pro
            self.mean  = mu
            self.cov   = sigma
            self.bic   = bic

# class GMM_MISSING_MPI(object):
#     """
#     This class fits a Gaussian Mixture Model of full covariance to the data
#     """
        

#     def __init__(self,data,nComp=2,tol=1e-4,maxIter=1000,rho=0.5,tolCov=1e-2,n_tri=1,
#                     ipro=0,imu=0,isigma=0,init="GMM"):
#         nComp = np.array(nComp)
#         data = np.array(data)
#         if (np.any(nComp<=1)):
#             print("Cluster number must be grater than one.\n Save time! use other function.")
#         N_g  = len(nComp)
#         hbic = -np.infty
#         self.bics=[]
#         N,D  = data.shape        
#         rank,size = comm.Get_rank(),comm.Get_size()

#         #------- finds missing data----
#         isobs    = np.isfinite(data)
#         ind_comp = np.where(np.sum(isobs,axis=1) == D)[0]
#         N_c      = len(ind_comp)
#         N_m      = N - N_c
#         pC       = (100.*N_c)/(1.0*N)
#         if rank==0:
#             print "Percentage of complete data:",pC
#         if (N_c == 0):
#             print("Not a single complete datum.")
#         #------------------

#         my_N   = N/size

#         if N%size != 0 :
#             sys.exit("Data not divisible by CPU's.")

#         # ----- Scatters data -------

#         my_indx  = range(my_N*rank,my_N*rank+my_N)
#         my_data  = data[my_indx,]

#         # computes own index
#         isobs    = np.isfinite(my_data)

#         #--------- Loop over different Models -----
#         for G in nComp :
#             # ----- initial set of parameters and Gaussians--------
#             z       = np.empty((N,G))
#             my_z    = np.empty((my_N,G))
#             pro     = np.empty(G)
#             mu      = np.empty((G,D))
#             sigma   = np.empty((G,D,D))
#             veritas = True
#             noPSD   = False
            
#             cTri    = 1
#             while veritas and cTri <= n_tri :
#                 if rank==0:
#                     if init=="GMM":
#                         #-------- GMM ---------
#                         gmm   = mixture.GMM(n_components=G, 
#                             covariance_type='full',tol=tol,
#                             random_state=int(time()))
#                         gmm.fit(data[ind_comp,])
#                         pro   = gmm.weights_
#                         mu    = gmm.means_
#                         sigma = gmm.covars_
#                         if not gmm.converged_:
#                             sys.exit("mixture.GMM did not converged!")
#                     # ----- one last chance -----
#                     # ------ this may take longer to converge -----
#                     # if cTri== n_tri:
#                     #     sigma = np.repeat(np.cov(data[ind_comp,],rowvar=0)[None,:,:],G,axis=0)
#                     if init=="K-means" : 
#                         #---k-means -----------
#                         km    = cluster.KMeans(n_clusters=G,max_iter=1000,n_init=1,tol=tol,random_state=int(time()))
#                         km.fit(data[ind_comp,])
#                         mu    = km.cluster_centers_
     
#                         for i in range(G):
#                             idx    = np.where(km.labels_==i)[0]
#                             pro[i] = len(idx)/float(len(ind_comp))
#                             if pro[i] <=0:
#                                 sys.exit("Zero in fraction")
#                             sigma[i] = covariance.empirical_covariance(data[ind_comp[idx],])
#                             eigs = scipy.linalg.eigvals(sigma[i])
#                             if np.any(eigs<0):
#                                 print("Setting covariance to that of all complete data.")
#                                 sigma[i] = covariance.empirical_covariance(data[ind_comp,])
#                                 eigs = scipy.linalg.eigvals(sigma[i])
#                                 if np.any(eigs<0):
#                                     noPSD = True
#                                     print ("NO pSD inital matrix.")
#                                     sys.exit("NO pSD inital matrix.")

#                     if init == "own":
#                         #------ use initial values ----
#                         pro = ipro
#                         mu = imu
#                         sigma = isigma

#                 pro     = comm.bcast(pro,root=0)
#                 mu      = comm.bcast(mu,root=0)
#                 sigma   = comm.bcast(sigma,root=0)

                
#                 for r in range(my_N):
#                     den  = np.empty(G)
#                     w = isobs[r,]
#                     for i in range(G):
#                         sg      = sigma[i][np.ix_(w,w)]
#                         x       = my_data[r,w]-mu[i,w]
#                         div     = 0.5*(sum(isobs[r,])*np.log(2*np.pi)+np.linalg.slogdet(sg)[1])
#                         mvn     = -0.5*np.dot(x,np.dot(inv(sg),x.T))
#                         den[i]  = pro[i]*np.exp(mvn-div)
#                         if np.isnan(den[i]) or np.isneginf(den[i]) or den[i]==0 :
#                             den[i] =1e-200
#                         if np.isinf(den[i]):
#                             den[i] =1e200
#                     my_z[r,]    = den/np.sum(den)


#                 comm.barrier()
#                 # Allgather data into original array

#                 comm.Allgather( [my_z, MPI.DOUBLE], [z, MPI.DOUBLE] )


#                 veritas = np.isnan(z).any()
#                 if veritas and rank==0:
#                     print "Z returned nan, trying again ..."

#                 #--- count trials ---
#                 cTri +=1

#             if veritas and rank==0:
#                     raise RuntimeError("After n_tri trials Z remained NaN")


            

#            #----- DO fit ---------- 
#             comm.barrier()
#             fit    = OneModelMPI(my_data,G,my_z,pro,mu,sigma,isobs,tol,maxIter,rho,tolCov)

#             #------ BIC comparison ------
#             if rank == 0:
#                 print "BIC of model with ",G," gaussians: ",fit.bic
#             self.bics.append(fit.bic)
#             if fit.bic > hbic:
#                 hbic      = fit.bic
#                 self.best = fit
#             comm.barrier()

# class OneModelMPI(object):
#         """
#         Fit one model
#         data
#         Number of Gaussians
#         membership probabilities
#         fractions
#         means
#         covariances
#         tolerance in absolute LogLikelihood
#         iterations
#         index of observed entries
#         """

#         def __init__(self,my_data,G,my_z,pro,mu,sigma,isobs,tol,maxIter,rho,tolCov):
#             my_N,D = my_data.shape
#             ollik  = -1
#             rates  = np.array([1.0,1.0,1.0,1.0,1.0]) 
#             rate   = 1.0
#             count  = 1
#             rank   = comm.Get_rank()
#             size   = comm.Get_size()
#             N      = my_N*size
#             z      = np.empty((N,G))
#             cond   = True
#             opro   = pro
#             omu    = mu
#             osigma = sigma


#             while cond:
#                 llik  = 0.0
#                 den  = np.empty(G)
#                 lliks = np.zeros(N)
#                 my_lliks = np.zeros(my_N)
#                 #-------------------- E step ----------------
#                 for r in range(my_N):
#                     w = isobs[r,]
#                     for i in range(G):
#                         sg      = sigma[i][np.ix_(w,w)]
#                         x       = my_data[r,w]-mu[i,w]
#                         div     = 0.5*(sum(isobs[r,])*np.log(2*np.pi)+np.linalg.slogdet(sg)[1])
#                         mvn     = -0.5*np.dot(x,np.dot(inv(sg),x.T))
#                         den[i]  = pro[i]*np.exp(mvn-div)
#                         if np.isnan(den[i]) or np.isneginf(den[i]) or den[i]==0 :
#                             den[i] =1e-200
#                         if np.isinf(den[i]):
#                             den[i] =1e200
#                     my_z[r,]    = den/np.sum(den)
#                     my_lliks[r] = np.log(den[np.argmax(my_z[r,])])
#                 comm.barrier()
#                 # Allgather data into original array
#                 comm.Allgather( [my_z, MPI.DOUBLE], [z, MPI.DOUBLE] )
#                 comm.Allgather( [my_lliks, MPI.DOUBLE], [lliks, MPI.DOUBLE] )

                
#                 if np.isnan(z).any():
#                     raise RuntimeError(
#                         "Probability returned NaN")
#                     sys.exit()

#                 llik = np.sum(lliks)
#                 if rank ==0:
#                     print llik
#                 # ------------------ M step ------------------
#                 # compute parameters
#                 # ----- fractions
#                 pro  = np.sum(z,axis=0)/N
#                 assert np.all(pro >0) and np.all(pro <1)
#                 #------ means ---------
#                 for i in range(G):
#                     mu_a   = np.zeros((N,D,D))
#                     mu_b   = np.zeros((N,D))
#                     mmu_a  = np.zeros((my_N,D,D))
#                     mmu_b  = np.zeros((my_N,D))
#                     for r in range(my_N):
#                         w  = isobs[r,]
#                         sg = sigma[i][np.ix_(w,w)]
#                         M  = np.eye(D)[w,] 
#                         H  = np.dot(M.T,inv(sg))
#                         mmu_a[r,:,:] = my_z[r,i]*np.dot(H,M)
#                         mmu_b[r,:]   = my_z[r,i]*np.dot(H,my_data[r,w])
#                     comm.barrier()
#                     # Allgather data into A again
#                     comm.Allgather( [mmu_a, MPI.DOUBLE], [mu_a, MPI.DOUBLE] )
#                     comm.Allgather( [mmu_b, MPI.DOUBLE], [mu_b, MPI.DOUBLE] )

#                     if rank==0:
#                         mu[i] = np.dot(inv(np.sum(mu_a,axis=0)),np.sum(mu_b,axis=0))
#                     mu[i] = comm.bcast(mu[i])

                    
#                     #-------  covariances ----------
#                     dif = 1
#                     while dif > tolCov :
#                         delta    = np.zeros((N,D,D))
#                         my_delta = np.empty((my_N,D,D))
#                         for r in range(my_N):
#                             w    = isobs[r,]
#                             sg   = sigma[i][np.ix_(w,w)]
#                             M    = np.eye(D)[w,]
#                             H    = np.dot(M.T,inv(sg))
#                             out  = np.outer((my_data[r,w]-mu[i,w]),(my_data[r,w]-mu[i,w]))
#                             my_delta[r] = my_z[r,i]*(np.dot(H,np.dot(out,H.T))-np.dot(H,np.eye(D)[w,:]))
#                         comm.barrier()
#                         comm.Allgather( [my_delta, MPI.DOUBLE], [delta, MPI.DOUBLE] )
#                         if rank==0:
#                             sdel      = np.sum(delta,axis=0)/(N*pro[i])
#                             sdel      = 0.5*(sdel+sdel.T)
#                             Delta     = 0.5*(np.dot(sigma[i],np.dot(sdel,sigma[i])))
#                             nosg      = np.linalg.norm(sigma[i])
#                             osigma    = sigma[i].copy()
#                             try:
#                                 np.linalg.cholesky(sigma[i])
#                                 sigma[i]  = osigma + rho*Delta
#                             except np.linalg.LinAlgError:
#                                 sigma[i]  = osigma
#                                 # print "NO pSD"

#                             nnsg      = np.linalg.norm(sigma[i])
#                             dif       = np.abs(1.0-(nnsg/nosg))
#                         sigma[i] = comm.bcast(sigma[i])
#                         dif      = comm.bcast(dif)
#                         comm.barrier()
#                 #---------- Comparison------------
#                 rate     = (llik-ollik)/np.abs(ollik)
#                 ollik    = llik
#                 rates[4] = rates[3]
#                 rates[3] = rates[2]
#                 rates[2] = rates[1]
#                 rates[1] = rates[0]
#                 rates[0] = rate
#                 count += 1
#                 cond   = not ((np.all(np.abs(rates) < tol)) or (count > maxIter))#and np.all(rates > 0)

#             if rank==0:
#                 print "Done in ",count," iterations."
#                 if count == maxIter: print "Max iteration reached!"
            
#             #---- BIC ------
#             kmod = (G-1)+(G*D)+(G*(D*(D+1)*0.5)) # parameters of fractions,means and covariance matrices
#             bic  = 2*llik-(kmod)*np.log(N)

#             self.frac  = pro
#             self.mean  = mu
#             self.cov   = sigma
#             self.bic   = bic


# ###############
# #-------------- DATA ----------------------
# D    = 2
# # # #--------- hyper parameters ----
# n1  = 1000
# n2  = 1000
# n3  = 1000
# n4  = 1000
# n5  = 1000
# mu1 = np.array([10,10])
# mu2 = np.array([10,-10])
# mu3 = np.array([-10,10])
# mu4 = np.array([-10,-10])

# sg   = 3.0*np.eye(D)
# u_sg = 0.05*np.eye(D)

# # #-------------------------------
# N      = n1+n2+n3+n4+n5
# tpro   = np.array([n1,n4,n3,n5,n2])/(1.0*N)#
# tmu    = np.stack((mu3,mu2,mu4,mu1),axis=0)
# tsigma = np.stack((sg,sg,sg,sg),axis=0)#

# # Generate random sample, two components
# np.random.seed(0)
# X1 = 25*np.array([np.random.uniform(low=-1,high=1,size=n1),np.random.uniform(low=-1,high=1,size=n1)]).T
# X2 = np.random.multivariate_normal(mu1, sg+u_sg,n2)
# X3 = np.random.multivariate_normal(mu2, sg+u_sg,n3)
# X4 = np.random.multivariate_normal(mu3, sg+u_sg,n4)
# X5 = np.random.multivariate_normal(mu4, sg+u_sg,n5)

# X   = np.vstack((X1,X2,X3,X4,X5))
# u_X = invwishart.rvs(2,u_sg,N)

# #-------------------
# fit = GMM_U2(X,u_X,[5],tol=1e-5)
# # print fit.best.frac,tpro
# # print fit.best.mean,tmu
# # print fit.best.cov
# print fit.mima
# print np.sum(fit.best.frac-tpro)
# print np.sum(fit.best.mean-tmu)
# print np.sum(fit.best.cov-tsigma)
# # print fit.best.cov

