import sys
import numpy as np
from time import time

from numpy.linalg import inv as inv
from scipy.stats import multivariate_normal as dmvnorm
from scipy.misc  import logsumexp as logsumexp
from sklearn import mixture, cluster, covariance
import scipy

# If mpi4py is installed, import it.
try:
    from mpi4py import MPI
    MPI = MPI
except ImportError:
    MPI = None

comm = MPI.COMM_WORLD
rank,size = comm.Get_rank(),comm.Get_size()

class GMM_MISSING(object):
    """
    This class fits a Gaussian Mixture Model of full covariance to the data
    """
    def __init__(self,data,nComp=1,tol=1e-4,maxIter=1000,rho=1.0,tolCov=1e-2,n_tri=10):
        #rho     Value of increasing step in gradient descent rho<2
        nComp = np.array(nComp)
        data = np.array(data)
        if (np.any(nComp<=1)):
            print("Cluster number must be grater than one.\n Save time! use other function.")
        N_g  = len(nComp)
        hbic = -np.infty
        self.bics=[]
        N,D  = data.shape 

        # #------- finds missing data----
        ismiss   = np.logical_not(np.isfinite(X))
        isobs    = np.isfinite(X)
        indx     = np.sum(ismiss*1,axis=1)
        ind_comp = np.where(indx == 0)[0]       

        # #------- finds missing data----
        # ismiss   = np.isnan(data)
        # isobs    = np.logical_not(ismiss)
        # indx     = np.sum(ismiss*1,axis=1)
        # ind_comp = np.where(indx == 0)[0]
        N_c      = len(ind_comp)
        N_m      = N - N_c
        if (N_c == 0):
            print("Not a single complete datum.")
            if (any(indx==(D-1))):
               print("This version does not deal with 1D missing data.")
            if (any(indx==D)):
               sys.exit("One datum was entiry missing")
        #------------------

        #--------- Loop over different Models -----
        for G in nComp :
            # ----- initial set of parameters and Gaussians--------
            veritas = True
            cTri    = 0
            while veritas and cTri <= n_tri :
                #-------- GMM ---------
                gmm   = mixture.GMM(n_components=G, covariance_type='full',
                        random_state=0)
                gmm.fit(data[ind_comp,])
                pro   = gmm.weights_
                mu    = gmm.means_
                sigma = gmm.covars_
                if not gmm.converged_:
                        sys.exit("mixture.GMM did not converged!")

                z     = np.empty((N,G))
                lden  = np.zeros(G)

                for r in range(N):
                    w = isobs[r,]
                    for i in range(G):
                        sg      = sigma[i][np.ix_(w,w)]
                        x       = data[r,w]-mu[i,w]
                        div     = 0.5*np.log(((2*np.pi)**(sum(isobs[r,])))*np.linalg.det(sg))
                        mvn     = -0.5*np.dot(x,np.dot(inv(sg),x.T))
                        lden[i] = np.log(pro[i])+mvn-div
                    z[r,]    = np.exp(lden)/np.sum(np.exp(lden))

                veritas = np.isnan(z).any()

                #--- count trials ---
                cTri +=1

            if veritas :
                raise RuntimeError("After n_tri trials Z remained NaN")

           #----- DO fit ---------- 

            fit    = OneModel(data,G,z,pro,mu,sigma,isobs,tol,maxIter,rho,tolCov)

            #------ BIC comparison ------
            print "BIC of model with ",G," gaussians: ",fit.bic
            self.bics.append(fit.bic)
            if fit.bic > hbic:
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

        def __init__(self,data,G,z,pro,mu,sigma,isobs,tol,maxIter,rho,tolCov):
            N,D    = data.shape
            ollik  = -1
            rates  = np.array([1.0,1.0]) 
            rate   = 1.0
            count  = 1

            while ((np.any(rates > tol) and (count < maxIter))):
                #-------------------- E step ----------------
                lden  = np.zeros(G)
                lliks = np.zeros(N)
                for r in range(N):
                    w = isobs[r,]
                    for i in range(G):
                        sg      = sigma[i][np.ix_(w,w)]
                        x       = data[r,w]-mu[i,w]
                        div     = 0.5*np.log(((2*np.pi)**(sum(isobs[r,])))*np.linalg.det(sg))
                        mvn     = -0.5*np.dot(x,np.dot(inv(sg),x.T))
                        lden[i] = np.log(pro[i])+mvn-div
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
                #------ means ---------
                for i in range(G):
                    mu_a = np.zeros((N,D,D))
                    mu_b = np.zeros((N,D))
                    for r in range(N):
                        w = isobs[r,]
                        M = np.eye(D)[w,] 
                        H = np.dot(M.T,inv(sigma[i][np.ix_(w,w)]))
                        mu_a[r,:,:] = z[r,i]*np.dot(H,M)
                        mu_b[r,:]   = z[r,i]*np.dot(H,data[r,w])
                    mu[i] = np.dot(inv(np.sum(mu_a,axis=0)),np.sum(mu_b,axis=0))

                    #-------  covariances ----------
                    dif = 1
                    while dif > tolCov :
                        delta = np.zeros((N,D,D))
                        for r in range(N):
                            w    = isobs[r,]
                            M    = np.eye(D)[w,]
                            mu_i = mu[i,w]
                            y_i  = data[r,w]
                            H    = np.dot(M.T,inv(sigma[i][np.ix_(w,w)]))
                            out  = np.outer((y_i-mu_i),(y_i-mu_i))
                            delta[r] = z[r,i]*(np.dot(H,np.dot(out,H.T))-np.dot(H,np.eye(D)[w,:]))
                        sdel      = np.sum(delta,axis=0)/(N*pro[i])
                        sdel      = 0.5*(sdel+sdel.T)
                        Delta     = 0.5*(np.dot(sigma[i],np.dot(sdel,sigma[i])))
                        nosg      = np.linalg.norm(sigma[i])
                        osigma    = sigma[i].copy()
                        try:
                            np.linalg.cholesky(sigma[i])
                            sigma[i]  = osigma + rho*Delta
                        except LinAlgError:
                            sigma[i]  = osigma
                            print "NO pSD"
                            
                        nnsg      = np.linalg.norm(sigma[i])
                        dif       = np.abs(1.0-(nnsg/nosg))
                #---------- Comparison------------
                rate = np.abs(llik-ollik)/np.abs(ollik)
                ollik= llik
                rates[1] = rates[0]
                rates[0] = rate
                count += 1
            if count == maxIter: print "Max iteration reached"
            print "Done in ",count," iterations."
            #---- BIC ------
            kmod = (G-1)+(G*D)+(G*(D*(D+1)*0.5)) # parameters of fractions,means and covariance matrices
            bic  = 2*llik-(kmod)*np.log(N)

            self.frac  = pro
            self.mean  = mu
            self.cov   = sigma
            self.bic   = bic

class GMM_MISSING_MPI(object):
    """
    This class fits a Gaussian Mixture Model of full covariance to the data
    """
        

    def __init__(self,data,nComp=2,tol=1e-4,maxIter=1000,rho=0.5,tolCov=1e-2,n_tri=1,
                    ipro=0,imu=0,isigma=0,init="GMM"):
        nComp = np.array(nComp)
        data = np.array(data)
        if (np.any(nComp<=1)):
            print("Cluster number must be grater than one.\n Save time! use other function.")
        N_g  = len(nComp)
        hbic = np.infty
        self.bics=[]
        N,D  = data.shape        
        rank,size = comm.Get_rank(),comm.Get_size()

        #------- finds missing data----
        isobs    = np.isfinite(data)
        ind_comp = np.where(np.sum(isobs,axis=1) == D)[0]
        N_c      = len(ind_comp)
        N_m      = N - N_c
        pC       = (100.*N_c)/(1.0*N)
        if rank==0:
            print "Percentage of complete data:",pC
        if (N_c == 0):
            print("Not a single complete datum.")
        #------------------

        my_N   = N/size

        if N%size != 0 :
            sys.exit("Data not divisible by CPU's.")

        # ----- Scatters data -------

        my_indx  = range(my_N*rank,my_N*rank+my_N)
        my_data  = data[my_indx,]

        # computes own index
        isobs    = np.isfinite(my_data)

        #--------- Loop over different Models -----
        for G in nComp :
            # ----- initial set of parameters and Gaussians--------
            z       = np.empty((N,G))
            my_z    = np.empty((my_N,G))
            pro     = np.empty(G)
            mu      = np.empty((G,D))
            sigma   = np.empty((G,D,D))
            veritas = True
            noPSD   = False
            
            cTri    = 1
            while veritas and cTri <= n_tri :
                if rank==0:
                    if init=="GMM":
                        #-------- GMM ---------
                        gmm   = mixture.GMM(n_components=G, 
                            covariance_type='full',tol=tol,
                            random_state=int(time()))
                        gmm.fit(data[ind_comp,])
                        pro   = gmm.weights_
                        mu    = gmm.means_
                        sigma = gmm.covars_
                        if not gmm.converged_:
                            sys.exit("mixture.GMM did not converged!")
                    # ----- one last chance -----
                    # ------ this may take longer to converge -----
                    # if cTri== n_tri:
                    #     sigma = np.repeat(np.cov(data[ind_comp,],rowvar=0)[None,:,:],G,axis=0)
                    if init=="K-means" : 
                        #---k-means -----------
                        km    = cluster.KMeans(n_clusters=G,max_iter=1000,n_init=1,tol=tol,random_state=int(time()))
                        km.fit(data[ind_comp,])
                        mu    = km.cluster_centers_
     
                        for i in range(G):
                            idx    = np.where(km.labels_==i)[0]
                            pro[i] = len(idx)/float(len(ind_comp))
                            if pro[i] <=0:
                                sys.exit("Zero in fraction")
                            sigma[i] = covariance.empirical_covariance(data[ind_comp[idx],])
                            eigs = scipy.linalg.eigvals(sigma[i])
                            if np.any(eigs<0):
                                print("Setting covariance to that of all complete data.")
                                sigma[i] = covariance.empirical_covariance(data[ind_comp,])
                                eigs = scipy.linalg.eigvals(sigma[i])
                                if np.any(eigs<0):
                                    noPSD = True
                                    print ("NO pSD inital matrix.")
                                    sys.exit("NO pSD inital matrix.")

                    if init == "own":
                        #------ use initial values ----
                        pro = ipro
                        mu = imu
                        sigma = isigma

                pro     = comm.bcast(pro,root=0)
                mu      = comm.bcast(mu,root=0)
                sigma   = comm.bcast(sigma,root=0)

                
                for r in range(my_N):
                    den  = np.empty(G)
                    w = isobs[r,]
                    for i in range(G):
                        sg      = sigma[i][np.ix_(w,w)]
                        x       = my_data[r,w]-mu[i,w]
                        div     = 0.5*(sum(isobs[r,])*np.log(2*np.pi)+np.linalg.slogdet(sg)[1])
                        mvn     = -0.5*np.dot(x,np.dot(inv(sg),x.T))
                        den[i]  = pro[i]*np.exp(mvn-div)
                        if np.isnan(den[i]) or np.isneginf(den[i]) or den[i]==0 :
                            den[i] =1e-200
                        if np.isinf(den[i]):
                            den[i] =1e200
                    my_z[r,]    = den/np.sum(den)


                comm.barrier()
                # Allgather data into original array

                comm.Allgather( [my_z, MPI.DOUBLE], [z, MPI.DOUBLE] )


                veritas = np.isnan(z).any()
                if veritas and rank==0:
                    print "Z returned nan, trying again ..."

                #--- count trials ---
                cTri +=1

            if veritas and rank==0:
                    raise RuntimeError("After n_tri trials Z remained NaN")


            

           #----- DO fit ---------- 
            comm.barrier()
            fit    = OneModelMPI(my_data,G,my_z,pro,mu,sigma,isobs,tol,maxIter,rho,tolCov)

            #------ BIC comparison ------
            if rank == 0:
                print "BIC of model with ",G," gaussians: ",fit.bic
            self.bics.append(fit.bic)
            if fit.bic < hbic:
                hbic      = fit.bic
                self.best = fit
            comm.barrier()

class OneModelMPI(object):
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

        def __init__(self,my_data,G,my_z,pro,mu,sigma,isobs,tol,maxIter,rho,tolCov):
            my_N,D = my_data.shape
            ollik  = -1
            rates  = np.array([1.0,1.0,1.0,1.0,1.0]) 
            rate   = 1.0
            count  = 1
            rank   = comm.Get_rank()
            size   = comm.Get_size()
            N      = my_N*size
            z      = np.empty((N,G))
            cond   = True
            opro   = pro
            omu    = mu
            osigma = sigma


            while cond:
                llik  = 0.0
                den  = np.empty(G)
                lliks = np.zeros(N)
                my_lliks = np.zeros(my_N)
                #-------------------- E step ----------------
                for r in range(my_N):
                    w = isobs[r,]
                    for i in range(G):
                        sg      = sigma[i][np.ix_(w,w)]
                        x       = my_data[r,w]-mu[i,w]
                        div     = 0.5*(sum(isobs[r,])*np.log(2*np.pi)+np.linalg.slogdet(sg)[1])
                        mvn     = -0.5*np.dot(x,np.dot(inv(sg),x.T))
                        den[i]  = pro[i]*np.exp(mvn-div)
                        if np.isnan(den[i]) or np.isneginf(den[i]) or den[i]==0 :
                            den[i] =1e-200
                        if np.isinf(den[i]):
                            den[i] =1e200
                    my_z[r,]    = den/np.sum(den)
                    my_lliks[r] = np.log(den[np.argmax(my_z[r,])])
                comm.barrier()
                # Allgather data into original array
                comm.Allgather( [my_z, MPI.DOUBLE], [z, MPI.DOUBLE] )
                comm.Allgather( [my_lliks, MPI.DOUBLE], [lliks, MPI.DOUBLE] )

                
                if np.isnan(z).any():
                    raise RuntimeError(
                        "Probability returned NaN")
                    sys.exit()

                llik = np.sum(lliks)
                if rank ==0:
                    print llik
                # ------------------ M step ------------------
                # compute parameters
                # ----- fractions
                pro  = np.sum(z,axis=0)/N
                assert np.all(pro >0) and np.all(pro <1)
                #------ means ---------
                for i in range(G):
                    mu_a   = np.zeros((N,D,D))
                    mu_b   = np.zeros((N,D))
                    mmu_a  = np.zeros((my_N,D,D))
                    mmu_b  = np.zeros((my_N,D))
                    for r in range(my_N):
                        w  = isobs[r,]
                        sg = sigma[i][np.ix_(w,w)]
                        M  = np.eye(D)[w,] 
                        H  = np.dot(M.T,inv(sg))
                        mmu_a[r,:,:] = my_z[r,i]*np.dot(H,M)
                        mmu_b[r,:]   = my_z[r,i]*np.dot(H,my_data[r,w])
                    comm.barrier()
                    # Allgather data into A again
                    comm.Allgather( [mmu_a, MPI.DOUBLE], [mu_a, MPI.DOUBLE] )
                    comm.Allgather( [mmu_b, MPI.DOUBLE], [mu_b, MPI.DOUBLE] )

                    if rank==0:
                        mu[i] = np.dot(inv(np.sum(mu_a,axis=0)),np.sum(mu_b,axis=0))
                    mu[i] = comm.bcast(mu[i])

                    
                    #-------  covariances ----------
                    dif = 1
                    while dif > tolCov :
                        delta    = np.zeros((N,D,D))
                        my_delta = np.empty((my_N,D,D))
                        for r in range(my_N):
                            w    = isobs[r,]
                            sg   = sigma[i][np.ix_(w,w)]
                            M    = np.eye(D)[w,]
                            H    = np.dot(M.T,inv(sg))
                            out  = np.outer((my_data[r,w]-mu[i,w]),(my_data[r,w]-mu[i,w]))
                            my_delta[r] = my_z[r,i]*(np.dot(H,np.dot(out,H.T))-np.dot(H,np.eye(D)[w,:]))
                        comm.barrier()
                        comm.Allgather( [my_delta, MPI.DOUBLE], [delta, MPI.DOUBLE] )
                        if rank==0:
                            sdel      = np.sum(delta,axis=0)/(N*pro[i])
                            sdel      = 0.5*(sdel+sdel.T)
                            Delta     = 0.5*(np.dot(sigma[i],np.dot(sdel,sigma[i])))
                            nosg      = np.linalg.norm(sigma[i])
                            osigma    = sigma[i].copy()
                            try:
                                np.linalg.cholesky(sigma[i])
                                sigma[i]  = osigma + rho*Delta
                            except np.linalg.LinAlgError:
                                sigma[i]  = osigma
                                print "NO pSD"

                            nnsg      = np.linalg.norm(sigma[i])
                            dif       = np.abs(1.0-(nnsg/nosg))
                        sigma[i] = comm.bcast(sigma[i])
                        dif      = comm.bcast(dif)
                        comm.barrier()
                #---------- Comparison------------
                rate     = (llik-ollik)/np.abs(ollik)
                ollik    = llik
                rates[4] = rates[3]
                rates[3] = rates[2]
                rates[2] = rates[1]
                rates[1] = rates[0]
                rates[0] = rate
                count += 1
                cond   = not ((np.all(np.abs(rates) < tol)) or (count > maxIter))#and np.all(rates > 0)

            if rank==0:
                print "Done in ",count," iterations."
                if count == maxIter: print "Max iteration reached!"
            
            #---- BIC ------
            kmod = (G-1)+(G*D)+(G*(D*(D+1)*0.5)) # parameters of fractions,means and covariance matrices
            bic  = -2*llik+(kmod)*np.log(N)

            self.frac  = pro
            self.mean  = mu
            self.cov   = sigma
            self.bic   = bic


# # # ###############
# # # #-------------- DATA ----------------------
# # # Number of samples per component
# N_m  = 8000
# D    = 5


# # # #--------- hyper parameters ----
# n1  = 1000
# n2  = 1000
# n3  = 1000
# n4  = 1000
# n5  = 1000
# n6  = 1000
# n7  = 1000
# n8  = 1000
# n9  = 1000
# n0  = 1000
# mu1 = np.repeat(-8,D)
# mu2 = np.repeat(-6,D)
# mu3 = np.repeat(-4,D)
# mu4 = np.repeat(-2,D)
# mu5 = np.repeat(0,D)
# mu6 = np.repeat(2,D)
# mu7 = np.repeat(4,D)
# mu8 = np.repeat(6,D)
# mu9 = np.repeat(8,D)
# sg = 1.0*np.eye(D)
# # #-------------------------------
# N      = n1+n2+n3+n4+n5+n6+n7+n8+n9
# tpro   = np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9])/(1.0*N)#
# tmu    = np.stack((mu3,mu8,mu5,mu1,mu9,mu7,mu4,mu6,mu2),axis=0)
# tsigma = np.stack((sg,sg,sg,sg,sg,sg,sg,sg,sg),axis=0)#

# # Generate random sample, two components
# np.random.seed(0)
# X1 = np.random.multivariate_normal(mu1, sg,n1)
# X2 = np.random.multivariate_normal(mu2, sg,n2)
# X3 = np.random.multivariate_normal(mu3, sg,n3)
# X4 = np.random.multivariate_normal(mu4, sg,n4)
# X5 = np.random.multivariate_normal(mu5, sg,n5)
# X6 = np.random.multivariate_normal(mu6, sg,n6)
# X7 = np.random.multivariate_normal(mu7, sg,n7)
# X8 = np.random.multivariate_normal(mu8, sg,n8)
# X9 = np.random.multivariate_normal(mu9, sg,n9)

# X  = np.vstack((X1,X2,X3,X4,X5,X6,X7,X8,X9))

# #------- missing data ----------
# X_miss = X.copy()
# ind_miss  = np.random.choice(np.arange(N),N_m,replace=False)
# for i in range(N_m):
#     X_miss[ind_miss[i],np.random.permutation(np.arange(D))[:3]] = np.nan
# #-------------------
# t1 = time()
# fit = GMM_MISSING_MPI(X_miss,[9],tol=1e-5,tolCov=1e-3,rho=0.1)
# if rank ==0:
#     print time()-t1
#     print np.sum(fit.best.frac-tpro)
#     print np.sum(fit.best.mean-tmu)
#     print np.sum(fit.best.cov-tsigma)
#     print fit.best.cov

