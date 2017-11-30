#!/usr/bin/env python
import sys
import numpy as np
import h5py
from time import time
import logging
import emcee
from cosmoHammer.pso.ParticleSwarmOptimizer import ParticleSwarmOptimizer as MPSO
from cosmoHammer.pso.ParticleSwarmOptimizer import Particle
#from cosmoHammer.util.MpiUtil import mpiMean,mpiBarrier,mpiBCast
from cosmoHammer import LikelihoodComputationChain

from LogPosteriorParallel import LogPosteriorModule
from ContextSpline import CoreModule
#########NOTES #######################
# CosmoHammer has been modified in the following aspects
# MpiParticleSwarmOptimizer has been changed in the Particle create, it is now an numpy array with shape.
# UniformIntervalPositionGenerator has been created to produce walkers uniformly covering the min max interval.
# persistValuesBurnin has been modified to write blobs ONLY on sampling (NOT in Burnin phase)
# in sample storechaine=False. it was true.
# Ensamble emcee modified to report about problems in intital lnprob.
# It is posible to initialize with walkers positions. In this way it is possible to use positions of PSO
# PSO now uses Clerc and Kennedy 2002 values instead of Kennedy 2001 (inertia)
#######################################

###################     KNOBS   ########################
#----- Threads ----
k    = 1    
#-----PSO ----
partCount = 2       # Number of particles
maxIter   = 1        # Max number of iterations     
maxNorm   = 0.1      # Max relative norm to stop PSO
fracPart  = 1.0      # Fraction of particles to use in mean and norm computations
#----- emcee -------
mcmc_it   = 1        # emcee iterations between PSOs
tol       = 1        # Relative tolerance to stop PSO and PSO-emcee
maxCount  = 1        # Max number of PSO-emcee iterations
###############################################################
#--------------------------------------------------------------
############################## DATA ###########################################################
#----------- Data Files -----
'''# Data loaded from
data = '/home/jromero/HierarchicalModel/Dataset_syn_complete.h5' # Dataset and constants
init = '/home/jromero/HierarchicalModel/Initial_hm.h5' # Initial positions and shapes of group parameters
swrm = '/home/jromero/HierarchicalModel/Prueba/swarm.out'  '''

# Data loaded from
data = 'G:/Uned/TFM/hmStarClusters-develop/HierarchicalModelSpline/Dataset.h5' # Dataset and constants
init = 'G:/Uned/TFM/hmStarClusters-develop/HierarchicalModelSpline/Initial_hm.h5' # Initial positions and shapes of group parameters
swrm = 'G:/Uned/TFM/hmStarClusters-develop/HierarchicalModelSpline/Prueba/swarm.out'                      # Output file                    # Output file
#----------------------------
##################### LOG #############
logLevel=logging.INFO
fileLog =swrm.replace("swarm.out", "log.out")
#------ Logger ----
logger=logging.getLogger(__name__)
logger.setLevel(logLevel)
fh = logging.FileHandler(fileLog, "w")
fh.setLevel(logLevel)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
def log(logger, message, level=logging.INFO):
    """
    Logs a message to the logfile
    """
    logger.log(level, message)
########################################
##################################### INITIAL SOLUTION #########################
print 'Reading Initial Solution'
with h5py.File(init,'r') as hf:
    params = np.array(hf.get("params"))
    inpos  = np.array(hf.get("positions"))
    szs    = np.array(hf.get("szs_parms")).astype(int)

mins = params[:,1]
maxs = params[:,2]
nparam = len(mins)

#--------------------------------------------------------------------------------
#################################################################################################
####################################### LIKELIHOOD CHAIN ########################
#-------------------- Setup the Chain --------
# if mins, and maxs included, the code performs checkin 
# and can calculate a null iteration of Generated Quantities in case of rejection
print 'Setting up chain'
# lik_pool = multiprocessing.Pool(k)
chain        = LikelihoodComputationChain()
coremodule   = CoreModule(szs)
logPosterior = LogPosteriorModule(data, threads=k)
# logPosterior = LogPosteriorModule(data)
chain.addCoreModule(coremodule)
chain.addLikelihoodModule(logPosterior)
chain.setup()
#######################################################################################
############################### Particle Swarm Optimizer #############################
pso = MPSO(chain, low=mins, high=maxs, particleCount=partCount,req=1e-8,threads=1,InPos=inpos)
smp = emcee.EnsembleSampler(partCount,nparam,chain,
                        threads=1,
                        a=2.0,
                        pool=pso.pool,
                        live_dangerously=True)
print "Starting PSO"
# Saves the best positions into file
delta    = np.inf 
ofly_lnp = -np.inf
p0    = np.zeros((partCount,nparam))
count = 0
log(logger,"Particles: %s, Max Iterations PSO : %s, mean: %s, n: %s, p: %s "%(
                            partCount,maxIter,tol,maxNorm,fracPart))
with open(swrm, "w") as f:
    t1 = time()
    while delta > tol and count < maxCount:
        count +=1
        #----- do PSO -------------
        for i, cswarm in enumerate(pso.sample(maxIter=maxIter,p=fracPart,m=tol,n=maxNorm)):
                if(pso.isMaster()):
                        print(i)     
                        for particle in cswarm:
                            f.write("%s\t%f\t"%(i, particle.fitness))
                            f.write("\t".join(['{:025.20f}'.format(p) for p in particle.position]) + "\n")
                            f.flush()
        #--------- compute differences ------------
        t2 = time()-t1
        print "Time pso",t2
        sys.exit()
        if(pso.isMaster()):

            log(logger,"Iteration: %s .Time: %s seconds"%(count,time()-t1))
            log(logger,"Best found: %f \n %s"%(pso.gbest.fitness, pso.gbest.position))
            p0       = np.array([part.position for part in cswarm])
        p0 = mpiBCast(p0)
        smp.reset()
        for pos, prob, rstate, datas in smp.sample(p0, iterations=mcmc_it,storechain=False):
            if(pso.isMaster()):
                for i in range(partCount):
                    f.write("%s\t%f\t"%(-count, prob[i]))
                    f.write("\t".join(['{:025.20f}'.format(q) for q in pos[i]]) + "\n")
                    f.flush()
        if(pso.isMaster()):
            delta    = np.abs(1.0 - np.abs(pso.gbest.fitness/ofly_lnp))
            ofly_lnp = pso.gbest.fitness
            log(logger, "The relative difference between last two runs is %s "%(delta))
            log(logger,"Mean acceptance fraction:%s"%(np.mean(smp.acceptance_fraction)))
            for i in range(partCount):
                pso.swarm[i].fitness  = prob[i]
                pso.swarm[i].position = pos[i]
        delta     = mpiBCast(delta)
        pso.swarm = mpiBCast(pso.swarm)  
    #------ launch a final run of emcee with a=8.0 to over disperse the initial solution ----
    p0 = mpiBCast(pos)    
    for pos, prob, rstate, datas in smp.sample(p0, iterations=2,storechain=False):
            if(pso.isMaster()):
                for i in range(partCount):
                    f.write("%s\t%f\t"%(np.infty, prob[i]))
                    f.write("\t".join(['{:025.20f}'.format(q) for q in pos[i]]) + "\n")
                    f.flush()

if(pso.isMaster()):
        log(logger,"emcee iterations: %s, Total (PSO+emcee) iterations: %s "%(mcmc_it,count))
        log(logger,"The time consumed was %s seconds"%(time()-t1))
logPosterior.pool.close()
sys.exit("Done")

