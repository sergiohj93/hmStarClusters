#!/usr/bin/env python
import sys
import os
import numpy as np
import h5py
from time import time
import logging
import emcee
from cosmoHammer.pso.MpiParticleSwarmOptimizer import MpiParticleSwarmOptimizer as MPSO
from cosmoHammer.pso.ParticleSwarmOptimizer import Particle
from cosmoHammer.util.MpiUtil import mpiMean,mpiBarrier,mpiBCast
from cosmoHammer import LikelihoodComputationChain
from LogPosteriorCuda import LogPosteriorModule
from Context import CoreModule

#########NOTES #######################
#CosmoHammer has been modified in the following aspects
# One of the walkers has params[:,0] as initial position
# MpiParticleSwarmOptimizer has been changed in the Particle create, it is now an numpy array with shape.
# UniformIntervalPositionGenerator has been created to produce walkers uniformly covering the min max interval.
# persistValuesBurnin has been modified to write blobs ONLY on sampling (NOT in Burnin phase)
# in sample storechaine=False. it was true.
# Ensamble emcee modified to report about problems in intital lnprob.
# It is posible to initialize with walkers positions. In this way it is possible to positions of PSO
# PSO now uses Clerc and Kennedy 2002 values instead of Kennedy 2001 (inertia)
#######################################

###################     KNOBS   ########################
#----- Threads ----
k    = 1     
#-----PSO ----
partCount = 40
maxIter   = 5000
tol       = 1e-5 #Relative tolerance of pso an iterative pso convergences
meanNorm  = 1e-2  # idem but on norm and just for pso
fracPart  = 1.0
mcmc_it   = 100
maxCount  = 50    # max number of emcee repetitions 
nIterFin  = 2000  # emcee iterations after convergence. This disperse solution and helps to reduce emcee time.
req       = 1e-10  # Relative distance at which forces in pso are approx in equilibrium. 
# Must be higher than the format in which positions are written; otherwise  written positions will be identical.
###############################################################
#--------------------------------------------------------------
############################## DATA ###########################################################
#----------- Data Files -----
# Data loaded from
data   = sys.argv[1]
init   = sys.argv[2]
swrm   = sys.argv[3]
#----------------------------
fileParams = swrm.replace("swarm.out", "params.out")
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
chain        = LikelihoodComputationChain()
coremodule   = CoreModule(szs)
logPosterior = LogPosteriorModule(data, threads=k)
chain.addCoreModule(coremodule)
chain.addLikelihoodModule(logPosterior)
chain.setup()
#######################################################################################
############################### Particle Swarm Optimizer #############################
pso = MPSO(chain, low=mins, high=maxs, particleCount=partCount,req=req,threads=1,InPos=inpos)
smp = emcee.EnsembleSampler(partCount,nparam,chain,
                        threads=1,
                        a=1.5,
                        pool=pso.pool, # this must be the pool of MPI from MPSO.
                        live_dangerously=True)
print "Starting PSO"
# Saves the best positions into file
delta    = np.inf 
ofly_lnp = -np.inf
p0    = np.zeros((partCount,nparam))
count = 0
log(logger,"Particles: %s, Max Iterations PSO : %s, mean: %s, n: %s, p: %s "%(
                            partCount,maxIter,tol,meanNorm,fracPart))
f  = open(swrm, "w")
t1 = time()
#---iterate PSO + emcee ---till they converge or reach maxCount -----
while delta > tol and count < maxCount:
    print "Done",count,"iterations of PSO + emcee from a max of",maxCount,"."
    count +=1
    #----- do PSO -------------
    t2 = time()
    for i, cswarm in enumerate(pso.sample(maxIter=maxIter,p=fracPart,m=tol,n=meanNorm)):
            if(pso.isMaster()):
                    for particle in cswarm:
                        f.write("%s\t%f\t"%(i, particle.fitness))
                        f.write("\t".join(['{:025.20f}'.format(p) for p in particle.position]) + "\n")
                        f.flush()
    #--------- compute differences ------------
    if(pso.isMaster()):
        log(logger,"Iteration: %s .Time: %s seconds"%(count,time()-t1))
        log(logger,"Best found: %f \n %s"%(pso.gbest.fitness, pso.gbest.position))
        p0       = np.array([part.position for part in pso.swarm])
    p0 = mpiBCast(p0)
    smp.reset()
    print "emcee"
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
    print "Time PSO+emcee:",time()-t2
    print "Total time:",time()-t1i
    
    
#------ launch a final run of emcee with a=8.0 to over disperse the initial solution ----
p0 = mpiBCast(pos)  
cont = 0
for pos, prob, rstate, datas in smp.sample(p0, iterations=nIterFin,storechain=False):
        if(pso.isMaster()):
            for i in range(partCount):
                f.write("%s\t%f\t"%(cont, prob[i]))
                f.write("\t".join(['{:025.20f}'.format(q) for q in pos[i]]) + "\n")
                f.flush()
            cont += 1
            print(cont)
f.close()


if(pso.isMaster()):
        log(logger,"emcee iterations: %s, Total (PSO+emcee) iterations: %s "%(mcmc_it,count))
        log(logger,"The time consumed was %s seconds"%(time()-t1))
logPosterior.pool.close()
# sys.exit("Done")

