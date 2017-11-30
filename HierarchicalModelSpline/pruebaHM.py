#!/usr/bin/env python
import sys
import numpy as np
import h5py
from time import time
import pandas

from cosmoHammer import CosmoHammerSampler as CHSampler
from writeBlobs import DerivedParamterFileUtil
from cosmoHammer.util.UniformIntervalPositionGenerator import UniformIntervalPositionGenerator
# from cosmoHammer.util.SampleTruncatedBallPositionGenerator import SampleTruncatedBallPositionGenerator

# from mpi_pool import MPIPool,_close_pool_message

from cosmoHammer import LikelihoodComputationChain
from LogPosteriorParallel import LogPosteriorModule
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

#Check that lnprob in inital position wont be infinite AT ANY WALKER! otherwise it can get stucked.

#Parallelization of CosmoHammer on a cluster or cloud with N nodes and n cores per node. 
#For distributing the workload between different nodes in the cluster, MPI has to be used. 
#Run your python script with:

# $mpiexec -n $NUM ./<script>.py,

#where $NUM>=N is the number of MPI jobs and <script> is your python script to launch the CosmoHammer. 
#Note that $NUM cannot be greater than the number of walkers.

 #Using MPI, OpenMP, and Python multiprocessing. 
 #Choose the number m of OpenMP threads and 
 #the number k of multiprocessing threads such that n=k*m. 

 #Choose $NUM=N when executing the python script

#run before $export OMP_NUM_THREADS=m
###################     KNOBS   ########################
#----- Threads ----
k    = 12     
#----- emcee ----------------
wr   = 2  # Walkers ratio
Nbi  = 0  # Burning iterations
Nit  = 1  # Sampling iterations
###############################################################
#--------------------------------------------------------------
############################## DATA ###########################################################
#----------- Data Files -----
# Data loaded from
data = '/pcdisk/boneym5/jolivares/Data/Boneym_10G_1e+04/Dataset.h5'
init = '/pcdisk/boneym5/jolivares/Data/Boneym_10G_1e+04/Initial_prueba.h5'
samp = '/pcdisk/boneym5/jolivares/Prueba/prueba'
#----------------------------
##################################### INITIAL SOLUTION #########################
print 'Reading Initial Solution'
with h5py.File(init,'r') as hf:
    params = np.array(hf.get("params"))
    positions = np.array(hf.get("positions"))
    szs       = np.array(hf.get("szs_parms")).astype(int)

#--------------------------------------------------------------------------------
#################################################################################################
####################################### LIKELIHOOD CHAIN ########################
#-------------------- Setup the Chain --------
# if mins, and maxs included, the code performs checkin 
# and can calculate a null iteration of Generated Quantities in case of rejection
chain        = LikelihoodComputationChain()
coremodule   = CoreModule(szs)
logPosterior = LogPosteriorModule(data,threads=k)
# logPosterior = LogPosteriorModule(data,hypr)
chain.addCoreModule(coremodule)
chain.addLikelihoodModule(logPosterior)
chain.setup()
#######################################################################################
############################## SAMPLER ########################################
#-------- Writes Derived parameters into a file------
storageUtil = DerivedParamterFileUtil(samp)
#------------- Setup the sampler -----------
sampler = CHSampler(
                params= params, 
                likelihoodComputationChain=chain, 
                filePrefix=samp, 
                walkersRatio=wr, 
                burninIterations=Nbi, 
                sampleIterations=Nit,
                storageUtil=storageUtil,
                initPositions=positions,
                threadCount=1,
                )
#----------------------- Start sampling-------------------------

print("start sampling")
t1 = time()
sampler.startSampling()
b = time()-t1
print("done!")
print 'Doing ',Nit,' iterations takes ',b,' seconds'
sys.exit(0)

