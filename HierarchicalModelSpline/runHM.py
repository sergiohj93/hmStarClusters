#!/usr/bin/env python
import sys
import os
import numpy as np
import h5py
from time import time
# import pandas

from cosmoHammer import MpiCosmoHammerSampler as CHSampler
from writeBlobs import DerivedParamterFileUtil
from cosmoHammer.util.UniformIntervalPositionGenerator import UniformIntervalPositionGenerator
# from cosmoHammer.util.SampleTruncatedBallPositionGenerator import SampleTruncatedBallPositionGenerator

from cosmoHammer import LikelihoodComputationChain
from LogPosteriorParallel import LogPosteriorModule
# from LogPosterior import LogPosteriorModule
from Context import CoreModule

#import multiprocessing

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
a    = 1.3
wr   = 2   # Walkers ratio
Nbi  = 2  # Burning iterations
Nit  = 10  # Sampling iterations
###############################################################
#--------------------------------------------------------------
############################## DATA ###########################################################
#----------- Data Files -----
# Data loaded from
data   = sys.argv[1]
init   = sys.argv[2]
samp   = sys.argv[3]
# posi   = sys.argv[4]
#----------------------------
##################################### INITIAL SOLUTION #########################
print 'Reading Initial Solution'
with h5py.File(init,'r') as hf:
    params    = np.array(hf.get("params"))
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
#---------- Initial Position Generator --------
pos_gen   = UniformIntervalPositionGenerator()
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
                initPositionGenerator=pos_gen)     
#----------------------- Start sampling-------------------------
sampler._sampler.a = a
sampler.startSampling()
print "Done!"
sys.exit(0)

