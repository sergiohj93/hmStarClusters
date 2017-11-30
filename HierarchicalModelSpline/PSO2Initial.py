#!/usr/bin/env python
import sys
import numpy as np
import h5py
from time import time
import pandas


###################     KNOBS   ########################
usePOS = False
usePSO = True
# in case of PSo How many particles in the pso file?
partic = 40
wr     = 2  # Walkers ratio
###############################################################
#--------------------------------------------------------------
############################## DATA ###########################################################
#----------- Data Files -----
# Data loaded from
dir_   = "/pcdisk/boneym5/jolivares/Data/Boneym_10G_0.8_1e+04/"
init   = dir_ + "Initial.h5"
out    = dir_ + "Initial_hm_syn.h5"

posi   = "/pcdisk/boneym5/jolivares/Boneym/pso/37_swarm.out"
# posi   = "/pcdisk/boneym5/jolivares/Boneym/samples/71_burnin.out"

lstitr = 280 # discard last lstitr iterations

print "Last ",lstitr," will be discarded!"
#----------------------------
##################################### INITIAL SOLUTION #########################
print 'Reading Initial Solution'
print posi
with h5py.File(init,'r') as hf:
    params = np.array(hf.get("params"))
    inpos  = np.array(hf.get("positions"))
    szs    = np.array(hf.get("szs_parms")).astype(int)

#--------------------------------------------------------------------------------
#######################################################################################
if usePSO :
        print "Reading Position from PSO"
        nwlk    = len(params)*wr
        swarm = np.array(pandas.read_csv(posi, sep='\t', comment='#',header=None))
        # if (swarm.shape[1]-2) != len(params):
        #         sys.exit("error in parameters of swarm")
        it    = swarm.shape[0]/ partic
        index = np.arange(0,swarm.shape[0]-lstitr)[-nwlk::]
        idxrep= np.arange(0,(swarm.shape[0]-lstitr)-nwlk)[::-1]
        fitness = swarm[index,1]
        positions = swarm[index,2:]
        # infinits in fitness are removed
        infs  = np.where(fitness==-np.inf)[0]
        j=0
        for i in range(len(infs)):     
                while fitness[infs[i]]==-np.inf:
                        fitness[infs[i]]=swarm[idxrep[j],1]
                        positions[infs[i]]=swarm[idxrep[j],2:]
                        j+=1
        if (np.any(np.isinf(fitness)) or np.any(np.isnan(fitness))): 
                sys.exit("infinity in fitness or nan. Try diferent index")
        # print positions[0,[18,19]]
        # print positions[0,[20,21]]
        # print positions[0,[28,29]]
        # print positions[0,[30,31]]
        # sys.exit()
        # positions = np.delete(positions,[20,21,30,31],1)
        positions = positions.reshape((nwlk,len(params)),order='F')
        swarm = None
        if (len(positions) != nwlk) or (len(positions[0]) != len(params)):
                sys.exit("error in positions")
        if np.any(np.isnan(positions)):
                sys.exit("error in positions")

if usePOS :
        print "Reading Position from previous emcee"
        nwlk    = len(params)*wr
        allpos = np.array(pandas.read_csv(posi, sep='\t', comment='#',header=None))
        positions = allpos[-nwlk::,]
        # positions = np.delete(positions,[2,4],1)
        # positions = np.insert(positions,[20],positions[:,[18,19]], axis=1)
        # positions = np.insert(positions,[28],positions[:,[28,29]], axis=1)
        if (len(positions) != nwlk) or (len(positions[0]) != len(params)):
                sys.exit("error in positions")
        if np.any(np.isnan(positions)):
                sys.exit("error in positions")

with h5py.File(out, 'w') as hf:
    hf.create_dataset('params',    data=params)
    hf.create_dataset('szs_parms', data=szs)
    hf.create_dataset('positions', data=positions)

# print len(positions[0])
# print positions[0]
print "Positions written"
print out

# print positions[1,:]-positions[0,:]


