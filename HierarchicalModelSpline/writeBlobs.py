# Copyright (C) 2013 ETH Zurich, Institute for Astronomy

'''
Created on Mar 6, 2014

author: jakeret

Example of a file util that extends the functionallity of the 
standard ``SampleFileUtilSampleFileUtil`` in order to save additional parameters

'''
from cosmoHammer.util import SampleFileUtil

VALUE_1_KEY = "PC1"
VALUE_2_KEY = "PC2"
# VALUE_3_KEY = "PhPs"
# VALUE_4_KEY = "PhBs"

# OUTPUT_ORDER = [VALUE_1_KEY,VALUE_2_KEY,VALUE_3_KEY,VALUE_4_KEY]
OUTPUT_ORDER = [VALUE_1_KEY,VALUE_2_KEY]

class DerivedParamterFileUtil(SampleFileUtil):
    '''
    Persists the derived parameters to a file
    '''

    def __init__(self, filePrefix, master=True, reuseBurnin=False):
        '''
        Constructor
        '''
        super(DerivedParamterFileUtil, self).__init__(filePrefix, master=master, reuseBurnin=reuseBurnin)
        
        self.paramsFile = open(self.filePrefix+"_params.dat", "w")
        
    def persistValues(self, posFile, probFile, pos, prob, data):
        #make sure the super class writes the position and likelihood to the disk
        super(DerivedParamterFileUtil, self).persistValues(posFile, probFile, pos, prob, data)
        
        #handle special parameters
        for derParams in data:
            self.paramsFile.write("\t".join(derParams[key] for key in OUTPUT_ORDER))
            self.paramsFile.write("\n")
                                  
        self.paramsFile.flush();