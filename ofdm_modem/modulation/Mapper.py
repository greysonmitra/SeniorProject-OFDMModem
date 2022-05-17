#!/usr/bin/python3
#
# Modulation mapper through QPSK
#
# Author: Greyson Mitra
# Date: 2021-05-24
#

import numpy as np

class Mapper:

    def __init__(self):
        self.oddInput = None


    # bitStream will be a 1D array. Map bits in groups of 2 to complex exp's 
    def map(self, bitStream):
        if(self.oddInput != None): #If there is bit leftover from previous input prepend it to new data
            bitStream = np.append(self.oddInput, bitStream)
            self.oddInput = None

        
        parity = bitStream.size % 2
        outLength = int(bitStream.size/2) #Cast to int because get float when dividing??
        
        if parity == 1: #Input bit array is odd so save an element for the next set of input data
            self.oddInput = bitStream[bitStream.size-1]
            bitStream = bitStream[0:bitStream.size-1:1]

        out = np.zeros((bitStream.size//2,), dtype=complex)
  
        for i in range(out.size):
            bitIndex1 = 2*i
            bitIndex2 = (2*i)+1
            
            if (bitStream[bitIndex1]==0) and (bitStream[bitIndex2]==0):
                out[i] = np.exp(1j*((np.pi)/4))
            elif (bitStream[bitIndex1]==0) and (bitStream[bitIndex2]==1):
                out[i] = np.exp(1j*7*((np.pi)/4))
            elif (bitStream[bitIndex1]==1) and (bitStream[bitIndex2]==0):
                out[i] = np.exp(1j*3*((np.pi)/4))
            else: #bit pattern 11
                out[i] = np.exp(1j*5*((np.pi)/4))

        return out
