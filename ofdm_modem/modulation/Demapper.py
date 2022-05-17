#!/usr/bin/python3
#
# Modulation demapper through for QPSK constellation
#
# Author: Greyson Mitra
# Date: 2021-05-24
#

import numpy as np

class Demapper:

    #def __init__(self):
        

    # bitStream will be a 1D array. Map complex exp's to groups of 2 bits
    def demap(self, bitStream):
        parity = bitStream.size % 2 #DFT/FFT class ensures that data is even so parity should always be 0. Data only sent in groups of 256 subcarriers which are then converted to 240 QPSK symbols.
        
        outLength = int(bitStream.size*2) #Cast to int because don't want error like mapper
        if parity == 0: #Input bit array is even, create output array twice length of input array
            out = np.zeros((outLength,), dtype=int)
        else: #Input array of symbols is odd so that's weird... maintain size of output array and print error message
            out = np.zeros((outLength,), dtype=int)
            print('ERROR: somehow odd amount of symbols recieved')


        #Assign bit pattern from QPSK symbols in certain ranges
        for i in range(bitStream.size):
            bitIndex1 = 2*i
            bitIndex2 = (2*i)+1
             
            if ( ((bitStream[i].real > 0) and (bitStream[i].imag > 0)) or ((bitStream[i].real == 0) and (bitStream[i].imag == 1)) ): #complex number is in region 1 or on edge of regions 1 and 2
                out[bitIndex1] = 0
                out[bitIndex2] = 0
            elif ( ((bitStream[i].real < 0) and (bitStream[i].imag > 0)) or ((bitStream[i].real == 1) and (bitStream[i].imag == 0)) ): #complex number is in region 2 or on edge of regions 2 and 3
                out[bitIndex1] = 1
                out[bitIndex2] = 0
            elif ( ((bitStream[i].real < 0) and (bitStream[i].imag < 0)) or ((bitStream[i].real == 0) and (bitStream[i].imag == -1)) ): #complex number is in region 3 or on edge of regions 3 and 4
                out[bitIndex1] = 1
                out[bitIndex2] = 1
            else: #complex input in fourth region or on edge of regions 4 and 1
                out[bitIndex1] = 0
                out[bitIndex2] = 1

        return out
