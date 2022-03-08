#!/usr/bin/python3
#
# N-point Discrete Fourier transform unit through FFT algorithm
#
# Author: Greyson Mitra
# Date: 2021-05-27
#

import numpy as np

class Demodulator:

    def __init__(self, subcarrierTotal, usedSubcarriers, emptySubcarriers):
        self.subcarrierTotal = subcarrierTotal
        self.usedSubcarriers = usedSubcarriers
        self.emptySubcarriers = emptySubcarriers
        self.reverseIndex = -1*(self.usedSubcarriers//2)
        

    # complexArr will be a 1D complex array.   
    def dft_demod(self, complexArr):
        ofdmSymbols = complexArr.size // self.subcarrierTotal
        complexArr = np.reshape(complexArr, (ofdmSymbols, self.subcarrierTotal))
        tform = np.fft.fft(complexArr)

        #return tform
        
        #print('undone IDFT: %r'%tform)

        tform = np.reshape(tform, (ofdmSymbols, self.subcarrierTotal)) #Break the tformed array into an array of ofdm symbols again.
        out = np.zeros((ofdmSymbols, self.usedSubcarriers), dtype=complex)
        for i in range(ofdmSymbols): #Rearrange the data so that it is in the same order it was put into IDFT module
            for j in range(self.usedSubcarriers):
                if self.reverseIndex+j < 0:
                    #print(self.reverseIndex+j, j)
                    out[i][j] = tform[i][self.reverseIndex+j]
                else:
                    #print(self.reverseIndex+j+1, j)
                    out[i][j] = tform[i][self.reverseIndex+j+1]


        return out.flatten()
    
    #def temp_rm_subs(self, tform):
    #    ofdmSymbols = tform.size // self.subcarrierTotal
    #    tform = np.reshape(tform, (ofdmSymbols, self.subcarrierTotal)) #Break the tformed array into an array of ofdm symbols again.
    #    out = np.zeros((ofdmSymbols, self.usedSubcarriers), dtype=complex)
    #    for i in range(ofdmSymbols): #Rearrange the data so that it is in the same order it was put into IDFT module
    #        for j in range(self.usedSubcarriers):
    #            out[i][j] = tform[i][self.reverseIndex+j]
    #    return out.flatten()

    
    def rm_prefix(self, complexArr, prefixLen):
        ofdmSymbols = complexArr.size // (self.subcarrierTotal+prefixLen) #Number of symbols with added prefix in input data
        complexArr = np.reshape(complexArr, (ofdmSymbols, self.subcarrierTotal+prefixLen)) #Make subarrays with each subarray having prefix+subcarriers length each
        trimmed = np.zeros((ofdmSymbols, self.subcarrierTotal), dtype=complex) 

        for i in range(ofdmSymbols): #Each subarray has the cyclic prefix trimmed off and stored into new array 
            trimmed[i] =  complexArr[i][0+prefixLen:complexArr.size:1]

        return trimmed.flatten()
