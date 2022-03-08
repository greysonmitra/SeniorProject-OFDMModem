#!/usr/bin/python3
#
# N-point Inverse discrete Fourier transform unit through IFFT algorithm
#
# Author: Greyson Mitra
# Date: 2021-05-26
#

import numpy as np

class Modulator:

    def __init__(self, subcarrierTotal, usedSubcarriers, emptySubcarriers):
        self.subcarrierTotal = subcarrierTotal
        self.usedSubcarriers = usedSubcarriers
        self.emptySubcarriers = emptySubcarriers
        self.rollingInput = np.array([])
        self.reverseIndex = -1*(self.usedSubcarriers//2)
        

    # complexArr will be a 1D complex array.   
    def idft_mod(self, complexArr):

        if self.rollingInput.size > 0:
            complexArr = np.append(self.rollingInput, complexArr) #Put the rolling input onto the front of the new input and store back in variable 

        if complexArr.size < self.usedSubcarriers: #If the data is too small, just store in rollingInput until get more data
            self.rollingInput = complexArr
            
        ofdmSymbols = complexArr.size // self.usedSubcarriers #Number of OFDM symbols in the data
        untformed = np.zeros((ofdmSymbols, self.subcarrierTotal), dtype=complex) #Empty array to fit all OFDM symbols but with empty subcarriers

        if complexArr.size >= self.usedSubcarriers: #If the data has more symbols than the desired amount of subcarriers (240) calculate the how much to slice off and store as rolling input
            remainderStartIndex = complexArr.size - (complexArr.size % self.usedSubcarriers)
            self.rollingInput = complexArr[remainderStartIndex:complexArr.size:1]

            complexArr = complexArr[0:remainderStartIndex:1]
            complexArr = np.reshape(complexArr, (complexArr.size // self.usedSubcarriers, self.usedSubcarriers)) #Make an n-dim array AKA an array of arrays where each sub-array is a group of 240 (or any desired size of symbol) QPSK symbols

            for i in range(ofdmSymbols): #Loop for number of sets of 240 (or desired number of used subcarriers), but decrement
                for j in range(self.usedSubcarriers): #Loop from 0 to 240 (or desired amount of used subcarriers) for an individual OFDM symbol
                    if self.reverseIndex+j < 0: #Avoids using subcarrier 0. Uses subcarriers 1-120 instead of 0-119
                        #print(self.reverseIndex+j, j)
                        untformed[i][self.reverseIndex+j] = complexArr[i][j] #Rearrange QPSK symbols so that the first symbol goes to subcarrier -120 and last goes to 119 (or from any desired halfway point of used subcarriers)
                    else:
                        #print(self.reverseIndex+j+1, j)
                        untformed[i][self.reverseIndex+j+1] = complexArr[i][j]



        #untformed = untformed.flatten() #Flatten the array back to 1D before inverse tranforming...
                                         #no need to flatten. The correct operation anyways would be to do a separate IDFT on each set of 256 which the IFFT function can handle automatically if a multidimensional array is passed to it.
        invTformed = np.fft.ifft(untformed)

        return invTformed.flatten()


    #untformed = untformed.flatten() #Flatten the array back to 1D before inverse tranforming...
                                         #no need to flatten. The correct operation anyways would be to do a separate IDFT on each set of 256 which the IFFT function can handle automatically if a multidimensional array is passed to it.

    
    # Add cyclic prefix onto the time domain signal produced by the IDFT unit
    def add_prefix(self, complexArr, prefixLen):
        ofdmSymbols = complexArr.size // self.subcarrierTotal
        complexArr = np.reshape(complexArr, (ofdmSymbols, self.subcarrierTotal)) #Turn the inverse tranformed array back into n-dimensions
        prepended = np.zeros((ofdmSymbols, self.subcarrierTotal + prefixLen), dtype=complex) #Prepare an empty array to fit all OFDM symbols plus a prefix added onto each symbol

        for i in range(ofdmSymbols): #i represents a single OFDM symbol in the larger array of ofdm symbols
            prefixStart = complexArr[i].size - prefixLen
            prefixSlice = complexArr[i][prefixStart:complexArr[i].size:1] #Create slice of last 18 samples of the ith OFDM symbol
            prepended[i] = np.append(prefixSlice, complexArr[i]) #Append the actual OFDM symbol onto the prefix and store in ith location of larger array

        return prepended.flatten()
