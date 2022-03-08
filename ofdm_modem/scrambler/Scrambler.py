#!/usr/bin/python3
#
# PRBS through multiplicative scrambler
#
# Author: Greyson Mitra
# Date: 2021-05-20
#

import numpy as np
import sys

class Scrambler:

    def __init__(self, tapsBits, shiftRegLength):
        self.shiftReg = 0
        self.tapsBits = tapsBits
        self.shiftRegLength = shiftRegLength
        

    # bitStream will be a 1D array.   
    def scrambleOld(self, bitStream):
        out = np.zeros((bitStream.size,), dtype=int) # Create output array for scrambled data same length as input array
        
        for i in range(out.size):
            tap1 = (self.shiftReg >> 5) & 1 #Select 17th bit of shiftReg and isolate/extract it by essentially shifting it to the 2^0 position and anding with 1 (which is equal to 2^0)
            #print(bin(1<<5))
            tap2 = self.shiftReg & (1 << 0) # tap2 is the 23rd bit of shiftReg
            #print('tap1: ' + bin(tap1) + ', tap2: ' + bin(tap2))
            tapsXOR = tap1 ^ tap2 # Compute bitwise XOR of the two taps
            #print('XOR of the taps: ' + bin(tapsXOR))
            out[i] = bitStream[i] ^ tapsXOR # Compute XOR of above result and input bit
            #print('output: ', out[i])

            self.shiftReg = self.shiftReg >> 1 # Shift register moves all bits right one position in prep for the next iteration of scrambling
            self.shiftReg = self.shiftReg | (out[i] << 22) # Shift the output into the first/leftmost spot of the shiftReg
            #print('shifted out: ' + bin(out[i] << 22) + ', shiftReg: ' + bin(self.shiftReg))

            #self.shiftReg = self.shiftReg >> 1 # Shift register moves all bits right one position in prep for the next iteration of scrambling

            #print('shiftReg: ' + bin(self.shiftReg) + '\n')

        return out


    # Count of ones/set-bits in a unsigned binary integer
    @staticmethod
    def _hammingWeight(num):
        binary_rep = np.binary_repr(num) # Binary representation of int as string

        count = 0
        for i in binary_rep:
            if i == "1":
                count += 1
        return count


    # bitStream will be a 1D array.   
    def scramble(self, bitStream):
        out = np.zeros((bitStream.size,), dtype=int) # Create output array for scrambled data same length as input array
        
        if (self.shiftRegLength > 63) | (len(bin(self.tapsBits)) > 63):
            sys.exit('ERROR: cannot operate on integer values greater than 64 bits')
            

        for i in range(out.size):
            extractTaps = self.tapsBits & self.shiftReg # Should extract only the bits in the LFSR which have taps and record them in some binary/int number
            tapsXOR = self._hammingWeight(extractTaps) % 2 # Should give the XOR of all taps. If the parity of 1s in the number is even then result is 0, if parity is odd then result is 1 

            out[i] = bitStream[i] ^ tapsXOR # Compute XOR of above result and input bit

            self.shiftReg = self.shiftReg >> 1 # Shift register moves all bits right one position in prep for the next iteration of scrambling
            self.shiftReg = self.shiftReg | (out[i] << (self.shiftRegLength-1)) # Shift the output into the first/leftmost spot of the shiftReg

        return out
    
