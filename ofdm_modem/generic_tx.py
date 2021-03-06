#!/usr/bin/python3
#
# Generic streaming transmitter template.
#
# Author: SLT
# Date: 20210505
#

import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

import scrambler.Scrambler
import scrambler.Descrambler
import modulation.Mapper
import modulation.Demapper
import ofdm.Modulator
import ofdm.Demodulator

if __name__ == '__main__':

  nProc = 1000 # Number of items to generate per processing loop iteration

  # Parse command line inputs
  parser = argparse.ArgumentParser(description='Base OFDM modulator code')
  parser.add_argument('-v','--verbose',action='store_true',help='Print verbose output to stdout')
  parser.add_argument('-f','--filename')
  
  parser.add_argument('-t','--threshold', default=0.6)
  parser.add_argument('-s','--symsPerBurst', default=2)
  args = parser.parse_args()
  print('args: ',args)
  verbose = args.verbose
  writeFile = False
  if args.filename != None:
    writeFile = True

  # Open output file if requested
  oFile = None
  if writeFile:
    oFile = open(args.filename,'wb')

  # print('SLT DEVELOPMENT NOTES: Probably want to get rid of "import time"')
  # The try-except-finally clause allows us to hit ctrl-c on the keyboard to
  # quit and exit gracefully, e.g., by closing open files etc.
  try:
    

    tapsBits = 2**0+2**5
    scramSize = 23 #or size of the shift register

    totalSubcarriers = 256
    usedSubcarriers = 240
    emptySubcarriers = totalSubcarriers - usedSubcarriers # Should be 256
    cpLen = 18
    ofdmSymLength = totalSubcarriers + cpLen # Should be 274
    bitsPerSym = usedSubcarriers*2 # Should be 480 

    #Init objects of various modules of OFDM modem:
    scram = scrambler.Scrambler.Scrambler(tapsBits, scramSize) 
    descram = scrambler.Descrambler.Descrambler(tapsBits, scramSize)
    mapper = modulation.Mapper.Mapper()
    demapper = modulation.Demapper.Demapper()
    idft_mod_unit = ofdm.Modulator.Modulator(totalSubcarriers, usedSubcarriers, emptySubcarriers)
    dft_demod_unit = ofdm.Demodulator.Demodulator(totalSubcarriers, usedSubcarriers, emptySubcarriers)

    # Defining sample rate and timing for burst-y Tx:
    subcarrierSpacing = 15000 #15 kHz for subcarrier spacing in freq domain
    sampleRate = totalSubcarriers * subcarrierSpacing #15K * 256 = 3.84 MHz aka 3,840,000 samples per second
    burstPeriod = .0005708333333333333 # .00028541666666666666*2 seconds(?)
    durationInSymbols = args.symsPerBurst #How many OFDM symbols per burst
    extraInput = np.array([], dtype=int)
    
    # Produce zeros in between bursts. Do for remaining time in cycle after "real" data is sent:
    totalSamples = int(burstPeriod * sampleRate)
    dataSamples = durationInSymbols*ofdmSymLength
    zeroSamples = totalSamples - dataSamples # Should be 1096*2 minus 274*2. This is how many zero samples to send.

    preamble = np.random.randint(0, 2, bitsPerSym, dtype=int) # dtype=np.int8  #480 random ints uniformly distributed between 0 and 2 (2 not inclusive) aka random bits
    mapModPreamble = mapper.map(preamble) # Put the preamble in the time domain and add CP (but don't scramble it without descrambling it cuz obviously its already random enough!)
    tfSignalPreamble = idft_mod_unit.idft_mod(mapModPreamble)
    preambleCP = idft_mod_unit.add_prefix(tfSignalPreamble, cpLen)


    # PROCESS DATA IN INFINITE LOOP:
    burstCount = 0
    lastBurst = np.zeros(preamble.size - 1, dtype=int) #None # Needs to persist through loops, so assign here and init to array of zeros for first burst
    burstsToGen = 1
    while True:
      symbolsCount = 0
      inputData = np.ones(bitsPerSym*durationInSymbols, dtype=int)
      #inputData = np.ones(743, dtype=int)

      # BURST LOOP... send one burst:
      while burstCount < burstsToGen:

#==============================================================================================================================================================
        # Tx
#==============================================================================================================================================================
        if extraInput.size > 0: # Prepend extra data if there is more extra from the last burst
          inputData = np.append(extraInput, inputData)

        if inputData.size >= (bitsPerSym*durationInSymbols): #If the input data has more data than can fit into the symbols we are going to send in this burst, then slice it and store for next burst.
          extraInput = inputData[(bitsPerSym*durationInSymbols):inputData.size:1]
          inputData = inputData[0:(bitsPerSym*durationInSymbols):1]
        else: #If input has less than an OFDM symbol's worth of data, just store as extra
          extraInput = inputData

        prbs = scram.scramble(inputData) #give 1D array to scrambler object to scramble... 480 bits so 240 mapped symbols and then 256 IDFT subcarriers
        mapMod = mapper.map(prbs)
        tformedSignal = idft_mod_unit.idft_mod(mapMod)
        #print('inv tformed signal: %r' %tformedSignal)
        prefixed = idft_mod_unit.add_prefix(tformedSignal, cpLen)

        prefixed = np.append(preambleCP, prefixed) # Since we do one burst per loop, add preamble (with CP) to each burst

        burstWithZeros = np.append(prefixed, np.zeros(zeroSamples,dtype=complex)) # Fill the rest of the burst with zero samples if there are not enough OFDM symbols being transmitted per the sample rate.

        """ lastBurst = prefixed[-preamble.size + 1:prefixed.size:1]  Data grab so that we can append onto a subsequent burst so that we don't get errors. Want the length of the preamble minus 1. The math here goes that length from the end, to the end.
        if burstCount == 0: # On the first burst, add preamble length amount of zeros before matched filter in order to prevent issue of overlap(?)
          filterZeros = np.zeros(preamble.size - 1, dtype=int)
          burstWithZeros = np.append(filterZeros, burstWithZeros)
        else: # If not the first burst, then prepend part of the last burst to prevent overlap(?)
          burstWithZeros = np.append(lastBurst, burstWithZeros) """

        burstWithZeros = np.append(lastBurst, burstWithZeros) # 1st time it'll prepend the zeros initialized above but every subsequent burst will prepend the next line! No need for branching
        lastBurst = prefixed[-preamble.size + 1:prefixed.size:1] # Data grab so that we can append onto a subsequent burst so that we don't get errors. Want the length of the preamble minus 1. The math here goes that length from the end, to the end.

        # Setup for freq offset on each data sample
        E_0 = .1
        for index in range(burstWithZeros.size):
          setFreq = np.exp(1j*2*np.pi*E_0*index/totalSubcarriers)
          burstWithZeros[index] = burstWithZeros[index]*setFreq

#==============================================================================================================================================================
        #Rx
#==============================================================================================================================================================
        #Matched filter:
        xcorr = np.correlate(burstWithZeros, preambleCP) # Smaller signal should go second
        magXcorr = np.abs(xcorr)


        threshold = args.threshold # No set of data should correlate so well with the preamble that it reaches above this threshold.
        for i in range(magXcorr.size):
          #fm = np.square(magXcorr)
          if magXcorr[i] > threshold:
            #print('fm: ', fm[i])
            #print('magnitude: ', magXcorr[i])
            preambleLoc = i

            # Process data since it is the signal we are looking for:
            dataAmount = (durationInSymbols+1)*ofdmSymLength # The number of data samples in the burst including the preamble
            removeBurstExtra = burstWithZeros[preambleLoc:preambleLoc+dataAmount] #Will slice the array right at the preamble including the preamble's CP, and up to the end of the number of symbols. Don't even need the 1 interval because that is default!

            # Carrier recovery/freq offset estimation:
            removeBurstExtra = np.reshape(removeBurstExtra, (removeBurstExtra.size // ofdmSymLength, ofdmSymLength)) #Divide data into subgroups of 274 samples each. Also, extra stuff on burst may not be zeros now due to noise, but can still remove it.
            totalSum = 0
            for j in range(durationInSymbols+1): #Sum of all symbols
              firstSum = 0
              for k in range(cpLen): #Sum of all samples in one symbol
                firstSum += removeBurstExtra[j][k]*np.conj(removeBurstExtra[j][k+totalSubcarriers])
              totalSum += firstSum #The rectangular form of estimation sum term

            E_0_hat = np.angle(totalSum)/(-2*np.pi) #We just want to angle of the estimation sum term (just discard magnitude of sum term) and then we want to multiply it by this -2pi term.
            print('E_0_hat: ', E_0_hat) 

            # Adjust phase on each data sample and prepare for further processing of data:
            removeBurstExtra = removeBurstExtra.flatten()
            for estimationIndex in range(removeBurstExtra.size):
              offsetFreq = np.exp(-1j*2*np.pi*E_0_hat*estimationIndex/totalSubcarriers)
              removeBurstExtra[estimationIndex] = removeBurstExtra[estimationIndex]*offsetFreq

            #Continue processing data:
            trimmed = dft_demod_unit.rm_prefix(removeBurstExtra, cpLen)
            untformedSignal = dft_demod_unit.dft_demod(trimmed)
            #print('untformedSignal:', untformedSignal)

            #Channel estimation and equalization:
            untformedSignal = np.reshape(untformedSignal, (untformedSignal.size // usedSubcarriers, usedSubcarriers)) #Divide data into subgroups of 240 samples
            preambleDFT = untformedSignal[0] # aka R_l... used to create estimate
            X_l = mapModPreamble #We know preamble and what it is in the frequency domain (its QPSK syms) so we can just use that here
            #print('preamble DFT (should have no empty subcarriers):', preambleDFT)
            H_l_hat = preambleDFT/X_l #Make the estimate only using the DFT of the recieved preamble and the known preamble DFT
            equalizedSignal = (untformedSignal[1:durationInSymbols+1]/H_l_hat).flatten() #Just drop preamble since we don't need it anymore
            
            #Continues processing data:
            mapDemod = demapper.demap(equalizedSignal)
            #noPreamble = mapDemod[bitsPerSym:mapDemod.size:1]
            prbsSink = descram.descramble(mapDemod)

        symbolsSent = prefixed.size // ofdmSymLength #How many groups of 274 samples

        plt.figure(1)
        plt.suptitle('Cross correlation of signal and preamble', fontsize=14, fontweight='bold')
        plt.grid(True)
        plt.plot(xcorr)
        plt.figure(2)
        plt.plot(magXcorr)
        plt.grid(True)
         #plt.show()

        symbolsCount += symbolsSent  


        print('Descrambled bits: %r'%prbsSink[0:1000:1])
        print('Num bits descrambled: %r'%prbsSink.size)

        #error = prbs.size - np.count_nonzero(prbsSink) 
        #print('error: ', error)

        burstCount += 1

      time.sleep(5.0)
      #flag = True
      #while flag:
      #  willCont = input('Press Enter/Return to continue\n')
      #  if willCont == '':
      #    flag = False





        
      '''
      if writeFile:
        oFile.write(bits)
      if verbose:
        print('bits: %r'%bits)
      '''
      
    # This part is not needed other than to slow things down for us humans.
      #print('Sleeping...')
      #time.sleep(5.0)
  except KeyboardInterrupt as e:
    print('\n' + 'I caught the keyboard interrupt: ',e)
  finally:
    print('Finally done')
    if writeFile:
      if not oFile.closed:
        print('Need to close the file')
        oFile.close()
