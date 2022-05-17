#!/usr/bin/python3
#
# Plot data after each stage of OFDM modem
#
# Author: Greyson Mitra
# Date: 2021-05-26


import numpy as np
#import argparse
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

import scrambler.Scrambler
import scrambler.Descrambler
import modulation.Mapper
import modulation.Demapper
import ofdm.Modulator
import ofdm.Demodulator

if __name__ == '__main__':


  # The try-except-finally clause allows us to hit ctrl-c on the keyboard to
  # quit and exit gracefully, e.g., by closing open files etc.
  try:
    
    scram = scrambler.Scrambler.Scrambler(2**0+2**5, 23) #Init scrambler object
    descram = scrambler.Descrambler.Descrambler(2**0+2**5, 23)

    mapper = modulation.Mapper.Mapper()
    demapper = modulation.Demapper.Demapper()

    idft_mod_unit = ofdm.Modulator.Modulator(256, 240, 16)
    dft_demod_unit = ofdm.Demodulator.Demodulator(256, 240, 16)

    allOnes = np.ones((480,), dtype=int)

    
    # This is the processing loop
    while True:



      prbs = scram.scramble(allOnes) #give 1D array to scrambler object to scramble... 480 bits so 240 mapped symbols and then 256 IDFT subcarriers
      #print('scrambled bits: %r'%prbs)

      mapMod = mapper.map(prbs)

      #Test IDFT and DFT units with scramblers and modulation mapping
      tformedSignal = idft_mod_unit.idft_mod(mapMod)
      prefixed = idft_mod_unit.add_prefix(tformedSignal, 18)
      trimmed = dft_demod_unit.rm_prefix(prefixed, 18)
      
      untformedSignal = dft_demod_unit.dft_demod(trimmed)
      mapDemod = demapper.demap(untformedSignal)
      prbsSink = descram.descramble(mapDemod)


      #print('Descrambled bits: %r'%prbsSink)

      error = prbs.size - np.count_nonzero(prbsSink) 
      #print('error: ', error)


      
      # This part is not needed other than to slow things down for us humans.
      print('Sleeping...')
      time.sleep(5.0)






      plt.figure(figsize=(14,6))
      plt.figure(1)
      plt.suptitle('Bit stream before and after Scrambler', fontsize=14, fontweight='bold')

      ax0 = plt.subplot(1, 2, 1)
      plt.title('Before')
      #plt.grid(True)
      text = ('Input Array: ' + np.array2string(allOnes, separator=','))
      plt.text(0.01, 0.5, text, ha='left', wrap=True)
      plt.axis('off')
      #ax0.set(xlim=(0, 0.5), ylim=(0, 1))

      ax1 = plt.subplot(1, 2, 2)
      plt.title('After')
      text2 = ('Output Array: ' + np.array2string(prbs, separator=','))
      plt.text(0.01, 0.5, text2, ha='left', wrap=True)
      plt.axis('off')
      #plt.show()

      
      plt.figure(figsize=(8,6))
      plt.figure(2)
      plt.suptitle('QPSK symbols generated from bit stream', fontsize=14, fontweight='bold')
      plt.plot(np.real(mapMod), np.imag(mapMod), '.')
      plt.grid(True)
      plt.vlines(0,-1,1)
      plt.hlines(0,-1,1)
      plt.axis([-1,1, -1,1])
      plt.xlabel('Imag', size='x-large')
      plt.ylabel('Real', size='x-large')

      plt.figure(figsize=(8,6))
      plt.figure(3)
      plt.suptitle('Ouput of IFFT with CP', fontsize=14, fontweight='bold')
      plt.plot(np.abs(prefixed), '.-')
      #plt.plot(np.real(prefixed), np.imag(prefixed), '.')
      plt.grid(True)
      plt.xlabel('Samples', size='x-large')
      plt.ylabel('Magnitude', size='x-large')

      plt.figure(figsize=(8,6))
      plt.figure(4)
      plt.suptitle('Output of FFT with CP removed', fontsize=14, fontweight='bold')
      plt.plot(np.real(untformedSignal), np.imag(untformedSignal), '.')
      plt.grid(True)
      plt.vlines(0,-1,1)
      plt.hlines(0,-1,1)
      plt.axis([-1,1, -1,1])
      plt.xlabel('Imag', size='x-large')
      plt.ylabel('Real', size='x-large')

      plt.figure(figsize=(14,6))
      plt.figure(5)
      plt.suptitle('Scrambled recieved bits and descrambled bits', fontsize=14, fontweight='bold')
      ax3 = plt.subplot(1, 2, 1)
      plt.title('After QPSK demapping')
      text3 = ('Input Array: ' + np.array2string(mapDemod, separator=','))
      plt.text(0.01, 0.5, text3, ha='left', wrap=True)
      plt.axis('off')
      ax4 = plt.subplot(1, 2, 2)
      plt.title('After PRBS sink')
      text4 = ('Output Array: ' + np.array2string(prbsSink, separator=','))
      plt.text(0.01, 0.5, text4, ha='left', wrap=True)
      plt.axis('off')

      plt.show()

      
  except KeyboardInterrupt as e:
    print('\nI caught the keyboard interrupt: ',e)
  finally:
    print('Finally done')

