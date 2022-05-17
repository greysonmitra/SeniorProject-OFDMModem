#!/usr/bin/env python3
import multiprocessing as mp
import threading
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import time

import scrambler.Scrambler
import scrambler.Descrambler
import modulation.Mapper
import modulation.Demapper
import ofdm.Modulator
import ofdm.Demodulator

import os

# From this module:
from sdr import SDR


#Formerly skeleton_prog.py

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
durationInSymbols = 30  #How many OFDM symbols per burst... 60 or so upper limit on my machine
numExtraSymbols = 10 #Some number of extra symbols to aid in padding zeros onto sent data (?) As long as it makes it some multiple of 274
burstPeriod = (ofdmSymLength*(durationInSymbols+numExtraSymbols)) / sampleRate #Small number in seconds..... but should it be durationInSymbols or durationInSymbols+preamble???
#extraInput = np.array([], dtype=int)

# Produce zeros in between bursts. Do for remaining time in cycle after "real" data is sent:
totalSamples = int(burstPeriod * sampleRate)
dataSamples = durationInSymbols*ofdmSymLength
zeroSamples = totalSamples - dataSamples # Should be 1096*2 minus 274*1(???). This is how many zero samples to send.

#480 random ints uniformly distributed between 0 and 2 (2 not inclusive) aka random bits using np.random.randint():
preamble = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0,
       1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
       0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,
       1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
       0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
       1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
       1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0,
       0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
       1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1,
       0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
       1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
       1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0,
       0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
       0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
       0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]) 
mapModPreamble = mapper.map(preamble) # Put the preamble in the time domain and add CP (but don't scramble it without descrambling it cuz obviously its already random enough!)
tfSignalPreamble = idft_mod_unit.idft_mod(mapModPreamble)
preambleCP = idft_mod_unit.add_prefix(tfSignalPreamble, cpLen)

# PROCESS DATA IN INFINITE LOOP:
#lastBurst = np.zeros(preambleCP.size - 1, dtype=int) # Needs to persist through loops, so assign here and init to array of zeros for first burst
#dataSave = np.empty(0)
burstsToGen = 2


def rx_proc(sdr, pipe):
    '''
    Receives and processes data from the SDR object "sdr". Exits when the pipe
    receives a -1.
    '''

    lastBurst = np.zeros(preambleCP.size - 1, dtype=int) # Needs to persist through loops, so assign here and init to array of zeros for first burst
    dataSave = np.empty(0)

    totalErrorBits = 0
    burstCount = 0

    # See SDR.rxburst() in sdr.py for burst mode operation
    try:
        sdr.rxstart() # Start receiving
        while not pipe.poll():
            # --- RECEIVER PROCESSING LOOP ---
            data, start_time = sdr.get() # Get data from the SDR queue
            burstWithZeros = data
            #data = np.asarray(sdr.rxburst()[0])
            #print("received: %r" %data[0][0:1000])

            start1 = time.perf_counter()
            #Should create a growing array of all data coming into the reciever:           
            #dataSave = np.append(dataSave, data)
            #print("\ndataSave size: ",dataSave.size)

            burstWithZeros = np.append(lastBurst,burstWithZeros) #Append samples from the previous burst to prevent overlapping errors in matched filter.
            lastBurst = data[-preambleCP.size + 1:dataSave.size] # Data grab for preventing some matched filter overlap error stuff.
            
            #if dataSave.size > 1: 
            #    lastBurst = dataSave[-preambleCP.size + 1:dataSave.size] # Data grab for preventing some matched filter overlap error stuff.

            #dataSave = data

            stop1 = time.perf_counter()
            dataGrabTime = stop1 - start1
            
            start2 = time.perf_counter()

            #burstWithZeros = np.append(lastBurst,burstWithZeros) #Append samples from the previous burst to prevent overlapping errors in matched filter.

            #Prepare and normalize data for correlation/matched filter:
            corrCopy = burstWithZeros #Copy of the data
            zerosLoc = np.argwhere(burstWithZeros == 0) #Find elements in data equal to zero
            zerosLoc = zerosLoc.flatten()
            epsilon = np.finfo(np.complex128).eps
            for i in range(zerosLoc.size): #Replace zeros in data with really small number aka epsilon
                corrCopy[zerosLoc[i]] = epsilon+epsilon*1j

            normCopy = corrCopy/np.absolute(corrCopy) #Normalize data by each complex numbers magnitude. Makes every complex number's MAGNITUDE is equal to 1
            normPreambleCP = preambleCP/np.absolute(preambleCP) #Normalize preamble as well

            stop2 = time.perf_counter()
            dataPrepFilterTime = stop2-start2

            start3 = time.perf_counter()

            #Matched filter:
            xcorr = signal.oaconvolve(normCopy, np.flip(np.conj(normPreambleCP)), 'valid') #xcorr = np.correlate(normCopy, normPreambleCP) # Smaller signal should go second
            magXcorr = np.abs(xcorr)
            
            #plt.figure(1)
            #plt.plot(magXcorr)
            #plt.plot(np.abs(burstWithZeros))
            #plt.grid(True)
            #plt.show()

            threshold = 175 # No set of data should correlate so well with the preamble that it reaches above this threshold, unless it is the preamble itself!
            indices = np.argwhere(magXcorr > threshold) #All indices where the peaks exceed the threshold
            indices = indices.flatten()
            #print("indices: %r"%indices)
            currSamp = np.amax(magXcorr) #If signal reaches above threshold then this value overwritten. If not, assigns something as peak to avoid crash and continue sending
            for idx in range(indices.size):
                if magXcorr[indices[idx]] > currSamp: #If the current peak sample is larger than the last then record it as new highest peak
                    currSamp = magXcorr[indices[idx]]
                #print("corr points: ", magXcorr[indices[idx]])

            preambleLoc = np.argwhere(magXcorr == currSamp)[0][0]
            #print("PREAMBLE LOC: %r" %preambleLoc)
            #print("curr sample: ", currSamp)

            stop3 = time.perf_counter()
            filterFindSignalTime = stop3-start3
            

            start4 = time.perf_counter()
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

            # Adjust phase on each data sample and prepare for further processing of data:
            removeBurstExtra = removeBurstExtra.flatten()
            for estimationIndex in range(removeBurstExtra.size):
                offsetFreq = np.exp(-1j*2*np.pi*E_0_hat*estimationIndex/totalSubcarriers)
                removeBurstExtra[estimationIndex] = removeBurstExtra[estimationIndex]*offsetFreq

            stop4 = time.perf_counter()
            procDataFreqOffsetTime = stop4-start4

            start5 = time.perf_counter()
            #Continue processing data:
            trimmed = dft_demod_unit.rm_prefix(removeBurstExtra, cpLen)
            untformedSignal = dft_demod_unit.dft_demod(trimmed)
            #print('untformedSignal:', untformedSignal)

            #Channel estimation and equalization:
            untformedSignal = np.reshape(untformedSignal, (untformedSignal.size // usedSubcarriers, usedSubcarriers)) #Divide data into subgroups of 240 samples
            preambleDFT = untformedSignal[0] # aka R_l... used to create estimate
            X_l = mapModPreamble #We know preamble and what it is in the frequency domain (its QPSK syms) so we can just use that here
            H_l_hat = preambleDFT/X_l #Make the estimate only using the DFT of the recieved preamble and the known preamble DFT
            equalizedSignal = (untformedSignal[1:durationInSymbols+1]/H_l_hat).flatten() #Drop preamble since we don't need it anymore

            stop5 = time.perf_counter()
            chanEstEqAndFFTTime = stop5-start5

            start6 = time.perf_counter()
            #Continues processing data:
            mapDemod = demapper.demap(equalizedSignal)
            prbsSink = descram.descramble(mapDemod)
            numBitsError = prbsSink.size - np.count_nonzero(prbsSink)
            totalErrorBits += numBitsError
            BER = (numBitsError/prbsSink.size)*100
            burstCount+=1
            totalBER = (totalErrorBits/(burstCount*prbsSink.size))*100
            print('\nerror bits: ', numBitsError)
            print('BER: ',BER,"%")
            print('TOTAL BER: ',totalBER,'%')
            print('BURST #',burstCount)
            if numBitsError > 10:
                print("OUT: %r"%prbsSink[0:1000])
            
            stop6 = time.perf_counter()
            demodTime = stop6-start6

            overallTime = dataGrabTime+dataPrepFilterTime+filterFindSignalTime+procDataFreqOffsetTime+chanEstEqAndFFTTime+demodTime
            '''
            print("-----------------------------------------------------------------------------------\n"
                  "| TIMING INFO:                        time elapsed (sec)      time % of total     |\n"
                  "| Grab prev burst data:              ",round(dataGrabTime, 15),'      ',round((dataGrabTime/overallTime)*100, 12), "%   |\n"
                  "| Prep data for Matched Filter:      ",round(dataPrepFilterTime, 15),'     ',round((dataPrepFilterTime/overallTime)*100, 12), "%    |\n"
                  "| Matched Filter and Locate Signal:  ",round(filterFindSignalTime, 15),'     ',round((filterFindSignalTime/overallTime)*100, 12), "%    |\n"
                  "| Carrier Recovery and data process: ",round(procDataFreqOffsetTime, 15),'     ',round((procDataFreqOffsetTime/overallTime)*100, 12), "%    |\n"
                  "| Channel Est/Eq and FFT:            ",round(chanEstEqAndFFTTime, 15),'     ',round((chanEstEqAndFFTTime/overallTime)*100, 12), "%    |\n"
                  "| Demod and finish data process:     ",round(demodTime, 15),'     ',round((demodTime/overallTime)*100, 12),"%    |\n"
                  "-----------------------------------------------------------------------------------\n")   
            '''
            #print("Prev burst grab time pctg: ", (totTime1/overallTime)*100,"%")

            #symbolsSent = prefixed.size // ofdmSymLength #How many groups of 274 samples

            #print('Rx PID: ', os.getpid())
    
    except Exception as exception: # Print any exceptions:
        print(type(exception).__name__, "in receive process:")
        print(*exception.args)
        print("Stopping receiver...")
    sdr.rxstop() # Stop receiving (clears the receive queue)

def tx_proc(sdr, pipe):
    '''
    Transmits data from the SDR object "sdr". Exits when the pipe gets a -1.
    '''
    # See SDR.txburst() in sdr.py for burst mode operation
    try:

        
        inputData = np.ones(bitsPerSym*durationInSymbols, dtype=int) #(np.complex64)
        sdr.txstart() # Start transmitting
        while not pipe.poll():
            # --- TRANSMITTER PROCESSING LOOP ---

            symbolsCount = 0

            #if extraInput.size > 0: # Prepend extra data if there is more extra from the last burst
            #    inputData = np.append(extraInput, inputData)

            #if inputData.size >= (bitsPerSym*durationInSymbols): #If the input data has more data than can fit into the symbols we are going to send in this burst, then slice it and store for next burst.
            #    extraInput = inputData[(bitsPerSym*durationInSymbols):inputData.size:1]
            #    inputData = inputData[0:(bitsPerSym*durationInSymbols):1]
            #else: #If input has less than an OFDM symbol's worth of data, just store as extra
            #    extraInput = inputData

            prbs = scram.scramble(inputData) #give 1D array to scrambler object to scramble... 480 bits so 240 mapped symbols and then 256 IDFT subcarriers
            mapMod = mapper.map(prbs)
            tformedSignal = idft_mod_unit.idft_mod(mapMod)
            prefixed = idft_mod_unit.add_prefix(tformedSignal, cpLen)

            prefixed = np.append(preambleCP, prefixed) # Since we do one burst per loop, add preamble (with CP) to each burst
            
            burstWithZeros = np.append(prefixed, np.zeros(zeroSamples,dtype=complex)) # Fill the rest of the burst with zero samples if there are not enough OFDM symbols being transmitted per the sample rate.

            
            #global lastBurst
            #burstWithZeros = np.append(lastBurst, burstWithZeros) # 1st time it'll prepend the zeros initialized above but every subsequent burst will prepend the next line! No need for branching
            #lastBurst = prefixed[-preamble.size + 1:prefixed.size:1] # Data grab so that we can append onto a subsequent burst so that we don't get errors. Want the length of the preamble minus 1. The math here goes that length from the end, to the end.

            #data = 2.*( burstWithZeros-np.min(burstWithZeros) ) / np.ptp(burstWithZeros)-1 # Normalize data between -1 and 1(?)
            data = 4.*burstWithZeros
            rxBufSize = int(1e6//4080*4080)
            data = np.append(data, np.zeros(rxBufSize-data.size))

            #print('data to send: %r' %data[0:1000])
            #print('data to send size: ', data.size)

            
            sdr.send(data) # Send data to the SDR queue
            #sdr.txburst(data)
            print("Transmitted data")

            #print('Tx PID: ', os.getpid())

    except Exception as exception: # Print any exceptions:
        print(type(exception).__name__, "in transmit process:")
        print(*exception.args)
        print("Stopping transmitter...")
    sdr.txstop() # Stop transmitting (finishes transmitting the transmit queue)


class SimultaneousTxRx:
    '''
    Wrapper for running an SDR object with transmit and receive simultaneously.
    This uses the above functions to transmit and receive data through the SDR
    '''
    def __init__(self, rx=True, tx=True, ref='int', serial=None, queuelen=2):
        '''
        rx:       whether to use the receiver
        tx:       whether to user the transmitter
        ref:      synchronization reference to use ('int', 'ext', or 'gps')
        serial:   SDR serial number (None for UHD default)
        quelelen: length of the transmit and receive queues
        '''
        self.rx = rx
        self.tx = tx
        self.sdr = SDR(queuelen) # SDR object
        self.sdr.ref = ref
        self.sdr.serial = serial
        self.pipes = []  # list of pipes
        self.procs = []  # list of processes

    def run(self):
        ''' Runs the SDR until the user hits Enter '''
        # Start a thread that waits for the user to hit enter and then exits:
        thrd = threading.Thread(target=self._wait_func)
        self.start() # Start the SDR processes
        thrd.start() # Start the thread
        thrd.join()  # Wait for the thread to finish

    def start(self):
        ''' Starts the SDR '''
        # Open the SDR object:
        self.sdr.open()
        # Reset some variables:
        self.procs = []
        self.pipes = []
        # Create pipes to communicate with rx_proc and tx_proc:
        if self.tx:
            tx_pipe = mp.Pipe()
            self.pipes.append(tx_pipe[0])
            self.procs.append(
                mp.Process(target=tx_proc, args=(self.sdr, tx_pipe[1])))
        if self.rx:
            rx_pipe = mp.Pipe()
            self.pipes.append(rx_pipe[0])
            self.procs.append(
                mp.Process(target=rx_proc, args=(self.sdr, rx_pipe[1])))
        # Start the processes:
        for proc in self.procs:
            proc.start()

    def stop(self):
        ''' Stops the SDR processes '''
        # Send a message to the pipes telling them to exit:
        for pipe in self.pipes:
            pipe.send(-1)
        # Wait for the processes to exit:
        for proc in self.procs:
            proc.join()
        # Close the SDR:
        self.sdr.close()

    def _wait_func(self):
        ''' Waits for the user to hit enter and then stops the SDR '''
        print("Running. Hit Enter to exit")
        input() # wait for input
        print("Waiting for SDR processes to close")
        self.stop() # Stop the SDR


def cli_launcher():
    ''' command line launcher for the SDR program '''
    import argparse
    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rx',action='store_true', help='run the receiver')
    parser.add_argument('--tx',action='store_true', help='run the transmitter')
    parser.add_argument('--samp_rate', type=float, default=3.84e6)
    parser.add_argument('--txfreq', type=float, required=True)
    parser.add_argument('--rxfreq', type=float, required=True)
    parser.add_argument('--txgain', type=int, default=65)
    parser.add_argument('--rxgain', type=int, default=55)
    parser.add_argument('--rx_antenna', type=str, default='RX2',
      help='receive antenna ("RX2" or "TX/RX" for a B200)')
    args = parser.parse_args()

    if not (args.rx or args.tx):
        print("WARNING: neither --rx or --tx specified")

    # Set up the transmitter and receiver:
    txrx = SimultaneousTxRx(rx=args.rx, tx=args.tx)
    # Configure the SDR object. Start with the transmitter:
    txrx.sdr.txgain = args.txgain # gain, max is 89.8 dB
    txrx.sdr.txfreq = args.txfreq # center frequency
    txrx.sdr.txrate = args.samp_rate    # sample rate
    txrx.sdr.txant = 'TX/RX' # configure transmit antenna
    # Configure the receiver:
    txrx.sdr.rxgain = args.rxgain # gain, max is 76 dB
    txrx.sdr.rxfreq = args.rxfreq # center frequency
    txrx.sdr.rxrate = args.samp_rate #2e6    # sample rate
    txrx.sdr.rxsamp = int(1e6//4080*4080) # receive buffer size
    # Configure receive antenna:
    txrx.sdr.rxant = args.rx_antenna

    # Run the transmitter and receiver:
    txrx.run()

if __name__=='__main__':
    cli_launcher()
    print('Main PID: ', os.getpid())
