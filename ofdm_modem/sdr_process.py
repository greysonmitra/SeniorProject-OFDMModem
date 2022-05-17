import time
import queue
import threading
from enum import IntEnum
import numpy as np
import uhd

class SdrCommand(IntEnum):
    CLOSE = -1
    STOP_TX_CONT = 0
    START_TX_CONT = 1
    TX_BURST = 2
    STOP_RX_CONT = 3
    START_RX_CONT = 4
    RX_BURST = 5
    GET_TX_GAIN = 6
    GET_TX_FREQ = 7
    GET_TX_RATE = 8
    GET_TX_ANT = 9
    SET_TX_GAIN = 10
    SET_TX_FREQ = 11
    SET_TX_RATE = 12
    SET_TX_ANT = 13
    GET_RX_GAIN = 14
    GET_RX_FREQ = 15
    GET_RX_RATE = 16
    GET_RX_ANT = 17
    SET_RX_GAIN = 18
    SET_RX_FREQ = 19
    SET_RX_RATE = 20
    SET_RX_SAMP = 21
    SET_RX_ANT = 22
    GET_SERNUM = 23
    GET_USRP_TIME = 24
    GET_REF_LOCKED = 25
    GET_GPS_TIME = 26
    GET_GPS_GPGGA = 27
    GET_GPS_LOCKED = 28


class SdrProcess:
    def __init__(self, pipe, tx_data_queue, rx_data_queue, txgain, txfreq,
      txrate, rxgain, rxfreq, rxrate, otw, txant=None, rxant=None, rxsamp=4080,
      ref='int', sernum=None, args=''):
        '''
        pipe:         pipe to receive commands from and respond to
        txdata_queue: queue to stream data from
        rxdata_queue: queue to stream data to
        txgain:       initial transmitter amplifier gain in dB
        txfreq:       initial transmitter center frequency in Hz
        txrate:       initial transmitter sample rate in Hz
        txant:        initial transmitter antenna
        rxgain:       initial receiver amplifier gain in dB
        rxfreq:       initial receiver center frequency in Hz
        rxrate:       initial receiver sample rate in Hz
        rxant:        initial receiver antenna
        rxsamp:       intiial number of samples for the receiver buffer
        otw:          over-the-wire format (np.int8 or np.int16)
        ref:          reference source ('int', 'ext', or 'gps')
        sernum:       device serial number
        args:         any additional device arguments
        '''
        # Store some parameters:
        self.pipe = pipe
        self.tx_data_q = tx_data_queue
        self.rx_data_q = rx_data_queue
        self.ref = ref
        # Configure the USRP:
        self.usrp = None    # USRP object
        self.sernum = None  # serial number string
        self.has_gps = None # whether this USRP has a GPS
        self._open_usrp(sernum, args)
        # Set some parameters:
        self.rxbuf = None # receive buffer. Stays None if mode=='tx'
        self._buf_dtype = np.uint16 if otw==np.int8 else np.uint32 # for rxbuf
        # buf_dtype needs to be twice as large as otw. Decoding is done by the
        # Frequency and rate variables (set by set_freq() and set_rate()):
        self.txgain = -1     # actual gain
        self.txfreq = -1     # actual frequency
        self.txrate = -1     # actual sample rate
        self.txreq_rate = -1 # requested rate
        self.txant = ''      # transmitter antenna
        self.rxgain = -1     # actual gain
        self.rxfreq = -1     # actual frequency
        self.rxrate = -1     # actual sample rate
        self.rxant = ''      # receiver antenna
        self.rxreq_rate = -1 # requested rate
        # main process
        self.set_txgain(txgain) # sets gain
        self.set_txfreq(txfreq) # sets freq
        self.set_txrate(txrate) # sets sample rate and the USRP's internal clock
        self.set_rxgain(rxgain) # sets gain
        self.set_rxfreq(rxfreq) # sets freq
        self.set_rxrate(rxrate) # sets sample rate and the USRP's internal clock
        self.set_rxsamp(rxsamp) # initializes self.rxbuf
        # Configure antennas:
        if txant is not None:
            self.set_txant(txant)
        else:
            self.txant = self.usrp.get_tx_antenna(0)
        if rxant is not None:
            self.set_rxant(rxant)
        else:
            self.rxant = self.usrp.get_rx_antenna(0)

        # Start SDR thread (SDR should be asynchronous from IO, which the main
        # thread handles.
        self._tx_interface_q = queue.Queue()
        self._rx_interface_q = queue.Queue()
        self._tx_streaming_q = queue.Queue(maxsize=1)
        self._rx_streaming_q = queue.Queue(maxsize=1)
        # Get streamargs:
        if otw==np.int8:
            streamargs = uhd.usrp.StreamArgs('sc8','sc8')
            data_dtype = np.uint16 # output buffer internal dtype
        else:
            streamargs = uhd.usrp.StreamArgs('sc16','sc16')
            data_dtype = np.uint32 # output buffer internal dtype
        streamargs.channels = [0]
        # Start SDR threads:
        self._tx_thread = threading.Thread(target=self._tx_sdr, args=(streamargs,))
        self._tx_thread.start()
        self._rx_thread = threading.Thread(target=self._rx_sdr, args=(streamargs,))
        self._rx_thread.start()

        # Return the serial number to let main know initialization is done:
        self.pipe.send((SdrCommand.GET_SERNUM, self.sernum))

    def run(self):
        ''' Starts receiving commands from the main process '''
        self._io_loop()


    # Main loop:
    def _io_loop(self):
        ''' Handle incoming commands from the main process '''
        while True:
            # Get data from main:
            ret_val = None # overridden if a command produces a return value
            command, val = self.pipe.recv()
            # Commands:
            if command==SdrCommand.CLOSE:
                self._tx_interface_q.put((command,val))
                self._rx_interface_q.put((command,val))
                break # break from the loop
            elif command==SdrCommand.STOP_TX_CONT:
                self._tx_interface_q.put((command,val))
            elif command==SdrCommand.START_TX_CONT:
                self._tx_interface_q.put((command,val))
            elif command==SdrCommand.TX_BURST:
                self._tx_interface_q.put((command,val))
            elif command==SdrCommand.STOP_RX_CONT:
                self._rx_interface_q.put((command,val))
            elif command==SdrCommand.START_RX_CONT:
                self._rx_interface_q.put((command,val))
            elif command==SdrCommand.RX_BURST:
                self._rx_interface_q.put((command,val))
            # Set commands:
            elif command==SdrCommand.SET_TX_GAIN:
                self.set_txgain(val)
                ret_val = self.txgain
            elif command==SdrCommand.SET_TX_FREQ:
                self.set_txfreq(val)
                ret_val = self.txfreq
            elif command==SdrCommand.SET_TX_RATE:
                self.set_txrate(val)
                ret_val = self.txrate
            elif command==SdrCommand.SET_TX_ANT:
                self.set_txant(val)
                ret_val = self.txant
            elif command==SdrCommand.SET_RX_GAIN:
                self.set_rxgain(val)
                ret_val = self.rxgain
            elif command==SdrCommand.SET_RX_FREQ:
                self.set_rxfreq(val)
                ret_val = self.rxfreq
            elif command==SdrCommand.SET_RX_RATE:
                self.set_rxrate(val)
                ret_val = self.rxrate
            elif command==SdrCommand.SET_RX_ANT:
                self.set_rxant(val)
                ret_val = self.rxant
            elif command==SdrCommand.SET_RX_SAMP:
                self.set_rxsamp(val)
            # Get commands:
            elif command==SdrCommand.GET_TX_GAIN:
                ret_val = self.txgain
            elif command==SdrCommand.GET_TX_FREQ:
                ret_val = self.txfreq
            elif command==SdrCommand.GET_TX_RATE:
                ret_val = self.txrate
            elif command==SdrCommand.GET_TX_ANT:
                ret_val = self.txant
            elif command==SdrCommand.GET_RX_GAIN:
                ret_val = self.rxgain
            elif command==SdrCommand.GET_RX_FREQ:
                ret_val = self.rxfreq
            elif command==SdrCommand.GET_RX_ANT:
                ret_val = self.rxant
            elif command==SdrCommand.GET_RX_RATE:
                ret_val = self.rxrate
            elif command==SdrCommand.GET_SERNUM:
                ret_val = self.sernum
            elif command==SdrCommand.GET_USRP_TIME:
                ret_val = self.usrp_time()
            elif command==SdrCommand.GET_REF_LOCKED:
                ret_val = self.ref_locked()
            elif command==SdrCommand.GET_GPS_TIME:
                ret_val = self.gps_time()
            elif command==SdrCommand.GET_GPS_GPGGA:
                ret_val = self.gps_gpgga()
            elif command==SdrCommand.GET_GPS_LOCKED:
                ret_val = self.gps_locked()
            # Respond to main:
            self.pipe.send( (command, ret_val) )
        self.pipe.send( (command, ret_val) )
        self._tx_thread.join()
        self._rx_thread.join()


    # Sensor functions:
    def usrp_time(self):
        ''' Returns the USRP internal clock time as a float '''
        return self.usrp.get_time_now().get_real_secs()
    def ref_locked(self):
        ''' Returns whether the reference is locked '''
        return self.usrp.get_mboard_sensor('ref_locked').to_bool()
    def gps_time(self):
        ''' Blocks until the next PPS and then returns the GPS time. '''
        if self.has_gps:
            return self.usrp.get_mboard_sensor('gps_time').to_int()
        else:
            return -1
    def gps_gpgga(self):
        ''' Returns a GPGGA string from the GPS '''
        if self.has_gps:
            return self.usrp.get_mboard_sensor('gps_gpgga').value
        else: # Return s sample GPGGA that indicates that it's not valid
            return 'GPGGA,000000.00,0000.0000,N,00000.0000,E,0,99,1.0,0.0,M,0.0,M,,*5C'
    def gps_locked(self):
        ''' Return whether the GPS reference is locked '''
        if self.has_gps:
            return self.usrp.get_mboard_sensor('gps_locked').to_bool()
        else:
            return False


    # USRP configuration functions:
    def set_txgain(self, gain):
        ''' gain = target transmit amplifier gain in dB '''
        self.usrp.set_tx_gain(gain,0)
        self.txgain = self.usrp.get_tx_gain(0)
    def set_rxgain(self, gain):
        ''' gain = target receive amplifier gain in dB '''
        self.usrp.set_rx_gain(gain,0)
        self.rxgain = self.usrp.get_rx_gain(0)
    def set_txfreq(self, freq):
        ''' freq = target transmit center frequency in Hz '''
        self.usrp.set_tx_freq(uhd.types.TuneRequest(freq))
        self.freq = self.usrp.get_tx_freq(0)
    def set_rxfreq(self, freq):
        ''' freq = target receive center frequency in Hz '''
        self.usrp.set_rx_freq(uhd.types.TuneRequest(freq))
        self.freq = self.usrp.get_rx_freq(0)
    def set_txrate(self, samp_rate):
        ''' rate = target transmit sample rate in Hz '''
        if samp_rate != self.txreq_rate: # Make sure the rate is actually changing
            self.usrp.set_tx_rate(samp_rate,0)
            self.rate = self.usrp.get_tx_rate(0)
            self._sync_to_clock() # rate changes can loose clock sync
            self.txreq_rate = samp_rate
    def set_rxrate(self, samp_rate):
        ''' rate = target receive sample rate in Hz '''
        if samp_rate != self.rxreq_rate: # Make sure the rate is actually changing
            self.usrp.set_rx_rate(samp_rate,0)
            self.rate = self.usrp.get_rx_rate(0)
            self._sync_to_clock() # rate changes can loose clock sync
            self.rxreq_rate = samp_rate
    def set_txant(self, antenna):
        ''' Set the transmitter antenna '''
        self.usrp.set_tx_antenna(antenna)
        self.txant = self.usrp.get_tx_antenna(0)
    def set_rxant(self, antenna):
        ''' Set the receiver antenna '''
        self.usrp.set_rx_antenna(antenna)
        self.rxant = self.usrp.get_rx_antenna(0)
    def set_rxsamp(self, rxsamp):
        ''' rxsamp = receive buffer size '''
        self.rxbuf = np.empty(int(rxsamp), dtype=self._buf_dtype)

    # SDR interface functions:
    def _tx_sdr(self, streamargs):
        ''' SDR thread for transmitters '''
        # Create a TX streamer with given arguments:
        tx_streamer = self.usrp.get_tx_stream(streamargs)
        # TX metadata:
        metadata = uhd.types.TXMetadata()
        while True:
            instruction, val = self._tx_interface_q.get()
            if instruction==SdrCommand.CLOSE:
                break
            if instruction==SdrCommand.START_TX_CONT:
                self._tx_continuous(tx_streamer) # start continuous transmit
            if instruction==SdrCommand.TX_BURST: # transmit once
                # Get timing instructions:
                metadata.has_time_spec, start_time = val
                metadata.time_spec = uhd.types.TimeSpec(*start_time)
                metadata.start_of_burst=True
                metadata.end_of_burst=True
                # Send data:
                tx_streamer.send(self.tx_data_q.get(), metadata, timeout=-1)
    def _rx_sdr(self, streamargs):
        ''' SDR thread for receivers '''
        # Create an RX streamer with given arguments:
        rx_streamer = self.usrp.get_rx_stream(streamargs)
        # Burst command:
        burst_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        # Data output from the stream:
        metadata = uhd.types.RXMetadata() # received signal metadata
        while True:
            instruction, val = self._rx_interface_q.get()
            if instruction==SdrCommand.CLOSE:
                break
            if instruction==SdrCommand.START_RX_CONT:
                self._rx_continuous(rx_streamer) # start continuous receive
            if instruction==SdrCommand.RX_BURST: # receive once
                burst_cmd.num_samps = self.rxbuf.size # configure num samps
                # Get timing instructions:
                burst_cmd.stream_now = not val[0]
                burst_cmd.time_spec = uhd.types.TimeSpec(*val[1])
                # Stream data:
                rx_streamer.issue_stream_cmd(burst_cmd)
                rx_streamer.recv(self.rxbuf, metadata, timeout=-1)
                # Send to data queue:
                if metadata.error_code.name=='late': # If time was too early
                    print('Got receive start time in the past')
                self.rx_data_q.put((self.rxbuf, (metadata.time_spec.get_real_secs(),
                    metadata.time_spec.get_frac_secs())), block=False)

    def _tx_continuous(self, tx_streamer):
        ''' Streams transmit data until given a stop command '''
        # TX metadata:
        metadata = uhd.types.TXMetadata()
        metadata.has_time_spec=False
        metadata.start_of_burst=True
        metadata.end_of_burst=False
        # Start the queue transfer thread:
        data_transfer_thrd = threading.Thread(target=self._tx_queue_transfer)
        data_transfer_thrd.start()
        # Start streaming transmit data:
        last_data = np.zeros(2040, dtype=self._buf_dtype)
        queue_cmd = SdrCommand.START_TX_CONT
        while queue_cmd!=SdrCommand.STOP_TX_CONT:
            while self._tx_interface_q.empty():
                try:
                    tx_data = self._tx_streaming_q.get(timeout=1)
                    tx_streamer.send(tx_data, metadata, timeout=-1)
                except queue.Empty:
                    pass
                metadata.start_of_burst=False
            queue_cmd = self._tx_interface_q.get()[0]
        # Closing SDR
        while not self._tx_streaming_q.empty(): # use everything else in the data queue
            tx_streamer.send(self._tx_streaming_q.get(), metadata, timeout=-1)
        # End of burst:
        metadata.end_of_burst=True
        tx_streamer.send(last_data, metadata, timeout=-1)
        # Wait for queue to exit:
        data_transfer_thrd.join()
    def _rx_continuous(self, rx_streamer):
        ''' Streams receive data until given a stop command '''
        # RX metadata:
        metadata = uhd.types.RXMetadata()
        # Commands to start and stop streams:
        start_cmd = uhd.types.StreamCMD(
            uhd.types.StreamMode(uhd.types.StreamMode.start_cont)
        ) # start continuous stream
        stop_cmd = uhd.types.StreamCMD(
            uhd.types.StreamMode(uhd.types.StreamMode.stop_cont)
        ) # stop continuous stream
        # Start the queue transfer thread:
        data_transfer_thrd = threading.Thread(target=self._rx_queue_transfer)
        data_transfer_thrd.start()
        # Start streaming data:
        queue_cmd = SdrCommand.START_RX_CONT
        rx_streamer.issue_stream_cmd(start_cmd)
        while queue_cmd != SdrCommand.STOP_RX_CONT:
            while self._rx_interface_q.empty():
                rx_streamer.recv(self.rxbuf, metadata, timeout=-1) # get data
                # Send data and time to data queue:
                self._rx_streaming_q.put((self.rxbuf,(metadata.time_spec.get_full_secs(),
                    metadata.time_spec.get_frac_secs())), block=False)
            queue_cmd = self._rx_interface_q.get()[0]
        # Stop streaming data
        rx_streamer.issue_stream_cmd(stop_cmd) # stop stream
        self._rx_streaming_q.put(int(0)) # stops the queue transfer thread
        data_transfer_thrd.join()
    def _tx_queue_transfer(self):
        '''
        Transfers data from the inter-process data queue to the streaming queue
        in this thread
        '''
        while True:
            val = self.tx_data_q.get()
            if isinstance(val,int):
                break
            else:
                self._tx_streaming_q.put(val)
    def _rx_queue_transfer(self):
        '''
        Transfers data from the streaming queue in this thread to the
        inter-process data queue
        '''
        while True:
            val = self._rx_streaming_q.get()
            self.rx_data_q.put(val)
            if isinstance(val,int):
                break

    # Misc functions:
    def _sync_to_clock(self):
        '''
        Set the USRP clock to UTC. If we're using a GPSDO this will set the
        USRP clock based on GPS time. Otherwise this will attempt to set the
        USRP time based on the host computer's internal clock.
        '''
        if self.ref=='gps':
            time_func = self.gps_time
        else:
            time_func = lambda: int(time.time())
        # Set USRP device time to UTC time:
        print('Synchronizing USRP clock to UTC...')
        self.usrp.set_time_next_pps(uhd.types.TimeSpec(time_func()+1),0)
        time.sleep(2)
        self.usrp.set_time_next_pps(uhd.types.TimeSpec(time_func()+1),0)
        if time_func() == self.usrp.get_time_last_pps().get_real_secs():
            print('USRP clock synchronized')

    def _open_usrp(self, sernum, args):
        '''
        Opens a USRP, configures the timebase and clock sources, and gets the
        device serial number,
        '''
        if sernum is not None: # if a serial number is given
            args += ' serial='+sernum # add it to the argument string

        self.usrp = uhd.usrp.MultiUSRP(args) # initialize device class
        # Check whether we have a GPS:
        self.has_gps = 'gps_gpgga' in self.usrp.get_mboard_sensor_names()
        # Set up clock and time soruces:
        if self.ref=='gps': # use the GPSDO
            if self.has_gps: # if we have a GPSDO
                self.usrp.set_sync_source('gpsdo','gpsdo',0)
            else: # if we don't have a GPSDO
                print('GPS not detected. Using internal references')
                ref = 'int'
        if self.ref=='int': # internal reference
            self.usrp.set_sync_source('internal','internal',0)
        elif self.ref=='ext': # external references
            self.usrp.set_sync_source('external','external',0)
        # Ger the USRP serial number:
        self.sernum = self.usrp.get_usrp_rx_info()['mboard_serial']
        #self.sernum = self.usrp.get_usrp_tx_info()['mboard_serial']

    def _wait_for_lock(self, ref):
        '''
        Waits for a reference lock. Also waits for a GPS lock if we're using a
        GPS reference.
        '''
        # If we're using the GPSDO, wait for that to lock:
        if self.ref=='gps':
            print('Waiting for GPS lock...')
            i=0
            while not self.gps_locked():
                if i>0:
                    print('.', end='', flush=True)
                else:
                    print('\r   \r', end='', flush=True)
                i = (i+1)%4
                time.sleep(1)
            print('\rGPS locked')
        # Next wait for a ref lock:
        print('Waiting for ref lock...')
        i=0
        while not self.ref_locked():
            if i>0:
                print('.', end='', flush=True)
            else:
                print('\r   \r', end='', flush=True)
            i = (i+1)%4
            time.sleep(1)
        print('\rRef locked')
        # Sync USRP clock to UTC:
        self._sync_to_clock()


# Simple wrapper for SdrProcess:
def sdr_process(*args, **kwargs):
    ''' Starts the SdrProcess class and closes it when it stops running '''
    sdr_proc = SdrProcess(*args, **kwargs)
    sdr_proc.run()
