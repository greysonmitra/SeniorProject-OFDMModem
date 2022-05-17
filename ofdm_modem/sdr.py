import time
import multiprocessing as mp
import threading
import numpy as np

from sdr_process import SdrCommand, sdr_process

class SDR:
    '''
    This class runs an SDR in a parallel process using SdrProcess. This can be
    used to run SDRs in one process while simultaneously generating or
    processing data in a parallel process.

    Parameters:
    txgain: Transmit gain setting in dB (default 40)
    txfreq: Transmit radio frequency setting in Hz (default 2.45e9)
    txrate: Transmit sample rate setting in Hz (default 1e6)
    txant:  Transmit antenna to use ("TX/RX" for a B200)
    rxgain: Receive gain setting in dB (default 40)
    rxfreq: Receive radio frequency setting in Hz (default 2.45e9)
    rxrate: Receive sample rate setting in Hz (default 1e6)
    rxant:  Receive antenna to use ("RX2" or "TX/RX" for a B200)
    rxsamp: Number of samples per data queue item (default 4080). For
            continuous streaming. This should be a multiple of the device
            streamer buffer size or samples will be dropped. On a B200 this
            is 2040 if otw=np.int16 or 4080 if otw=np.int8.
    args:   Device arguments (default "")
    serial: Device serial number (updated or set by open())
    ref:    Device PPS and 10 MHz references ("int", "ext", or "gps")
             "int": internal references
             "ext": external PPS and REF inputs
             "gps": lock to GPS reference
    otw:    Over-the-wire format (np.int16 or np.int8) (default np.int16)
    mode:   "tx" for transmit and "rx" for receive
    is_open:    whether the SDR session is open
    tx_running: whether the SDR is transmitting in continuous mode
    rx_running: whether the SDR is receiving in continuous mode

    Methods:
    open:       Open a USRP session through SdrProcess in a separate process.
                Changes to parameters after open() require a setter function
                or update_params().
    close:      Closes the USRP session. Must be run for Python to exit
                correctly.
    txstart:    Starts continuous transmit
    txstop:     Stops continuous transmit
    rxstart:    Starts continuous receive
    rxstop:     Stops continuous receive
    get:        Returns a NumPy array from continuous receive. Must be
                called between start() and stop().
    send:       Sends a NumPy array for continuous transmit. Must be called
                between start() and stop().
    txburst:    Transmits one NumPy array. Start time is an optional argument
    rxburst:    Records and returns one NumPy array with size rxsamp. Start time
                is an optional argument. The default is to start immediately
    time:       Returns the current USRP device time
    gps_time:   Returns the current GPS time
    gps_loc:    Returns a dictionary with GPS location, time, and metadata
    gps_locked: Returns a boolean for for the GPS lock status
    get_serial: Returns the USRP serial number
    txbacklog:  Returns the number of items in the transmit data queue. If this
                is growing during receive mode, data is being produced faster
                than it's being processed
    rxbacklog:  Returns the number of items in the receive data queue

    get_txgain: Returns the actual USRP transmit amplifier gain
    get_txfreq: Returns the actual USRP transmit center frequency
    get_txrate: Returns the actual USRP transmit sample rate
    get_txant:  Returns the USRP transmit antenna
    set_txgain: Sets the USRP transmit amplifier gain
    set_txfreq: Sets the USRP transmit center frequency
    set_txrate: Sets the USRP transmit sample rate
    set_txant:  Sets the USRP transmit antenna

    get_rxgain: Returns the actual USRP receive amplifier gain
    get_rxfreq: Returns the actual USRP receive center frequency
    get_rxrate: Returns the actual USRP receive sample rate
    get_rxant:  Returns the USRP receive antenna
    set_rxgain: Sets the USRP receive amplifier gain
    set_rxfreq: Sets the USRP receive center frequency
    set_rxrate: Sets the USRP receive sample rate
    set_rxant:  Sets the USRP receive antenna
    set_rxsamp: Sets the the number of samples for a receiver to collect
    update_params: Updates the USRP params based on gain, freq, and rate
    '''
    def __init__(self, queuelen=2):
        '''
        mode:     'tx' of 'rx' for transmit or receive
        queuelen: max number of items in the transmit and receive data queues.
                  When the queue is full the send() method will block until a
                  slot is available.
        '''
        # Some parameters for reception:
        self.txgain=40
        self.txfreq=2.45e9
        self.txrate=1e6
        self.txant=None
        self.rxgain=40
        self.rxfreq=2.45e9
        self.rxrate=1e6
        self.rxsamp = 4080
        self.rxant=None
        self.args = ''
        self.serial = None # updated by open()
        self.ref = 'int'
        self.otw = np.int16 # sets self._otw, self._q_dtype, and self._data_max

        # Multiprocessing things:
        self._pipe, self._procpipe = mp.Pipe() # communicates with SDR process
        self._pipe_lock = threading.Lock() # thread lock for the pipe (keeps this thread-safe)
        self._tx_data_q = mp.Queue(queuelen) # transfers SDR data to the SDR
        self._rx_data_q = mp.Queue(queuelen) # transfers SDR data from the SDR
        self._proc = None # SDR process object
        self._is_open = False # whether USRP session is open
        self._tx_running = False # whether continuous transmit is running
        self._rx_running = False # whether continuous receive is running

        # Function that returns the number of items in the queue:
        self.txbacklog = self._tx_data_q.qsize
        self.rxbacklog = self._rx_data_q.qsize

    # Properties:
    @property
    def otw(self):
        return self._otw
    @otw.setter
    def otw(self, val):
        if val==np.int8 or val==np.int16:
            self._otw = val
            if val==np.int8:
                self._data_max = 127 # 8-bit signed ADC
                self._q_dtype = np.uint16
            else:
                self._data_max = 32767 # 12-bit signed ADC
                self._q_dtype = np.uint32
        else:
            raise ValueError('SDR.otw must be np.int8 or np.int16')
    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, val):
        if val=='rx' or val=='tx':
            self._mode = val
        else:
            raise ValueError('SDR.mode must be "tx" or "rx"')
    @property
    def is_open(self):
        return self._is_open
    @property
    def tx_running(self):
        return self._tx_running
    @property
    def rx_running(self):
        return self._rx_running

    # Open/close functions:
    def open(self):
        ''' Opens a USRP session in a separate process '''
        if not self._is_open:
            self._proc = mp.Process(target=sdr_process,
              args=(self._procpipe, self._tx_data_q, self._rx_data_q,
                self.txgain, self.txfreq, self.txrate, self.rxgain,
                self.rxfreq, self.rxrate, self._otw, self.txant, self.rxant,
                self.rxsamp, self.ref, self.serial, self.args))
            self._proc.start()
            # Wait for process to return device serial number:
            command, val = self._pipe.recv()
            self.serial = val
            self._is_open=True
    def close(self):
        ''' Stop streaming and close the USRP session '''
        if self._is_open:
            if self._tx_running: # stop stream if necessary:
                self.txstop()
            if self._rx_running:
                self.rxstop()
            self._send_command(SdrCommand.CLOSE, None)

            # Wait for process to finish:
            self._proc.join() # wait for process to finish
            self._is_open = False


    # Start continuous mode:
    def txstart(self):
        ''' Start streaming '''
        if self._is_open:
            self._send_command(SdrCommand.START_TX_CONT, None)
            self._tx_running = True
    def rxstart(self):
        ''' Start streaming '''
        if self._is_open:
            self._send_command(SdrCommand.START_RX_CONT, None)
            self._rx_running = True


    # Stop continuous mode:
    def txstop(self):
        ''' Stop streaming without closing the USRP session '''
        if self._is_open:
            self._send_command(SdrCommand.STOP_TX_CONT, None)
            self._tx_data_q.put(int(0))
            # Wait for the data queue to empty:
            while not self._tx_data_q.empty():
                time.sleep(0.1)
            self._tx_running = False
    def rxstop(self):
        ''' Stop streaming without closing the USRP session '''
        if self._is_open:
            self._send_command(SdrCommand.STOP_RX_CONT, None)
            # Drain remaining items from the data queue:
            while not isinstance(self._rx_data_q.get(), int):
                pass # remove items from data queue until we get an int
            self._rx_running = False


    # Continuous mode get/send functions:
    def get(self):
        '''
        Return a tuple with a numpy array with received IQ samples and the
        array start time from the USRP metadata. The start time is a tuple of
        (whole_seconds, fractional_seconds).

        This function doesn't check to make sure the process is open and
        running in rx mode and will be blocked indefinitely or raise an error
        if it isn't
        '''
        # Get data:
        data, data_time = self._rx_data_q.get()
        # convert to complex64:
        data.dtype=self._otw
        data = np.array(data, dtype=np.float32)/self._data_max
        data.dtype=np.complex64
        return data, data_time
    def send(self, data):
        '''
        Transmit a NumPy array during continuous transmission.

        This function doesn't check to make sure the process is open and
        running in tx modeand will may block indefinitely if it isn't.

        The given data is added to a queue for the transmitter. The transmitter
        will underflow if the queue runs out of data
        '''
        # Make sure array is complex64 scaled from [-1,1] to DAC range:
        data = np.array(data, dtype=np.complex64)*self._data_max
        data.dtype = np.float32 # reinterpret cast
        data = np.array(data, dtype=self._otw) # cast to ints
        data.dtype = self._q_dtype # reinterpret cast
        # Send to transmit process:
        self._tx_data_q.put(data)


    # Burst mode:
    def txburst(self, data, start=None):
        '''
        data:  data array to send
        start: start time for transmission or reception as measured by the
               device timer. This can be a number or a tuple with
               (whole_seconds, fractional_seconds). Use None to transmit now.

        Transmits one array
        '''
        if self._is_open and not self._tx_running:
            if start is not None and not isinstance(start, tuple):
                # Convert to tuple if necessary:
                start = (int(start), start-int(start))
            if start is None: # Send now:
                spec = (False, (0,0.0))
            else: # Send when clock=start
                spec = (True, start)
            self._send_command(SdrCommand.TX_BURST, spec)
            # Send data:
            self.send(data)
    def rxburst(self, start=None):
        '''
        start: start time for transmission or reception as measured by the
               device timer. This can be a number or a tuple with
               (whole_seconds, fractional_seconds). Use None to receive now.

        Receives one array of length SDR.rxsamp
        '''
        if self._is_open and not self._rx_running:
            if start is not None and not isinstance(start, tuple):
                # Convert to tuple if necessary:
                start = (int(start), start-int(start))
            if start is None: # Receive now:
                spec = (False, (0,0.0))
            else: # Receive when clock=start
                spec = (True, start)
            # Send the burst commane:
            self._send_command(SdrCommand.RX_BURST, spec)
            # Fetch data:
            sig_data = self.get()
            return sig_data


    # Sensor functions:
    def time(self):
        ''' Return USRP device time '''
        return self._send_command(SdrCommand.GET_USRP_TIME, None)
    def gps_time(self):
        ''' Return GPS time '''
        return self._send_command(SdrCommand.GET_GPS_TIME, None)
    def gps_loc(self):
        '''
        Return GPS location
        '''
        if self._is_open:
            # Get GPS location as a GPGGA sentence:
            val = self._send_command(SdrCommand.GET_GPS_GPGGA, None)
            gpgga = val.split(',')
            # Parse GPGGA:
            utc = 60*(60*float(gpgga[1][:2])+float(gpgga[1][2:4])) +\
                float(gpgga[1][4:]) # seconds after the start of the day
            # Add UTC time at the start of today to the GPGGA UTC:
            now = time.time() # time now
            if utc > 82800 and now%86400 < 3600:
                # If now is before 1 AM and UTC time is after 11 PM, assume the
                # GPGGA time is from yesterday:
                utc += (now//86400 - 1)*86400 # UTC in seconds at the start of yesterday
            else:
                utc += now//86400*86400 # UTC in seconds at the start of today
            lat_deg = float(gpgga[2][:2])+float(gpgga[2][2:])/60
            if gpgga[3].upper()=='S':
                lat_deg = -lat_deg
            lon_deg = float(gpgga[4][:3])+float(gpgga[4][3:])/60
            if gpgga[5].upper()=='W':
                lon_deg = -lon_deg
            loc={
                'utc_sec':utc, # UTC time in seconds
                #'hour':int(gpgga[1][:2],
                #'minute':int(gpgga[1][2:4],
                #'second':float(gpgga[1][4:],
                'lat_degrees':lat_deg, # latitude in degrees
                'lon_degrees':lon_deg, # longitude in degrees
                'alt_meters':float(gpgga[9]), # altitude in meters
                'prec_meters':float(gpgga[8]), # horizontal precision in meters
                'sat_count':int(gpgga[7]), # number of satellites in use
                'fix_type':int(gpgga[6]) # fix type (0=invalid, 1=GPS, 2=DGPS)
            }
            return loc
    def gps_locked(self):
        ''' Return GPS lock status (True if locked, False otherwise) '''
        return self._send_command(SdrCommand.GET_GPS_LOCKED, None)
    def get_serial(self):
        ''' Return USRP serial number '''
        return self._send_command(SdrCommand.GET_SERNUM, None)


    # SDR parameter functions:
    def get_txgain(self):
        ''' Return USRP transmit gain '''
        return self._send_command(SdrCommand.GET_TX_GAIN, None)
    def get_txfreq(self):
        ''' Return USRP transmit center frequency '''
        return self._send_command(SdrCommand.GET_TX_FREQ, None)
    def get_txrate(self):
        ''' Return USRP transmit sample rate '''
        return self._send_command(SdrCommand.GET_TX_RATE, None)
    def get_txant(self):
        ''' Return USRP transmit antenna '''
        return self._send_command(SdrCommand.GET_TX_ANT, None)
    def set_txgain(self, gain):
        '''
        gain: new transmit amplifier gain
        Returns the actual amplifier gain

        Sets the USRP amplifier gain
        '''
        return self._send_command(SdrCommand.SET_TX_GAIN, gain)
    def set_txfreq(self, freq):
        '''
        freq: new transmit center frequency
        Returns the actual center frequency

        Sets the USRP center frequency
        '''
        return self._send_command(SdrCommand.SET_TX_FREQ, freq)
    def set_txrate(self, rate):
        '''
        rate: new transmit sample rate
        Returns the actual sample rate

        Sets the USRP sample rate (cannot be set while the USRP is transmitting
        or receiving)
        '''
        if not self._tx_running and not self._rx_running:
            return self._send_command(SdrCommand.SET_TX_RATE, rate)
    def set_txant(self, antenna):
        '''
        antenna: new transmit antenna
        Returns the actual transmitter antenna

        Sets the USRP transmitter antenna
        '''
        return self._send_command(SdrCommand.SET_TX_ANT, antenna)

    def get_rxgain(self):
        ''' Return USRP receive gain '''
        return self._send_command(SdrCommand.GET_RX_GAIN, None)
    def get_rxfreq(self):
        ''' Return USRP receive center frequency '''
        return self._send_command(SdrCommand.GET_RX_FREQ, None)
    def get_rxrate(self):
        ''' Return USRP receive sample rate '''
        return self._send_command(SdrCommand.GET_RX_RATE, None)
    def get_rxant(self):
        ''' Return USRP receive antenna '''
        return self._send_command(SdrCommand.GET_RX_ANT, None)
    def set_rxgain(self, gain):
        '''
        gain: new receive amplifier gain
        Returns the actual amplifier gain

        Sets the USRP amplifier gain
        '''
        return self._send_command(SdrCommand.SET_RX_GAIN, gain)
    def set_rxfreq(self, freq):
        '''
        freq: new receive center frequency
        Returns the actual center frequency

        Sets the USRP center frequency
        '''
        return self._send_command(SdrCommand.SET_RX_FREQ, freq)
    def set_rxrate(self, rate):
        '''
        rate: new receive sample rate
        Returns the actual sample rate

        Sets the USRP sample rate (cannot be set while the USRP is transmitting
        or receiving)
        '''
        if not self._tx_running and not self._rx_running:
            return self._send_command(SdrCommand.SET_RX_RATE, rate)
    def set_rxant(self, antenna):
        '''
        antenna: new receive antenna
        Returns the actual receiver antenna

        Sets the USRP receiver antenna
        '''
        return self._send_command(SdrCommand.SET_RX_ANT, antenna)
    def set_rxsamp(self, rxsamp):
        '''
        rxsamp: number of samples to measure during a receive

        Sets the number of sample the USRP collects on each receive. Cannot be
        changed while the USRP is receiving
        '''
        if not self._rx_running:
            self.rxsamp = self._send_command(SdrCommand.SET_RX_SAMP, rxsamp)

    def update_params(self):
        '''
        Updates the USRP gain, frequency, and sample rate using the gain, freq,
        and rate parameters. If the USRP is a receiver, this also updates the
        number of samples measured for each collection.
        '''
        self.set_txgain(self.txgain)
        self.set_txfreq(self.txfreq)
        self.set_txrate(self.txrate)
        self.set_rxgain(self.rxgain)
        self.set_rxfreq(self.rxfreq)
        self.set_rxrate(self.rxrate)
        self.set_rxsamp(self.rxsamp)


    # Internal functions:
    def _send_command(self, command, data):
        '''
        Sends a command "command" with data "data" to the USRP process and
        returns the USRP reponse
        '''
        if self._is_open:
            self._pipe_lock.acquire()
            # Send the command:
            self._pipe.send( (command, data) )
            # Wait for the response:
            return_command, return_data = self._pipe.recv()
            # Make sure the response is valid:
            assert return_command==command
            self._pipe_lock.release()
            return return_data
