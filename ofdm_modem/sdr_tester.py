#!/usr/bin/env python3
import time
import numpy as np
try:
    import matplotlib.pyplot as plt
    has_plt = True
except:
    has_plt = False
if __name__=='__main__':
    import sys
    import os
    parent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir))
    sys.path.insert(0, parent_dir)
# imports from this directory:
from sdr import SDR


def test_rx_continuous(serial=None):
    '''
    Runs an SDR in continuous receive mode and does some reconfiguration during
    reception
    '''
    sdr = SDR(10)
    sdr.serial=serial
    print('Setting SDR Params to 50 dB gain, 2.45 GHz center freq,')
    print('10 MHz sample rate, and 1M-sample buffer size')
    sdr.rxgain = 50
    sdr.rxfreq = 2.45e9
    sdr.rxrate = 10e6
    sdr.rxsamp = int(1e6)
    print('--- Openning SDR ---')
    sdr.open()

    print('--- Starting SDR ---')
    sdr.rxstart()
    for i in range(3):
        stop_time = time.time()+1
        while time.time()<stop_time:
            data, data_time = sdr.get()
            print('  Got %d samples at %f'%(data.size,sum(data_time)))
            print('  mean power: %e'%(np.linalg.norm(data)/data.size))
        if i==0:
            print('changing gain to 30 dB')
            sdr.set_rxgain(30)
        elif i==1:
            print('changing freq to 2.448 GHz')
            sdr.set_rxfreq(2.448e9)
    print('--- Stopping SDR ---')
    sdr.rxstop()
    print('--- Changing sample rate for high sample-rate test (40 MHz) ---')
    sdr.set_rxrate(40e6)
    print('Setting rx buffer size to 2M samples')
    sdr.set_rxsamp(int(2e6))
    sdr.rxstart()
    stop_time = time.time()+1
    while time.time()<stop_time:
        data, data_time = sdr.get()
        print('Got %d samples at %f'%(data.size,sum(data_time)))
    print('--- Closing SDR ---')
    sdr.close()
    print('Done')


def test_rx_burst(serial=None):
    '''
    Measures a single received signal at a specific time
    '''
    sdr = SDR()
    sdr.serial=serial
    print('Setting SDR Params to 40 dB gain, 2.45 GHz center freq,')
    print('10 MHz sample rate, and 1M-sample buffer size')
    sdr.rxgain = 40
    sdr.rxfreq = 2.45e9
    sdr.rxrate = 10e6
    sdr.rxsamp = int(1e6)
    print('--- Openning SDR ---')
    sdr.open()

    print('--- Collecting data now ---')
    rx_data1, time_spec = sdr.rxburst()
    print(' signal received at %f'%sum(time_spec))

    print('--- Collecting data in 5 seconds ---')
    col_time = time.time()+5
    rx_data2, time_spec = sdr.rxburst(start=col_time)
    print(' signal received at %f'%sum(time_spec))
    print('--- Closing SDR ---')
    sdr.close()

    if has_plt:
        # Plot the amplitude over time:
        print('--- Plotting data ---')
        time_ax = np.linspace(0, (rx_data1.size-1)/sdr.rxrate, rx_data1.size)
        plt.figure()
        plt.plot(time_ax*1e3, np.abs(rx_data1))
        plt.xlabel('time (ms)')
        plt.ylabel('amplitude (first collect)')

        time_ax = np.linspace(0, (rx_data2.size-1)/sdr.rxrate, rx_data2.size)
        plt.figure()
        plt.plot(time_ax*1e3, np.abs(rx_data2))
        plt.xlabel('time (ms)')
        plt.ylabel('amplitude (second collect)')
        plt.show()


def test_tx_continuous(serial=None):
    '''
    Runs an SDR in continuous transmit mode and does some reconfiguration
    during transmission
    '''
    sdr = SDR()
    sdr.serial=serial
    print('Setting SDR Params to 50 dB gain, 2.45 GHz center freq,')
    print('10 MHz sample rate, and 1M-sample transmit signal size')
    sdr.txgain = 50
    sdr.txfreq = 2.45e9
    sdr.txrate = 10e6
    rng = np.random.default_rng()
    # Generate a noise signal:
    tx_data = rng.standard_normal(2*int(1e6), dtype=np.float32)
    tx_data.dtype = np.complex64
    tx_data /= np.max(np.abs(tx_data))
    print('--- Openning SDR ---')
    sdr.open()

    print('--- Starting SDR ---')
    sdr.txstart()
    for i in range(3):
        stop_time = time.time()+1
        while time.time()<stop_time:
            sdr.send(tx_data)
        if i==0:
            print('changing gain to 70 dB')
            sdr.set_txgain(70)
        elif i==1:
            print('changing freq to 2.448 GHz')
            sdr.set_txfreq(2.448e9)
    print('--- Stopping SDR ---')
    sdr.txstop()
    time.sleep(1) # wait for the SDR to stop transmitting
    print('--- Changing sample rate for high sample-rate test (40 MHz) ---')
    sdr.set_txrate(40e6)
    print('Setting signal size to 2M samples')
    # Generate a noise signal:
    tx_data = rng.standard_normal(2*int(2e6), dtype=np.float32)
    tx_data.dtype = np.complex64
    tx_data /= np.max(np.abs(tx_data))
    sdr.txstart()
    stop_time = time.time()+1
    while time.time()<stop_time:
        sdr.send(tx_data)
    print('--- Closing SDR ---')
    sdr.close()
    print('Done')


def test_tx_burst(serial=None):
    '''
    Transmits a burst of data without a time spec and then with a time spec
    '''
    sdr = SDR()
    sdr.serial=serial
    print('Setting SDR Params to 50 dB gain, 2.45 GHz center freq,')
    print('10 MHz sample rate, and 1M-sample transmit signal size')
    sdr.txgain = 50
    sdr.txfreq = 2.45e9
    sdr.txrate = 10e6
    rng = np.random.default_rng()
    # Generate a linear chirp:
    duration = 2
    freq_delta = 5e6
    samp_time = np.linspace(-duration/2, duration/2, int(duration*sdr.txrate))
    tx_data = np.exp(2j*np.pi * 1/2*samp_time**2 * (2/duration) * freq_delta)
    print('--- Openning SDR ---')
    sdr.open()

    print('--- Sending burst now ---')
    sdr.txburst(data=tx_data)

    print('--- Sending burst in 5 seconds ---')
    send_time = time.time()+5
    sdr.txburst(start=send_time, data=tx_data)

    print('--- Closing SDR ---')
    sdr.close()


def test_gets(serial):
    '''
    Tests SDR sensor get functions
    '''
    sdr = SDR()
    sdr.serial=serial
    print('--- Openning SDR ---')
    sdr.open()

    print('TX antenna:', sdr.get_txant())
    print('RX antenna:', sdr.get_rxant())
    print('USRP time: ', sdr.time())
    print('GPS time:  ', sdr.gps_time())
    print('GPS coords:', sdr.gps_loc())
    print('GPS locked:', sdr.gps_locked())
    print('SDR serial:', sdr.get_serial())

    print('--- Closing SDR ---')
    sdr.close()


def test_gps(serial):
    '''
    Tests GPS synchronization and print GPS data
    '''
    sdr = SDR()
    sdr.serial=serial
    sdr.ref='gps'
    print('--- Openning SDR ---')
    sdr.open()

    print('--- Reading GPS data ---')
    print(' GPS locked:', sdr.gps_locked())
    print(' GPS time = %f (system time = %f)'%(sdr.gps_time(),time.time()))
    print(' USRP time = %f'%(sdr.time()))
    print(' GPS coordinate dict:', sdr.gps_loc())

    print('--- Closing SDR ---')
    sdr.close()


def test_sdr(mode, continuous, serial):
    '''
    Runs an SDR test based on the mode and whether to run in continuous mode
    '''
    if mode=='rx' and continuous:
        test_rx_continuous(serial)
    elif mode=='rx':
        test_rx_burst(serial)
    elif mode=='tx' and continuous:
        test_tx_continuous(serial)
    else:
        test_tx_burst(serial)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gets', action='store_true',
      help='test getting data from the SDR')
    parser.add_argument('--gps', action='store_true',
      help='test GPS synchronization and getting GPS data')
    parser.add_argument('--tx', action='store_true',
      help='test in transmit mode')
    parser.add_argument('--rx', action='store_true',
      help='test in receive mode')
    parser.add_argument('-c','--continuous', action='store_true',
      help='test continuous mode instead of burst mode')
    parser.add_argument('-s','--serial', type=str, default=None,
      help='device serial number')
    args = parser.parse_args()

    if args.gets:
        test_gets(args.serial)
        sys.exit(0)
    elif args.gps:
        test_gps(args.serial)
        sys.exit(0)
    elif args.rx:
        mode = 'rx'
    elif args.tx:
        mode = 'tx'
    else:
        print('Specify --tx, --rx, --gets, or --gps')
        sys.exit(0)

    test_sdr(mode, args.continuous, args.serial)
