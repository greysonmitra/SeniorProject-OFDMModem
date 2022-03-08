#!/usr/bin/env python3
import sys
import time
from argparse import ArgumentParser
import uhd

parser = ArgumentParser()
parser.add_argument('-s','--serial', type=str, default=None,
  help='USRP serial number')
args = parser.parse_args()

sdr_args = ''
if args.serial is not None:
    sdr_args += 'serial=%s'%args.serial

usrp = uhd.usrp.MultiUSRP(sdr_args)

if not 'gps_locked' in usrp.get_mboard_sensor_names():
    print('GPS not detected')
    sys.exit(0)

# Functions for checking the GPS and ref locks:
gps_locked = lambda: usrp.get_mboard_sensor('gps_locked').to_bool()
ref_locked = lambda: usrp.get_mboard_sensor('ref_locked').to_bool()

# Wait for GPS and reference locks:
print('Waiting for GPS lock')
i=1
while not gps_locked():
    time.sleep(1)
    if i:
        print('.', flush=True, end='')
    else:
        print('\r   \r', flush=True, end='')
    i = (i + 1) % 4

print('\nWaiting for ref lock')
i=1
while not ref_locked():
    time.sleep(1)
    if i:
        print('.', flush=True, end='')
    else:
        print('\r   \r', flush=True, end='')
    i = (i + 1) % 4
print('\nGPS is ready')
