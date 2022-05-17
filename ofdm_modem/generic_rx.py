#!/usr/bin/python3
#
# Generic streaming receiver template.
#
# Author: SLT
# Date: 20210505
#

import numpy as np
import argparse
import time

if __name__ == '__main__':

  nProc = 1000 # Number of items to generate per processing loop iteration

  # Parse command line inputs
  parser = argparse.ArgumentParser(description='Base OFDM demodulator code')
  parser.add_argument('-v','--verbose',action='store_true',help='Print verbose output to stdout')
  parser.add_argument('filename')
  args = parser.parse_args()
  print('args: ',args)
  verbose = args.verbose
  fName = args.filename

  # Open read file
  bytesPerRead = np.dtype(np.int8).itemsize*nProc
  iFile = open(fName,'rb')

  print('SLT DEVELOPMENT NOTES: Probably want to get rid of "import time"')
  # The try-except-finally clause allows us to hit ctrl-c on the keyboard to
  # quit and exit gracefully, e.g., by closing open files etc.
  try:
    # This is the processing loop
    while True:
      data = iFile.read(bytesPerRead)
      # Done reading file
      if data == b'':
        break
      # Convert binary data buffer to numpy data type
      bits = np.frombuffer(data,dtype=np.int8)

      if verbose:
        print('bits: %r'%bits)
      
      # This part is not needed other than to slow things down for us humans.
      print('Sleeping for 1.0 second')
      time.sleep(5.0)
  except KeyboardInterrupt as e:
    print('I caught the keyboard interrupt: ',e)
  finally:
    print('Finally done')
    if not iFile.closed:
      print('Need to close the file')
      iFile.close()
