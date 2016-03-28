import sys
import h5py
import sharpnsls2

def run(arguments=sys.argv):
    sharp = sharpnsls2.SharpNSLS2()
    status = sharp.run(arguments)
    fileName = sharp.getInputFile()
    f = h5py.File(fileName,'r')
    f.close()
    return status
