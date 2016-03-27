import sys
import sharpnsls2

def run(arguments=sys.argv):
    sharp = sharpnsls2.SharpNSLS2()
    return sharp.run(arguments)
