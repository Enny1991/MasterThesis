import numpy as np
import sys

cont = 0
for line in sys.stdin:
    cont += 1
    line = np.fromstring(line.strip(), sep=' ')
    print len(line)

print 'counter %i' % cont
