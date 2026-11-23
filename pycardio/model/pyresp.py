import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
import ctypes as ct
import numpy as np
from os import getcwd
import parameters
import mkmodel

m = mkmodel.MkModel()
p = parameters.SetParameterValues()

ndim = 13
maxtime = 100
ln = 1000 * maxtime
uu = np.zeros( (ln , ndim ) )
m.integrate( ct.c_int(ln), ct.c_int(ndim),
             uu,
             ct.c_double( maxtime ),
             ct.byref( p ) )

# tt = np.arange(ln)/1000.
# plt.subplot(411)

# tpi = tt > 85
# tp = tt[tpi]
# up = uu[tpi,:]
# plt.plot( tp, up[:,3] )
# plt.subplot(412)
# plt.plot( tp, up[:,2] )
# plt.subplot(413)
# plt.plot( tp, up[:,0] )
# plt.subplot(414)
# plt.plot( tp, up[:,4] )
# plt.savefig('out.pdf')
# plt.close()
