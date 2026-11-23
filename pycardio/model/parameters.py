import ctypes as ct

class Parameters(ct.Structure):
    _fields_ = [
        ("eiAd", ct.c_double),
        ("rs1", ct.c_double),
        ("hr", ct.c_double),
        ("rsa1", ct.c_double),
        ("rsa2", ct.c_double),
        ("po", ct.c_double),
        ("taup", ct.c_double),
        ("taul", ct.c_double),
        ("taupp", ct.c_double),
        ("baro", ct.c_double),
        ("sym", ct.c_double),
        ("db", ct.c_double),
        ("sb", ct.c_double),
        ("d1", ct.c_double),
        ("d2", ct.c_double),
        ("d3", ct.c_double),
        ("d4", ct.c_double),
        ("d5", ct.c_double)
    ]


def setParams(dParams=None, **kwargs):
    p = Parameters()
    if not isinstance(dParams, dict):
        dParams = defaultParams
    elif isinstance(dParams, dict):
        notin = dict([(dk, dv) for dk, dv in defaultParams.items() if dk not in dParams])
        dParams.update(notin)
    for k, v in dParams.items():
        setattr(p, k, v)
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p

defaultParams = dict(
    rs1=0.0001, # noise to early-i
    
    hr=0.00084, # baseline heart rate
    rsa1=0.0005, # ramp-I rsa
    rsa2=0.00035, # post-I rsa
    
    po=1.0,
    
    taup=1000,
    taul=1000,
    taupp=2000,
    
    baro = 0.0, # respiratory baroreflex
    sym = 0.0,
    db = 0.0, # cardiac baroreflex
    sb = 0.0,

    # drives to respiratory cpg
    d1 = 0.09, # drive to prei-i
    d2 = 0.564, # drive to early-i
    d3 = 0.45, # drive to aug-e
    d4 = 0.5, # drive to post-i
    d5 = 0.6 # drive to ramp-i
)

defaultBounds = dict(
    hr = (0.00045, 0.002),
    
    d4 = (0.35, 0.65),
    d5 = (0.45, 0.75),

    rsa1 = (0.00045, 0.0015),
    rsa2 = (0.0004, 0.0014)
    
)
