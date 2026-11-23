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
        ("d5", ct.c_double),
        ("C", ct.c_double),
        ("gL", ct.c_double),
        ("v12mp", ct.c_double),
        ("kmp", ct.c_double),
        ("EL", ct.c_double),
        ("v12hp", ct.c_double),
        ("khp", ct.c_double),
        ("tauhp", ct.c_double),
        ("gNaP", ct.c_double),
        ("ENa", ct.c_double),
        ("EK", ct.c_double),
        ("Ee", ct.c_double),
        ("Ei", ct.c_double),
        ("v12n", ct.c_double),
        ("kn", ct.c_double),
        ("gKdr", ct.c_double),
        ("gAD", ct.c_double),
        ("tad", ct.c_double),
        ("gE", ct.c_double),
        ("gI", ct.c_double),
        ("hb", ct.c_double),
        ("hb2", ct.c_double),
        ("st2", ct.c_double),
        ("st4", ct.c_double),
        ("a12", ct.c_double),
        ("b32", ct.c_double),
        ("b42", ct.c_double),
        ("b23", ct.c_double),
        ("b43", ct.c_double),
        ("b24", ct.c_double),
        ("b34", ct.c_double),
        ("a15", ct.c_double),
        ("b25", ct.c_double),
        ("b35", ct.c_double),
        ("b45", ct.c_double),
        ("b41", ct.c_double),
        ("b31", ct.c_double)
    ]


def setParams(dParams=None, **kwargs):
    p = Parameters()
    if not isinstance(dParams, dict):
        dParams = defaultParams
    elif isinstance(dParams, dict):
        notin = dict([(dk, dv) for dk, dv in defaultParams.items() if dk not in dParams])
        dParams.update(notin)
    valid_fields = {name for name, _ in Parameters._fields_}
    unknown = (set(dParams) | set(kwargs)) - valid_fields
    if unknown:
        raise ValueError("Unsupported model parameter(s): {}".format(", ".join(sorted(unknown))))
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
    d5 = 0.6, # drive to ramp-i
    C = 20.0,
    gL = 2.8,
    v12mp = -40.0,
    kmp = -6.0,
    EL = -60.0,
    v12hp = -55.0,
    khp = 10.0,
    tauhp = 4000.0,
    gNaP = 5.0,
    ENa = 50.0,
    EK = -85.0,
    Ee = 0.0,
    Ei = -75.0,
    v12n = -30.0,
    kn = -4.0,
    gKdr = 5.0,
    gAD = 10.0,
    tad = 2000.0,
    gE = 10.0,
    gI = 60.0,
    hb = 0.2,
    hb2 = 0.0,
    st2 = 0.0,
    st4 = 0.0,
    a12 = 0.5,
    b32 = 0.05,
    b42 = 0.42,
    b23 = 0.42,
    b43 = 0.2,
    b24 = 0.22,
    b34 = 0.0,
    a15 = 0.5,
    b25 = 0.3,
    b35 = 0.7,
    b45 = 0.7,
    b41 = 0.1,
    b31 = 0.0
)

defaultBounds = dict(
    hr = (0.00045, 0.002),
    
    d4 = (0.35, 0.65),
    d5 = (0.45, 0.75),

    rsa1 = (0.00045, 0.0015),
    rsa2 = (0.0004, 0.0014)
    
)
