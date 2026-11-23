/*
 * Parameter table for the cardiorespiratory model.
 *
 * This X-macro table is the single source of truth for every model
 * parameter: the C struct, the default values, and the names/descriptions
 * exposed to Python are all generated from it, so the Python side can never
 * drift out of sync with the C struct layout.
 *
 * Units: time in ms, voltage in mV, conductance in nS, capacitance in pF.
 * Heart-phase rates are in cycles per ms.
 *
 * Model by William Barnett & Robert Capps (2018); parameter table and
 * integrator rewritten during the 2026 overhaul (see CHANGES-AI.md).
 */
#ifndef PYCARDIO_MODEL_PARAMS_H
#define PYCARDIO_MODEL_PARAMS_H

#define PYCARDIO_PARAMS(X)                                                       \
  /* ---- noise ------------------------------------------------------------ */ \
  X(rs1,   0.0001,  "noise amplitude on the early-I adaptation (U(-1/2,1/2) per ms equivalent)") \
  /* ---- cardiac / vascular ---------------------------------------------- */ \
  X(hr,    0.00084, "baseline heart-phase velocity (cycles/ms)")                 \
  X(rsa1,  0.0005,  "post-I -> heart-phase coupling (RSA, cycles/ms)")           \
  X(rsa2,  0.00035, "ramp-I -> heart-phase coupling (RSA, cycles/ms)")           \
  X(po,    1.0,     "blood-pressure set point (a.u.)")                           \
  X(taup,  1000.0,  "blood-pressure time constant (ms)")                         \
  X(taul,  1000.0,  "lung volume time constant (ms)")                            \
  X(taupp, 2000.0,  "smoothed blood-pressure time constant (ms)")                \
  X(baro,  0.0,     "respiratory baroreflex gain (BP -> post-I)")                \
  X(sym,   0.0,     "sympathetic (ramp-I) drive to BP")                          \
  X(db,    0.0,     "cardiac baroreflex gain (smoothed BP -> heart phase)")      \
  X(sb,    0.0,     "vascular baroreflex gain (smoothed BP -> BP)")              \
  /* ---- tonic drives to the respiratory CPG ----------------------------- */ \
  X(d1,    0.09,    "drive to pre-I/I")                                          \
  X(d2,    0.564,   "drive to early-I")                                          \
  X(d3,    0.45,    "drive to aug-E")                                            \
  X(d4,    0.5,     "drive to post-I")                                           \
  X(d5,    0.6,     "drive to ramp-I")                                           \
  /* ---- external stimuli ------------------------------------------------- */ \
  X(st2,   0.0,     "inhibitory stimulus to early-I")                            \
  X(st4,   0.0,     "excitatory stimulus to post-I")                             \
  /* ---- membrane biophysics --------------------------------------------- */ \
  X(v12mp, -40.0,   "NaP activation half-voltage (mV)")                          \
  X(kmp,   -6.0,    "NaP activation slope (mV)")                                 \
  X(EL,    -60.0,   "leak reversal potential (mV)")                              \
  X(C,     20.0,    "membrane capacitance (pF)")                                 \
  X(gL,    2.8,     "leak conductance (nS)")                                     \
  X(v12hp, -55.0,   "NaP inactivation half-voltage (mV)")                        \
  X(khp,   10.0,    "NaP inactivation slope (mV)")                               \
  X(tauhp, 4000.0,  "max NaP inactivation time constant (ms)")                   \
  X(gNaP,  5.0,     "persistent sodium conductance (nS)")                        \
  X(ENa,   50.0,    "sodium reversal potential (mV)")                            \
  X(EK,    -85.0,   "potassium reversal potential (mV)")                         \
  X(Ee,    0.0,     "excitatory synaptic reversal potential (mV)")               \
  X(Ei,    -75.0,   "inhibitory synaptic reversal potential (mV)")               \
  X(v12n,  -30.0,   "delayed-rectifier half-voltage (mV)")                       \
  X(kn,    -4.0,    "delayed-rectifier slope (mV)")                              \
  X(gKdr,  5.0,     "delayed-rectifier conductance (nS)")                        \
  X(gAD,   10.0,    "adaptation conductance (nS)")                               \
  X(tad,   2000.0,  "adaptation time constant, aug-E and post-I (ms)")           \
  X(tad_ei, 2000.0, "adaptation time constant, early-I (ms)")                    \
  X(gE,    10.0,    "excitatory synaptic conductance scale (nS)")                \
  X(gI,    60.0,    "inhibitory synaptic conductance scale (nS)")                \
  X(hb,    0.2,     "lung-volume (Hering-Breuer) feedback to post-I")            \
  X(hb2,   0.0,     "lung-volume (Hering-Breuer) feedback to early-I")           \
  X(vlow,  -50.0,   "output nonlinearity: zero-rate voltage (mV)")               \
  X(vhigh, -20.0,   "output nonlinearity: saturation voltage (mV)")              \
  /* ---- connectivity ----------------------------------------------------- */ \
  X(a12,   0.5,     "pre-I/I -> early-I (excitatory)")                           \
  X(b32,   0.05,    "aug-E -| early-I")                                          \
  X(b42,   0.42,    "post-I -| early-I")                                         \
  X(b23,   0.42,    "early-I -| aug-E")                                          \
  X(b43,   0.2,     "post-I -| aug-E")                                           \
  X(b24,   0.22,    "early-I -| post-I")                                         \
  X(b34,   0.0,     "aug-E -| post-I")                                           \
  X(a15,   0.5,     "pre-I/I -> ramp-I (excitatory)")                            \
  X(b25,   0.3,     "early-I -| ramp-I")                                         \
  X(b35,   0.7,     "aug-E -| ramp-I")                                           \
  X(b45,   0.7,     "post-I -| ramp-I")                                          \
  X(b41,   0.1,     "post-I -| pre-I/I")                                         \
  X(b31,   0.0,     "aug-E -| pre-I/I")

typedef struct {
#define PYCARDIO_FIELD(name, value, doc) double name;
  PYCARDIO_PARAMS(PYCARDIO_FIELD)
#undef PYCARDIO_FIELD
} Parameters;

#endif
