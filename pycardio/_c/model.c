/*
 * Cardiorespiratory network model: a five-population respiratory CPG
 * (pre-I/I, early-I, aug-E, post-I, ramp-I) coupled to a heart-phase
 * oscillator and a lumped blood-pressure / baroreflex loop.
 *
 * Original model: William Barnett (March 2018) and Robert Capps (April 2018).
 * The equations in model_rhs() are unchanged from that version.
 *
 * 2026 overhaul (see CHANGES-AI.md):
 *   - GSL rk2 driver replaced by an embedded Dormand-Prince 5(4) integrator
 *     with the usual error control, so there are no external C dependencies.
 *   - The output grid is honoured (sample l is the state at t = l*dt). The old
 *     code ignored its time-step argument and always stepped 1 ms.
 *   - rand()/srand(time(0)) replaced by a seedable xoshiro256** generator,
 *     so runs are reproducible and calls are thread-safe.
 *   - The early-I adaptation is an ordinary state variable integrated with the
 *     rest (it used to be advanced by one frozen Euler step per output
 *     sample), and its noise is added as a properly scaled increment,
 *     sigma*sqrt(dt)*N(0,1), with sigma = rs1/sqrt(12). At dt = 1 ms this has
 *     the same variance as the original uniform kick, so fitted rs1 values keep
 *     their meaning, and the noise no longer depends on the step size.
 *   - Solver failures are returned to the caller instead of silently leaving
 *     zeros in the output array.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "model_params.h"

#if defined(_WIN32)
#define PC_EXPORT __declspec(dllexport)
#else
#define PC_EXPORT __attribute__((visibility("default")))
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_ODE 13  /* number of state variables */
#define N_OUT 13  /* columns in the output array */

enum {
  PC_OK = 0,
  PC_ERR_STEP_UNDERFLOW = 1,
  PC_ERR_NONFINITE = 2,
  PC_ERR_MAX_STEPS = 3,
  PC_ERR_BAD_ARGS = 4
};

/* ------------------------------------------------------------------------ */
/* Parameter table introspection (used by the Python wrapper)               */
/* ------------------------------------------------------------------------ */

#define PC_NAME(name, value, doc) #name,
#define PC_VALUE(name, value, doc) value,
#define PC_DOC(name, value, doc) doc,
static const char *const param_names[] = {PYCARDIO_PARAMS(PC_NAME)};
static const double param_defaults[] = {PYCARDIO_PARAMS(PC_VALUE)};
static const char *const param_docs[] = {PYCARDIO_PARAMS(PC_DOC)};
#undef PC_NAME
#undef PC_VALUE
#undef PC_DOC

#define N_PARAMS ((int)(sizeof(param_defaults) / sizeof(param_defaults[0])))

PC_EXPORT int pc_param_count(void) { return N_PARAMS; }
PC_EXPORT size_t pc_param_struct_size(void) { return sizeof(Parameters); }
PC_EXPORT const char *pc_param_name(int i) {
  return (i >= 0 && i < N_PARAMS) ? param_names[i] : NULL;
}
PC_EXPORT const char *pc_param_doc(int i) {
  return (i >= 0 && i < N_PARAMS) ? param_docs[i] : NULL;
}
PC_EXPORT double pc_param_default(int i) {
  return (i >= 0 && i < N_PARAMS) ? param_defaults[i] : NAN;
}
PC_EXPORT int pc_n_state(void) { return N_ODE; }
PC_EXPORT int pc_n_output(void) { return N_OUT; }

/* ------------------------------------------------------------------------ */
/* Random numbers: xoshiro256** seeded through splitmix64                   */
/* ------------------------------------------------------------------------ */

typedef struct {
  uint64_t s[4];
  int has_spare;
  double spare;
} Rng;

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(Rng *r) {
  uint64_t *s = r->s;
  const uint64_t result = rotl(s[1] * 5, 7) * 9;
  const uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

static void rng_seed(Rng *r, uint64_t seed) {
  for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&seed);
  r->has_spare = 0;
  r->spare = 0.0;
}

/* uniform on the open interval (0, 1) */
static double rng_uniform(Rng *r) {
  return ((double)(rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/* standard normal via the Marsaglia polar method */
static double rng_normal(Rng *r) {
  if (r->has_spare) {
    r->has_spare = 0;
    return r->spare;
  }
  double u, v, s;
  do {
    u = 2.0 * rng_uniform(r) - 1.0;
    v = 2.0 * rng_uniform(r) - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);
  const double m = sqrt(-2.0 * log(s) / s);
  r->spare = v * m;
  r->has_spare = 1;
  return u * m;
}

/* ------------------------------------------------------------------------ */
/* Model equations                                                          */
/* ------------------------------------------------------------------------ */

static inline double ninf(const Parameters *p, double v) { return 1.0 / (1.0 + exp((v - p->v12n) / p->kn)); }
static inline double mpinf(const Parameters *p, double v) { return 1.0 / (1.0 + exp((v - p->v12mp) / p->kmp)); }
static inline double hpinf(const Parameters *p, double v) { return 1.0 / (1.0 + exp((v - p->v12hp) / p->khp)); }
static inline double tauinf(const Parameters *p, double v) { return p->tauhp / cosh((v - p->v12hp) / p->khp); }

/* piecewise-linear firing-rate nonlinearity */
static inline double rate(const Parameters *p, double v) {
  if (v < p->vlow) return 0.0;
  if (v > p->vhigh) return 1.0;
  return (v - p->vlow) / (p->vhigh - p->vlow);
}

/* pulsatile cardiac output as a function of heart phase */
static inline double beat(double phi) { return pow((1.0 + sin(2.0 * M_PI * phi)) / 2.0, 20.0); }

static inline double pow4(double x) { double x2 = x * x; return x2 * x2; }

/*
 * State vector:
 *   y[0..4] membrane potentials: pre-I/I, early-I, aug-E, post-I, ramp-I
 *   y[5]    pre-I/I NaP inactivation
 *   y[6]    aug-E adaptation
 *   y[7]    post-I adaptation
 *   y[8]    lung volume
 *   y[9]    heart phase (cycles)
 *   y[10]   blood pressure
 *   y[11]   smoothed blood pressure
 *   y[12]   early-I adaptation (eiAd). Its drift is integrated here; the
 *           additive noise is applied between output intervals.
 */
static void model_rhs(const Parameters *p, const double *y, double *dy) {
  const double eiAd = y[12];
  const double f0 = rate(p, y[0]), f1 = rate(p, y[1]), f2 = rate(p, y[2]);
  const double f3 = rate(p, y[3]), f4 = rate(p, y[4]);

  /* pre-I/I */
  dy[0] = -p->gKdr * pow4(ninf(p, y[0])) * (y[0] - p->EK) - p->gL * (y[0] - p->EL)
          - p->gNaP * mpinf(p, y[0]) * y[5] * (y[0] - p->ENa);
  dy[0] += -p->gE * p->d1 * (y[0] - p->Ee) - p->gI * (p->b31 * f2 + p->b41 * f3) * (y[0] - p->Ei);
  dy[0] /= p->C;

  /* early-I */
  dy[1] = -p->gKdr * pow4(ninf(p, y[1])) * (y[1] - p->EK) - p->gL * (y[1] - p->EL)
          - p->gAD * eiAd * (y[1] - p->EK);
  dy[1] += -p->gE * (p->d2 + p->a12 * f0) * (y[1] - p->Ee)
           - p->gI * (p->b32 * f2 + p->b42 * f3 + p->hb2 * y[8] + p->st2) * (y[1] - p->Ei);
  dy[1] /= p->C;

  /* aug-E */
  dy[2] = -p->gKdr * pow4(ninf(p, y[2])) * (y[2] - p->EK) - p->gL * (y[2] - p->EL)
          - p->gAD * y[6] * (y[2] - p->EK);
  dy[2] += -p->gE * p->d3 * (y[2] - p->Ee) - p->gI * (p->b23 * f1 + p->b43 * f3) * (y[2] - p->Ei);
  dy[2] /= p->C;

  /* post-I */
  dy[3] = -p->gKdr * pow4(ninf(p, y[3])) * (y[3] - p->EK) - p->gL * (y[3] - p->EL)
          - p->gAD * y[7] * (y[3] - p->EK);
  dy[3] += -p->gE * (p->d4 + p->hb * y[8] + p->st4 + p->baro * y[10]) * (y[3] - p->Ee)
           - p->gI * (p->b24 * f1 + p->b34 * f2) * (y[3] - p->Ei);
  dy[3] /= p->C;

  /* ramp-I */
  dy[4] = -p->gKdr * pow4(ninf(p, y[4])) * (y[4] - p->EK) - p->gL * (y[4] - p->EL);
  dy[4] += -p->gE * (p->d5 + p->a15 * f0) * (y[4] - p->Ee)
           - p->gI * (p->b25 * f1 + p->b35 * f2 + p->b45 * f3) * (y[4] - p->Ei);
  dy[4] /= p->C;

  dy[5] = (hpinf(p, y[0]) - y[5]) / tauinf(p, y[0]); /* NaP inactivation */
  dy[6] = (f2 - y[6]) / p->tad;                     /* aug-E adaptation */
  dy[7] = (f3 - y[7]) / p->tad;                     /* post-I adaptation */
  dy[8] = (f4 - y[8]) / p->taul;                    /* lung volume */

  /* heart phase */
  dy[9] = p->hr + p->rsa1 * f3 + p->rsa2 * f4 - p->db * y[11];

  /* blood pressure and its slow average */
  dy[10] = (beat(y[9]) + p->sym * f4 - p->sb * y[11] - (y[10] - p->po)) / p->taup;
  dy[11] = (y[10] - y[11]) / p->taupp;

  dy[12] = (f1 - eiAd) / p->tad_ei; /* early-I adaptation (drift) */
}

PC_EXPORT void pc_rhs(const Parameters *p, const double *y, double *dy) { model_rhs(p, y, dy); }

PC_EXPORT void pc_default_initial_state(double *y0) {
  /* same values as the 2018 code; the last entry is eiAd */
  static const double init[N_ODE] = {-60, -60, -60, -60, -30, .3, 0, 0, 0, 0, .1, .1, 0.0};
  memcpy(y0, init, sizeof(init));
}

/* ------------------------------------------------------------------------ */
/* Dormand-Prince 5(4) with standard step-size control                      */
/* ------------------------------------------------------------------------ */

static const double c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9;
static const double a21 = 1.0 / 5;
static const double a31 = 3.0 / 40, a32 = 9.0 / 40;
static const double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
static const double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
static const double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176,
                    a65 = -5103.0 / 18656;
static const double a71 = 35.0 / 384, a73 = 500.0 / 1113, a74 = 125.0 / 192, a75 = -2187.0 / 6784,
                    a76 = 11.0 / 84;
static const double e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920, e5 = -17253.0 / 339200,
                    e6 = 22.0 / 525, e7 = -1.0 / 40;

typedef struct {
  double rtol, atol;
  double h;      /* step-size proposal carried between intervals */
  long n_steps;  /* accepted steps */
  long n_reject; /* rejected steps */
} Stepper;

/* Advance y from t0 to t1. Returns a PC_* code. */
static int dopri5_interval(const Parameters *p, double *y, double t0, double t1, Stepper *st) {
  double k1[N_ODE], k2[N_ODE], k3[N_ODE], k4[N_ODE], k5[N_ODE], k6[N_ODE], k7[N_ODE];
  double yt[N_ODE], ynew[N_ODE];
  const double span = t1 - t0;
  const double hmin = 1e-12 * (fabs(t1) + 1.0);
  const long max_steps = 1000000;
  long steps = 0;
  double t = t0;
  double h = st->h > 0 ? st->h : span;

  model_rhs(p, y, k1);
  while (t < t1) {
    if (++steps > max_steps) return PC_ERR_MAX_STEPS;
    int last = 0;
    double h_used = h;
    if (t + h_used >= t1 - 1e-12 * span) { h_used = t1 - t; last = 1; }

    int i;
    for (i = 0; i < N_ODE; i++) yt[i] = y[i] + h_used * a21 * k1[i];
    model_rhs(p, yt, k2);
    for (i = 0; i < N_ODE; i++) yt[i] = y[i] + h_used * (a31 * k1[i] + a32 * k2[i]);
    model_rhs(p, yt, k3);
    for (i = 0; i < N_ODE; i++) yt[i] = y[i] + h_used * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    model_rhs(p, yt, k4);
    for (i = 0; i < N_ODE; i++)
      yt[i] = y[i] + h_used * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    model_rhs(p, yt, k5);
    for (i = 0; i < N_ODE; i++)
      yt[i] = y[i] + h_used * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    model_rhs(p, yt, k6);
    for (i = 0; i < N_ODE; i++)
      ynew[i] = y[i] + h_used * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
    model_rhs(p, ynew, k7);

    double err = 0.0;
    int finite = 1;
    for (i = 0; i < N_ODE; i++) {
      const double ei = h_used * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
      const double sc = st->atol + st->rtol * fmax(fabs(y[i]), fabs(ynew[i]));
      const double r = ei / sc;
      err += r * r;
      if (!isfinite(ynew[i])) finite = 0;
    }
    err = finite ? sqrt(err / N_ODE) : INFINITY;

    if (err <= 1.0) {
      t = last ? t1 : t + h_used;
      memcpy(y, ynew, sizeof(ynew));
      memcpy(k1, k7, sizeof(k7)); /* FSAL */
      st->n_steps++;
      const double fac = err == 0.0 ? 5.0 : fmin(5.0, fmax(0.2, 0.9 * pow(err, -0.2)));
      /* a truncated final step says nothing about the natural step size */
      if (!last || h_used >= h) h = h_used * fac;
    } else {
      st->n_reject++;
      if (!isfinite(err)) h = 0.25 * h_used;
      else h = h_used * fmax(0.2, 0.9 * pow(err, -0.2));
      if (h < hmin) return finite ? PC_ERR_STEP_UNDERFLOW : PC_ERR_NONFINITE;
    }
  }
  st->h = h;
  return PC_OK;
}

static void store_row(const Parameters *p, const double *y, double *row) {
  for (int j = 0; j < 5; j++) row[j] = rate(p, y[j]); /* firing rates */
  row[5] = y[5];                                      /* NaP inactivation */
  row[6] = y[12];                                     /* early-I adaptation */
  for (int j = 6; j < 12; j++) row[j + 1] = y[j];     /* aug-E iAD ... smoothed BP */
}

/*
 * Simulate the model on the grid t_l = l*dt (ms), l = 0..n_out-1.
 *
 * out     : n_out x 13 row-major array, columns
 *           [f(pre-I/I), f(early-I), f(aug-E), f(post-I), f(ramp-I),
 *            NaP inactivation, early-I iAD, aug-E iAD, post-I iAD,
 *            lung volume, heart phase, BP, smoothed BP]
 * y0      : optional 13 initial state values (see model_rhs); NULL uses
 *           the defaults. On return, if y_final != NULL it receives the final
 *           state in the same layout (useful to continue a simulation).
 * n_steps : optional; receives the number of accepted solver steps.
 */
PC_EXPORT int pc_simulate(const Parameters *p, double dt, int n_out, double *out, uint64_t seed,
                          double rtol, double atol, const double *y0, double *y_final, long *n_steps) {
  if (!p || !out || n_out < 1 || !(dt > 0) || !(rtol > 0) || !(atol > 0)) return PC_ERR_BAD_ARGS;

  double y[N_ODE];
  if (y0) memcpy(y, y0, sizeof(y));
  else pc_default_initial_state(y);

  Rng rng;
  rng_seed(&rng, seed);
  const double sigma = p->rs1 / sqrt(12.0);
  const double sqdt = sqrt(dt);

  Stepper st = {rtol, atol, fmin(dt, 0.1), 0, 0};
  int status = PC_OK;

  store_row(p, y, out);
  for (int l = 1; l < n_out; l++) {
    status = dopri5_interval(p, y, (l - 1) * dt, l * dt, &st);
    if (status != PC_OK) {
      /* mark the unfinished part of the output as invalid */
      for (long k = (long)l * N_OUT; k < (long)n_out * N_OUT; k++) out[k] = NAN;
      break;
    }
    /* additive noise on the early-I adaptation (Wiener increment over dt) */
    if (sigma != 0.0) y[12] += sigma * sqdt * rng_normal(&rng);
    store_row(p, y, out + (long)l * N_OUT);
  }

  if (y_final) memcpy(y_final, y, sizeof(y));
  if (n_steps) *n_steps = st.n_steps;
  return status;
}

#if defined(_WIN32)
/* setuptools insists on exporting a module init symbol on Windows; this
 * library is loaded through ctypes and is never imported as a module. */
PC_EXPORT void PyInit__libmodel(void) {}
#endif
