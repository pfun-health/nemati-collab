# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.expanduser("~/Git/nemati-collab"))
import pycardio

f2y03 = pycardio.load_stats("f2y03","fantasia")

def give_bphase(stats,model=False):
    if not model:
        time, bix, resp = stats['signals']['time'], stats['signals']['hb'], stats['signals']['tv']
    elif model:
        time, bix, resp = stats['time'], stats['hb'], stats['tv']
    btimes, bphase = pycardio.process_data.DB.calc_beat_phase(time, bix, resp)
    return btimes, bphase

ytimes, yphase = give_bphase(f2y03)

plt.figure(figsize=(12,12),dpi=300)
plt.plot(ytimes, yphase, 'k.')
plt.xlabel('time (sec)')
plt.ylabel('phase [Inspiration -> Inspiration]')
plt.title('Fantasia Data')
plt.savefig('figs/f2y03_phase_time.svg')
plt.close()

f1o01 = pycardio.load_stats("f1o01", "fantasia")
otimes, ophase = give_bphase(f1o01)

plt.figure(figsize=(12,12),dpi=300)
plt.plot(otimes, ophase, 'k.')
plt.xlabel('time (sec)')
plt.ylabel('phase [Inspiration -> Inspiration]')
plt.title('Fantasia Data')
plt.savefig('figs/f1o01_phase_time.svg')
plt.close()

my = np.load('./model_f2y03.npy')[()]
mtimes, mphase = give_bphase(my,model=True)

plt.figure(figsize=(12,12),dpi=300)
plt.plot(mtimes, mphase, 'k.')
plt.xlabel('time (sec)')
plt.ylabel('phase [Inspiration -> Inspiration]')
plt.title('Simulation (Participant 2)')
plt.savefig('figs/model_phase_time_2.svg')
plt.close()

my = np.load('./model_f1o01.npy')[()]
mtimes, mphase = give_bphase(my,model=True)

plt.figure(figsize=(12,12),dpi=300)
plt.plot(mtimes, mphase, 'k.')
plt.xlabel('time (sec)')
plt.ylabel('phase [Inspiration -> Inspiration]')
plt.title('Simulation (Participant 1)')
plt.savefig('figs/model_phase_time_1.svg')
plt.close()
