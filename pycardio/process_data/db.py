import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import os
from ctypes import c_int, c_uint32
from ..load_data._physionet_db import PhysioNetDB
from ._process_data_ct import calc_rsa, calc_cvc, calc_dfa
from ..utils import calc_stderr, simple_fourier

# This package works with wfdb-python==1.3.7

"""
    db.py: Contains the DB (database) class for organizing physiology data and calculating relevant statistics.
"""

class DB(object):
    def __init__(self, parent=PhysioNetDB, database='fantasia', dpath="", ftype='ecg', **kwargs):
        self.obj = parent(database=database, dpath=dpath, ftype=ftype, **kwargs)

    def __call__(self, dtype='db', dmode='dict'):
        return self.obj.__call__(dtype=dtype, dmode=dmode)
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
    
    @staticmethod
    def check_for_fucking_nans(time, sig, other_sigs=[]):
        rnan = np.flatnonzero(np.isnan(sig)) # Check for nan in signal
        if len(rnan) > 0:
            sig = sig[rnan.max()+1:].copy()
            time = time[rnan.max()+1:].copy()
            if other_sigs: other_sigs = [osig[(osig > rnan.max())].copy() - rnan.max() for osig in other_sigs if osig is not None]
        return time, sig, other_sigs
    
    @staticmethod
    def smooth_signal(time, sig, wavelet_size=250, other_sigs=[], kernel='nuttall'):
        """ Smooth a given signal, also checks for nans and replaces them """
        time, sig, other_sigs = DB.check_for_fucking_nans(time, sig, other_sigs=other_sigs)
        if isinstance(kernel, str):
            kernels = {'nuttall': signal.nuttall, }
            wavelet = kernels[kernel](wavelet_size) # create wavelet
            sig_s = signal.convolve(sig, wavelet, mode='same') / wavelet.sum() # smooth signal
        else:
            wavelet = kernel
            sig_s = signal.correlate(sig, wavelet, mode='full') / wavelet.sum() # smooth signal
        nonzero_nan = np.flatnonzero(np.isnan(sig_s))
        if nonzero_nan.any():
            raise Exception("NaN encountered!").with_traceback(nonzero_nan)
        if other_sigs: return other_sigs, time, sig_s
        return time, sig_s

    @staticmethod
    def find_peaks(sig, order=250, method='max'):
        """ Find indices of extrema in 1D data """
        funcs = {'max': signal.argrelmax, 'min': signal.argrelmin}
        return funcs[method](sig, order=order)[0].astype('int32')

    @staticmethod
    def sanity_check_cycle(time, r0, r1, thresh0=1.0, thresh1=7.0):
        cycle_time = time[r1]-time[r0]
        return True if (thresh0 <= cycle_time) and (cycle_time <= thresh1) else False

    @staticmethod
    def normalize(things):
        return (things - things.min()) / (things.max() - things.min())

    @staticmethod
    def mean_insp_exp(time, resp, order=250):
        """ Find the mean inspiration and expiration times (seconds) for a subject """
        min_pks = DB.find_peaks(resp, order=order, method='min') # start of resp cycle
        cycles = [(min_pks[i0], min_pks[i0+1]) for i0 in range(min_pks.size-1)] # (start of insp, end of exp)
        cycle_pks, cycles_use = [], []
        for cycle_ix, cycle in enumerate(cycles):
            try:
                resp_cycle = resp[cycle[0]:cycle[1]]
                cpk = DB.find_peaks(resp_cycle, order=order, method='max')[0]
                cycle_pks.append(cpk)
                cycles_use.append(cycle)
            except IndexError:
                pass
        cycles = list(cycles_use)
        insp_times = [time[cycles[cix][0]:cycles[cix][1]][cycle_pks[cix]] - time[cycles[cix][0]] for cix in range(len(cycles))]
        exp_times = [time[cycles[cix][1]] - time[cycles[cix][0]:cycles[cix][1]][cycle_pks[cix]] for cix in range(len(cycles))]
        mean_insp, mean_exp = np.mean(insp_times), np.mean(exp_times)
        # print("Mean inspiration time: {:f} sec , Mean expiration time: {:f} sec".format(mean_insp, mean_exp))
        return insp_times, exp_times, mean_insp, mean_exp
    
    @staticmethod
    def calc_descriptive(time, resp, beat_ix, bp=None, order=250):
        """ Calculate the descriptive statistics for a subject (ave HR, ave Resp rate, rotation number, ave BP if available """
        HR = np.diff(time[beat_ix]).mean() # Average RR interval
        resp_peaks = DB.find_peaks(resp, order=order, method='max')
        resp_rate = resp_peaks.size / (time[-1] - time[0]) # resp_rate = (nr resp cycles) / (nr sec)
        rot_nr = HR / resp_rate # Rotation number = (HR / resp_rate)
        descriptive = {"hr" : HR, "resp_rate": resp_rate, "rot_nr": rot_nr}
        return descriptive
    
    @staticmethod
    def calc_RSA(time, beat_ix, resp, order=250, calc_var=True):
        """ Calculate the deviation of RR-interval vs respiration cycle phase (from onset of expiration,
        which follows from the literature) """
        resp_peaks = DB.find_peaks(resp, order=order, method='max')
        rri_phase = np.zeros((len(beat_ix),3)).astype('float')
        np.ascontiguousarray(beat_ix, dtype='int32')
        np.ascontiguousarray(time, dtype='float')
        np.ascontiguousarray(resp_peaks)
        np.ascontiguousarray(rri_phase)
        calc_rsa(time, beat_ix, c_int(len(beat_ix)), resp_peaks, c_int(len(resp_peaks)), rri_phase)
        phase = rri_phase[:,0]
        rri = rri_phase[:,1]
        rri_times = rri_phase[:,2]
        phase = phase[phase > -998]
        rri = rri[rri > -998] # remove flagged values (-999)
        rri_times = rri_times[rri_times > -998]
        if calc_var:
            phase_ix = np.argsort(phase)
            rri = rri[phase_ix]
            phase = phase[phase_ix]
            rri_dev = rri - rri.mean()
            popt, pcov = curve_fit(simple_fourier,
                                   phase, rri_dev,
                                   p0=[1.0, 1.0],
                                   sigma=[rri.std(),]*phase.size,
                                   absolute_sigma=True)
            Q = np.sqrt(np.power(popt, 2).sum())
            rri_var = (np.diag(pcov).sum() / np.power(Q, 2)) / np.sqrt(phase.size)
            print("Q : {:.2f} +- {:.4f} sec".format(Q, np.sqrt(rri_var)))
            rri_var = np.array([rri_var,]*phase.size, dtype=float)
            rri_dev = simple_fourier(phase, popt[0], popt[1])
            rri_mean = rri.mean()
            return rri_times, phase, rri, rri_mean, rri_var, rri_dev, Q
        rri_dev = rri - rri.mean()
        return rri_times, phase, rri, rri_dev

    @staticmethod
    def calc_beat_phase(time, beat_ix, resp, order=250, norm=True, retpks=False):
        """ Calculate the phase of heart beats from the onset of inspiration """
        pks_ix = DB.find_peaks(resp, order=order, method='min')
        def find_beats(r0,r1): # Finds the beats that occur within a respiratory cycle
            if DB.sanity_check_cycle(time, r0, r1): # Make sure the cycle is not too short or too long
                chosen_ix = beat_ix[(beat_ix >= r0) & (beat_ix <= r1)]
                cycle_time = time[r1]-time[r0] if norm else 1.0
                return np.array(time[chosen_ix], dtype=float), np.array((time[chosen_ix]-time[r0])/cycle_time, dtype=float)
            return None
        beats_norm = [find_beats(pks_ix[ri0], pks_ix[ri0+1]) for ri0 in range(0, len(pks_ix)-1)]
        beat_times = np.concatenate([bb[0] for bb in beats_norm if bb is not None]).astype(float)
        beats_norm = np.concatenate([bb[1] for bb in beats_norm if bb is not None]).astype(float)
        if retpks: return pks_ix, beats_norm
        return beat_times, beats_norm

    @staticmethod
    def calc_CVC(time, beat_ix, resp, order=250, resp_min=None):
        """ Calculate the latency of heart beats (ms), treating the end of expiration as 0 ms. """
        if not isinstance(resp_min, np.ndarray): resp_min = DB.find_peaks(resp, order=order, method='min')
        resp_max = DB.find_peaks(resp, order=order, method='max')
        beat_ix = beat_ix.flatten().astype('int32')
        beat_latency = np.zeros(len(beat_ix),dtype='float')
        calc_cvc(time, beat_ix, c_int(beat_ix.size), resp_min, c_int(resp_min.size), beat_latency)
        beat_latency = beat_latency[beat_latency > -998] # remove flagged values (-999)
        beat_latency = np.sort(beat_latency) # Sort for use in CDF
        return beat_latency

    @staticmethod
    def calc_DFA(x_signal, y_signal, min_N, max_N):
        """ 
        Calculate the detrended fluctuation analysis (DFA), given a y_signal to be detrended,
        and both min_N and max_N, which together define the range of window sizes.
        """
        y_detrended = (y_signal - y_signal.mean()).astype('float32')
        F = np.zeros((max_N-min_N),dtype='float32')
        calc_dfa(x_signal.astype('float32'), y_detrended, c_uint32(len(y_detrended)), F,
                 c_uint32(min_N), c_uint32(max_N))
        N_range = np.arange(min_N, max_N, dtype='uint32')
        return N_range, F
        

if __name__ == '__main__':
    db = DB(database='fantasia')
