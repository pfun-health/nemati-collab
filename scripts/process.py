import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from wfdb import rdheader, rdsamp, rdann, processing
# from wfdb import io
import os, json, pickle

# Works with wfdb-1.3.7

"""
    process.py: Contains the DB (database) class for organizing Physionet data.
"""

class DB(object):
    def __init__(self, dname, ftype='ecg'):
        self.dname = dname
        self.ftype = 'ecg'
        self.db = self.load_db(dname, ftype)
        self._set_data()
        
    def __call__(self, dtype='db', dmode='dict'):
        """ 
        Return a dictionary containing all the headers,
        records, and annotations in the database (default)
        OR 
        a dictionary containing 'dtype' data (e.g. dtype='ECG') 
        """
        if dtype == 'db': return self.db
        elif dmode == 'dict': return self.data(dtype)
        elif dmode == 'arr': return self.data_arr(dtype)

    def list_dbs(self):
        return io.get_dbs()

    def pickle_db(self, dpfn):
        print('Pickling the database instance...')
        with open(dpfn, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    def load_db(self, dname, ftype=None):
        """ Load an entire database (including header,
        record, annotation """
        if ftype is None: ftype = self.ftype
        db = {}
        repo_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-2])
        dpath = os.path.join(repo_dir, dname)
        print("database: {}".format(dpath,))
        if os.path.isdir(dpath):
            for fn in [ff for ff in os.listdir(dpath) if '.hea' in ff]:
                rname = os.path.join(dpath, fn.split('.')[0])
                head, rec, ann, rng = self.load_all(rname, ftype=ftype)
                db[rname.split('/')[-1]] = {'head': head, 'rec': rec, 'ann': ann, 'rng': rng}
        return db

    def load_all(self, fn, ftype=None):
        """ Load header, record, annotation, and rng labels """
        if ftype is None: ftype = self.ftype
        # print(fn)
        head = rdheader(fn)
        rec = rdsamp(fn)
        ann = rdann(fn, ftype)
        try:
            rng = np.loadtxt(fn+'.rng', ndmin=2).astype('int')
        except (FileNotFoundError, OSError) as e:
            rng = np.array([], dtype='int')
        return head, rec, ann, rng

    @staticmethod
    def get_data(rec, data_name, retix=False):
        """ Get a specific type of data from a record, e.g. ECG """
        rec_ix = rec.signame.index(data_name)
        data = rec.p_signals[:,rec_ix]
        if retix is True: return rec_ix, data
        return data

    def _set_data(self):
        """ Set all data in the self.db dictionary """
        db = self.db
        for dk, dv in db.items():
            nlist = list(dv['rec'].signame)
            nlist.append('time')
            nlist.append('rng')
            nlist.append('RR')
            nlist.append('beat')
            for sn in nlist:
                if sn not in dv.keys():
                    self.data(sn)
        return self.db

    def data(self, data_name, dmode='dict'):
        """ Get/set a specific type of data from every record in the db """
        if not any([dvk==data_name for dv in self.db.values() for dvk in dv.keys()]):
            db = self.db
            order = sorted(list(db.keys()))
            if all(list(map(lambda dx: data_name in dx['rec'].signame, list(db.values())))):
                data = dict([(ko, DB.get_data(db[ko]['rec'], data_name).copy()) for ko in order])
                for ko in order: # Update the DB dict to include this data name
                    self.db[ko][data_name] = data[ko]
                return data
            elif data_name == 'beat':
                for k, dv in self.db.items():
                    self.db[k]['beat'] = self.db[k]['ann'].sample
            elif data_name == 'time': # Update time in DB dict
                for k, dv in self.db.items():
                    self.db[k]['time'] = DB.calc_time(dv['rec'].siglen, dv['rec'].fs)
                return dict([(dk, self.db[dk]['time'].copy()) for dk in order])
            elif 'RR' in data_name:
                for k, dv in self.db.items():
                    self.db[k]['RR'] = DB.calc_RR(dv['rec'], dv['ann'])
                if data_name == 'RR':
                    return dict([(dk, self.db[dk]['RR']) for dk in order])
                elif 'rng_arr' in data_name:
                    return self.get_rng(data_name=data_name.split('rng')[0].replace('_',''))
            elif 'rng_arr' in data_name:
                return self.get_rng(data_name=data_name.split('rng')[0].replace('_',''))
        elif 'rng_arr' in data_name:
            return self.get_rng(data_name=data_name.split('rng')[0].replace('_',''))
        data = {}
        for dk, dv in self.db.items(): # For data that is not available for all participants (e.g. BP)
            if data_name in dv.keys():
                data[dk] = dv[data_name]
            elif data_name in dv['rec'].signame:
                data[dk] = dv['rec'].p_signals[:,dv['rec'].signame.index(data_name)]
                self.db[dk][data_name] = data[dk]
            else:
                data[dk] = None
                self.db[dk][data_name] = None
        return data

    def data_arr(self, data_name):
        names = []
        data = []
        for k, v in self.data(data_name).items():
            names.append(k)
            if data_name=='RR':
                data.append(v[1])
            else:
                data.append(v)
        return names, np.array(data).copy()

    def get_rng(self, data_name='', data=None, names=None):
        rng_key = data_name+'_'+'rng_arr'
        if any([dk for dk in self.db.keys() if rng_key in self.db[dk].keys()]):
            return dict([(dk, dv[rng_key]) for dk, dv in self.db.items()])
        if data_name != '' and data is None:
            names, data = self.data_arr(data_name)
        for ii, n in enumerate(names):
            if rng_key not in self.db[n].keys():
                self.db[n][rng_key] = []
                if self.db[n]['rng'] is not None:
                    for intr in self.db[n]['rng']:
                        rstart, rend = intr[0], intr[1]
                        if 'RR' == data_name:
                            samp = self.db[n]['ann'].sample.copy()
                            rarr = data[ii][(samp >= rstart) & (samp < rend)].copy()
                        elif 'beat' == data_name:
                            rarr = data[ii][(data[ii] >= rstart) & (data[ii] < rend)].copy()
                        else:
                            rarr = data[ii][rstart:rend].copy()
                        self.db[n][rng_key].append(rarr)
        return self.get_rng(data_name=data_name, data=data, names=names)
    
    @staticmethod
    def calc_time(N, fs):
        """ Give time vector in seconds """
        return np.linspace(0, N-1, N)/fs

    @staticmethod
    def short_len(data):
        """ Find the shortest array in a list of arrays """
        ix = np.argmin([len(d) for d in data])
        return len(data[ix])

    @staticmethod
    def apply_cutoff(data, cutoff):
        return np.array([d[0:cutoff] for d in data])

    @staticmethod
    def peaks_hr(rec, thresh=9.55):
        """ 
        finds peak indices and calculates heart rate
        *** NOTE *** This appears not to work as well as the
        corrected annotations from fantasia.
        """
        ix, x = get_data(rec, 'ECG', retix=True)
        peak_indices = processing.gqrs_detect(rec.adc()[:,ix], fs=rec.fs, adcgain=rec.adcgain[ix],
                                                   adczero=rec.adczero[ix], threshold=thresh)
        hrs = processing.compute_hr(siglen=x.shape[0], peak_indices=peak_indices, fs=rec.fs)
        return peak_indices, hrs

    @staticmethod
    def calc_RR(rec, ann=None, thresh=9.55):
        """ Returns the location of peaks and RR intervals """
        t = DB.calc_time(rec.siglen, rec.fs)
        if ann is None:
            pks_ix, hrs = DB.peaks_hr(rec, thresh=thresh)
            RR = 1.0/hrs
            return t, pks_ix, RR
        pks_ix = ann.sample
        x = DB.get_data(rec, 'ECG')
        RR = np.diff(t[pks_ix])
        RR = np.append(RR, RR[-1])
        return pks_ix, RR

    @staticmethod
    def smooth_signal(time, sig, wavelet_size=250, other_sigs=[]):
        """ Smooth a given signal, also checks for nans and replaces them """
        rnan = np.flatnonzero(np.isnan(sig)) # Check for nan in signal
        if len(rnan) > 0:
            sig = sig[rnan.max()+1:].copy()
            time = time[rnan.max()+1:].copy()
            if other_sigs: other_sigs = [osig[(osig > rnan.max())].copy() - rnan.max() for osig in other_sigs]
        wavelet = signal.nuttall(wavelet_size) # create wavelet
        sig_s = signal.convolve(sig, wavelet, mode='same') / wavelet.sum() # smooth signal
        nonzero_nan = np.flatnonzero(np.isnan(sig_s))
        if nonzero_nan.any():
            raise Exception("NaN encountered!").with_traceback(nonzero_nan)
        if other_sigs: return other_sigs, time, sig_s
        return time, sig_s

    @staticmethod
    def find_peaks(sig, order=250, method='max'):
        """ Find indices of extrema in 1D data """
        funcs = {'max': signal.argrelmax, 'min': signal.argrelmin}
        return funcs[method](sig, order=order)[0]

    @staticmethod
    def sanity_check_cycle(time, r0, r1, thresh0=1.0, thresh1=7.0):
        cycle_time = time[r1]-time[r0]
        return True if (thresh0 <= cycle_time) and (cycle_time <= thresh1) else False

    @staticmethod
    def normalize(things):
        return (things - things.min()) / (things.max() - things.min())

    @staticmethod
    def calc_RSA(time, beat_ix, resp, order=250):
        """ Calculate the deviation of RR-interval vs respiration cycle phase """
        pks_ix = DB.find_peaks(resp, order=order, method='min')
        def find_beats(r0,r1): # Finds the beats that occur within a respiratory cycle
            if DB.sanity_check_cycle(time, r0, r1): # Make sure the cycle is not too short or too long
                chosen_ix = beat_ix[(beat_ix >= r0) & (beat_ix <= r1)]
                beat_times = time[chosen_ix]
                cycle_time = time[r1]-time[r0]
                rri_phase = [[(((beat_times[ti] + beat_times[ti+1])/2) - time[r0])/cycle_time, beat_times[ti+1]-beat_times[ti]]\
                            for ti in range(len(beat_times)-1)]
                phases = [b[0] for b in rri_phase]
                rri = [b[1] for b in rri_phase]
                return phases, rri
        rri_phase = [find_beats(pks_ix[ri0], pks_ix[ri0+1]) for ri0 in range(0, len(pks_ix)-1)]
        rri = np.array([b[1] for bb in rri_phase for b in bb]).astype(float)
        phase = np.array([b[0] for bb in rri_phase for b in bb]).astype(float)
        rri_dev = rri - rri.mean()
        return phase, rri_dev

    @staticmethod
    def calc_beat_phase(time, beat_ix, resp, order=250, norm=True, retpks=False):
        """ Calculate the phase of heart beats from the onset of inspiration """
        pks_ix = DB.find_peaks(resp, order=order, method='min')
        def find_beats(r0,r1): # Finds the beats that occur within a respiratory cycle
            if DB.sanity_check_cycle(time, r0, r1): # Make sure the cycle is not too short or too long
                chosen_ix = beat_ix[(beat_ix >= r0) & (beat_ix <= r1)]
                cycle_time = time[r1]-time[r0] if norm else 1.0
                return np.array((time[chosen_ix]-time[r0])/cycle_time, dtype=float)
            return None
        beats_norm = [find_beats(pks_ix[ri0], pks_ix[ri0+1]) for ri0 in range(0, len(pks_ix)-1)]
        beats_norm = np.concatenate([bb for bb in beats_norm if bb is not None]).astype(float)
        # beats_norm = DB.normalize(beats_norm)
        if retpks: return pks_ix, beats_norm
        return beats_norm

    @staticmethod
    def calc_beat_latency(time, beat_ix, resp, order=250, lthresh=200.0, resp_min=None):
        """ Calculate the latency of heart beats (ms), treating the end of expiration as 0 ms. """
        sthresh = int((lthresh/1000.0)*250)
        if not isinstance(resp_min, np.ndarray): resp_min = DB.find_peaks(resp, order=order, method='min')
        resp_max = DB.find_peaks(resp, order=order, method='max')
        def check_beats(bb): return [bl for bl in bb if abs(bl) <= lthresh]
        def find_beats(r0,r1): # Finds the beats, peaks, between mins of each resp cycle
            beat_times = time[beat_ix]
            min_times = time[resp_min]
            chosen_beats = beat_times[(beat_times >= time[r0]) & (beat_times <= time[r1])]
            min_time = min_times[(min_times >= time[r0]) & (min_times <= time[r1])]
            if len(min_time) > 0 and DB.sanity_check_cycle(time, r0, r1):
                min_time = min_time[0]
                min_ix = np.where(time == min_time)[0][0]
                if min_ix > sthresh:
                    resp_cycle = resp[min_ix-sthresh:min_ix+sthresh].copy()
                    print("oops: {:d}, {:d}, {:d}".format(resp_cycle.shape[0],min_ix,sthresh)) if len(resp_cycle) != sthresh*2 else 0
                    if len(resp_cycle) >= sthresh*2:
                        return ( np.array((chosen_beats-min_time)*1000.0, dtype=float), resp_cycle )
            return None, None
        beat_latency = [find_beats(resp_max[ri0], resp_max[ri0+1]) for ri0 in range(0, len(resp_max)-1)]
        beat_latency = [(check_beats(bb), ri) for bb, ri in beat_latency if bb is not None]
        nr_resp_cycles = np.sum(np.array([1.0 for bb, ri in beat_latency],dtype=float))
        beat_latency = np.concatenate([bb for bb, ri in beat_latency]).astype(float)
        # beat_latency = DB.normalize(beat_latency)
        if len(beat_latency) > 0:
            print(" Mean beat latency: {:f} +/- {:f}  ;  Range: {:f}\n".format(
                    beat_latency.mean(),beat_latency.std(),beat_latency.max()-beat_latency.min()))
        return nr_resp_cycles, beat_latency

if __name__ == '__main__':
    db = DB('fantasia')
    
