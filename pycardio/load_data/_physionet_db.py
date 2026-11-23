import numpy as np
from wfdb import rdheader, rdsamp, rdann, processing
import os

# This package works with wfdb-python==1.3.7

class PhysioNetDB(object):
    """
    
        PhysioNetDB : The database class for loading physionet data (currently only supports the `fantasia` dataset).
    
    arguments: 
    - dname (str) : name of the database you want to access (e.g. 'fantasia')
    - dpath (str) : absolute path to the folder that contains any physionet databases you want to access (e.g. '/home/user/Public')
    - ftype (str) : the file extension for the ECG signal 
    """
    
    def __init__(self, database='fantasia', dpath="", path="", ftype='ecg', **kwargs):
        self.dname = database
        if not dpath: dpath = os.path.expanduser("~/Public/")
        if path and not dpath: dpath = path
        self.dpath = dpath
        self.ftype = 'ecg'
        self.db = self.load_db(self.dname, ftype)
        self._set_data()
        self.stats = {}
        
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

    def set_stats(self, subject, stats):
        """ set the stats for the given subject (see process_data.py)"""
        self.stats.update( { subject : stats } )
        
    def load_db(self, dname, ftype=None):
        """ Load an entire database (including header,
        record, annotation """
        if ftype is None: ftype = self.ftype
        db = {}
        dpath = os.path.join(self.dpath, dname)
        print("\nLoading database: {}...\n".format(dpath,))
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
                data = dict([(ko, PhysioNetDB.get_data(db[ko]['rec'], data_name).copy()) for ko in order])
                for ko in order: # Update the DB dict to include this data name
                    self.db[ko][data_name] = data[ko]
                return data
            elif data_name == 'beat':
                for k, dv in self.db.items():
                    self.db[k]['beat'] = self.db[k]['ann'].sample
            elif data_name == 'time': # Update time in DB dict
                for k, dv in self.db.items():
                    self.db[k]['time'] = PhysioNetDB.calc_time(dv['rec'].siglen, dv['rec'].fs)
                return dict([(dk, self.db[dk]['time'].copy()) for dk in order])
            elif 'RR' in data_name:
                for k, dv in self.db.items():
                    self.db[k]['RR'] = PhysioNetDB.calc_RR(dv['rec'], dv['ann'])
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
        t = PhysioNetDB.calc_time(rec.siglen, rec.fs)
        if ann is None:
            pks_ix, hrs = PhysioNetDB.peaks_hr(rec, thresh=thresh)
            RR = 1.0/hrs
            return t, pks_ix, RR
        pks_ix = ann.sample
        x = PhysioNetDB.get_data(rec, 'ECG')
        RR = np.diff(t[pks_ix])
        RR = np.append(RR, RR[-1])
        return pks_ix, RR
