try:
    from .db import *
except ModuleNotFoundError:
    from db import *

def compute_stats(dat, subject, order=250):
    """ Computes the relevant statistics for an individual subject and returns the database object, from which the stats attribute is accessed as a dictionary (e.g. dat.stats ). 
    
    Input arguments:
      dat (DB) : the database object
      subject (str) : the subject you want to select (e.g. "f1y01")
      path [optional] (str) : Absolute path to the folder containing the database (e.g. "/home/user/Public")
      order [optional] (int) : The size of the smoothing kernel in indices; (e.g. for the fantasia dataset, the sampling rate is 250Hz, so a smoothing kernel of order 250 is 1 second of data.
    
    Return:
      dat (class instance) : database class instance containing the stats dict
      stats (dict) : Dictionary containing all the statistics and smoothed signals
    
    stats is organized like so: 
    (In the list below, note that > key : description)
    
    - subject: ID of the subject
    - rsa: 
     > resp_cycle : the normalized respiration phase vector
     > rri : the RR intervals (seconds) for each pair of heartbeats
     > rri_dev : the deviation (RRI - mean(RRI)) of RRI from the mean
    - cvc:
     > beat_latency : contains the beat latency from the beginning of inspiration (beat_time - inspiration_time) in seconds
     > cdf : contains the cumulative distribution function for CVC (sorted from lowest latency to highest)
    - signals:
     > time : Time vector that corresponds to the smoothed signals (seconds)
     > tv : Smoothed tidal volume signal
     > hb : Heartbeat indices that correspond to the time vector
    - ave_resp:
    > exp_time : Mean time for expiration to take place (sec)
    > insp_time : Meant time for inspiration to take place (sec)
    - descriptive:
    > hr : Mean RR interval (s)
    > resp_rate : Mean respiration rate (breaths per sec)
    > rot_nr : Rotation number (HR / resp_rate)
    > bp : Mean blood pressure (not sure of units, only available in a subset of subjects)
    """
    
    time = dat('time')[subject].copy()[:] # Time vector
    resp = dat('RESP')[subject].copy()[:] # Tidal volume vector
    # np.savetxt("/home/rocapp/Documents/resp_f2y03.txt",resp)
    # print('saved resp')
    beat_ix = dat('beat')[subject].copy()[:] # Heartbeat indices
    if dat('BP')[subject] is not None: bp = dat('BP')[subject].copy()
    else: bp = None

    other_sigs = [beat_ix,]
    if bp is not None: other_sigs.append(bp)
    other_sigs, time, resp_s = dat.smooth_signal(time, resp, wavelet_size=250, other_sigs=other_sigs) # Smooth signal and remove NaNs
    beat_ix = other_sigs[0].astype('int32').flatten()
    if bp is not None: bp = other_sigs[1]

    # RSA
    rri_times, phase, rri, rri_mean, rri_var, rri_dev, Q = dat.calc_RSA(time, beat_ix, resp_s, order=order) # calculate the deviation of R-R interval wrt the respiration cycle
    rri_times = remove_flagged(rri_times)
    phase = remove_flagged(phase)
    rri = remove_flagged(rri)
    rri_mean = remove_flagged(rri_mean)
    rri_var = remove_flagged(rri_var)
    rri_dev = remove_flagged(rri_dev)
    
    # CVC
    beat_latency = dat.calc_CVC(time, beat_ix, resp_s, order=order) # calculate the time of each heart beat since the onset of inspiration
    beat_latency = remove_flagged(beat_latency)
    cvc_cdf =  np.linspace(0, 1, beat_latency.size) # calculate the CDF of the normalized beat latency since the onset of inspiration
    
    insp_times, exp_times, mean_insp, mean_exp = dat.mean_insp_exp(time, resp_s, order=order) # Mean inspiration, expiration
    descriptive = dat.calc_descriptive(time, resp_s, beat_ix, bp=bp, order=order)

    stats = {
        "subject": subject,
        "rsa": {
            "rri_times": rri_times,
            "resp_cycle": phase,
            "rri": rri,
            "rsa_mean": rri_mean,
            "rsa_var": rri_var,
            "rri_dev": rri_dev,
            "Q": Q},
        "cvc": {"beat_latency": beat_latency, "cdf": cvc_cdf},
        "ave_resp": {"exp_time": mean_exp, "insp_time": mean_insp,
                     "exp_times": exp_times, "insp_times": insp_times},
        "descriptive": descriptive,
        "signals": {"time": time, "tv": resp_s, "hb": beat_ix}
    }
    if bp is not None: stats["signals"].update({"bp": bp})
    dat.set_stats(subject,stats) # Set the stats object
    return dat

def save_stats(dat, path=None, overwrite=False):
    """ save the stats dictionary to a .npy (numpy) file.
    Input arguments
    dat (dict) : stats generated from compute_stats function
    OR
    dat (DB) : database object containing all stats
    path [optional] (str) : 
        The absolute path to the database folder
        (default is /home/$USER/Public/fantasia)
        The filename is generated programmatically : subject_name.npy,
        where subject_name is the subject ID (e.g. 'f1y01')
    """
    if isinstance(dat, DB):
        dpath = os.path.join(dat.dpath, dat.dname)
        for subject in dat.stats.keys():
            subject_path = '{}.npy'.format(subject,)
            path = os.path.join(dpath, subject_path)
            if not os.path.isfile(path) or overwrite:
                np.save(path, dat.stats[subject])
                print("saved stats as {}\n".format(path,))
            else:
                print("skipped {} because it already existed\n".format(path,))
        return 0
    elif isinstance(dat, dict):
        stats = dat
        subject_path = '{}.npy'.format(stats['subject'],)
        if not path:
            dpath = os.path.expanduser("~/Public")
            path = os.path.join(dpath, subject_path)
        else:
            path = os.path.join(path, subject_path)
        if not os.path.isfile(path) or overwrite:
            np.save(path, stats)
            print("saved stats as {}".format(path,))
        else:
            print("skipped {} because it already existed\n".format(path,))
        return 0

def compute_all_stats(database, path=None, save=True, overwrite=False, exclude=[]):
    """ Computes all stats for the given database 
    database (str) : name of database (e.g. 'fantasia')
    path [optional] (str) : path that contains database (e.g. '/home/user/Public')
    save [optional] (bool) : whether or not to save all the stats as .npy in the database folder
    overwrite [optional] (bool) : whether or not to overwrite saved .npy files in the database folder
    exclude [optional] (list) : a list of subjects to exclude if any, e.g. ['f2y09', 'f2y03']
    """
    if not path:
        path = os.path.expanduser("~/Public")
        dpath = os.path.join(path, 'fantasia')
    else:
        dpath = os.path.join(path, database)
    subjects = list(set([os.path.splitext(f)[0] for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))\
                         and '_' not in f\
                         and (overwrite or not os.path.isfile(os.path.join(dpath, os.path.splitext(f)[0]+'.npy')))\
                         and os.path.splitext(f)[0] not in exclude
                         ]))
    dat = DB(database=database, dpath=path) # Database object
    for subject in list(sorted(subjects)): # Compute the stats for each subject and update the dat.stats dictionary object
        print("Computing stats for... {}".format(subject,))
        dat = compute_stats(dat, subject, order=250)
        if save: save_stats(dat.stats[subject],path=dpath,overwrite=overwrite)
        print("Done with {}.\n".format(subject,))
    return dat

def remove_flagged(signal):
    """ Remove any flagged values (-999) """
    signal = signal[signal > -998]
    return signal

def get_stats_array(all_stats, stat_key):
    """ get a 1D numpy array of a particular stat from each subject (in alphanumeric order of subject ID) """
    subjects = all_stats.keys()
    stat_list = []
    for subject in list(sorted(subjects)):
        if isinstance(stat_key, str):
            stat = all_stats[subject][stat_key]
        elif isinstance(stat_key, list):
            for ki in range(len(stat_key)):
                sk = stat_key[ki]
                if ki == 0: stat0 = all_stats[subject][sk]
                elif ki == len(stat_key)-1: stat = stat0[sk]
                else: stat0 = stat0[sk]
        try:
            if not np.isnan(stat):
                stat_list.append(stat)
            elif np.isnan(stat):
                print("{} has a nan value for {}".format(subject, str(stat_key)))
        except (TypeError, ValueError) as E:
            stat_list.append(stat)
    return np.array(stat_list)

def get_averages(all_stats):
    mean_hr = get_stats_array(all_stats, ["descriptive", "hr"])
    mean_resp = get_stats_array(all_stats, ["descriptive", "resp_rate"])
    rot_nr = get_stats_array(all_stats, ["descriptive", "rot_nr"])
    ave_exp = get_stats_array(all_stats, ["ave_resp", "exp_time"])
    ave_insp = get_stats_array(all_stats, ["ave_resp", "insp_time"])
    print("Mean HR: {:f} +- {:f} bps".format(mean_hr.mean(),mean_hr.std()))
    print("Mean Resp Rate: {:f} +- {:f} breaths/sec".format(mean_resp.mean(),mean_resp.std()))
    print("Mean rotation number: {:f} +- {:f} beats/breath".format(rot_nr.mean(),rot_nr.std()))
    print("Mean expiration time: {:f} +- {:f} sec".format(ave_exp.mean(),ave_exp.std()))
    print("Mean inspiration time: {:f} +- {:f} sec".format(ave_insp.mean(),ave_insp.std()))
    return mean_hr, mean_resp, rot_nr, ave_exp, ave_insp

def exclude_range(excludes, group, age, erange):
    excludes.extend(['f{:d}{:s}{:02d}'.format(group,age,di,) for di in erange])
    return excludes

if __name__ == '__main__':
    database = 'fantasia'
    ftype = 'ecg'
    path = os.path.expanduser("~/Public")

    excludes = []
    
    # dat is a database object
    dat = compute_all_stats(database, path=path,
                            save=True, overwrite=True,
                            exclude=excludes)
    
    computed_stats = dict(dat.stats) # Dictionary of all stats, keys are subjects
    averages = get_averages(computed_stats) # Get certain average values and print them
    
    # all_stats = load_all_stats(database=database, path=path) # Dictionary of all stats, keys are subjects
    # print(all_stats.keys()) # Lists all the subject IDs
    # averages = get_averages(all_stats) # Get certain average values and print them
