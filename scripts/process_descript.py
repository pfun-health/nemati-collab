from process import *

"""
    process_descript.py: Find the average parameters (descriptive stats) for all subjects
"""

def calc_stats(subject, db):
    out_dict = {}
    time = db('time')[subject].copy()
    resp = db('RESP')[subject].copy()
    beat_ix = db('beat')[subject].copy()
    ave_bp = None
    if db('BP')[subject] is not None: ave_bp = db('BP')[subject].copy().mean()
    ann = db('ann')[subject].aux_note
    other_sigs, time, resp_s = db.smooth_signal(time, resp, wavelet_size=250, other_sigs=[beat_ix,])
    beat_ix = other_sigs[0]
    order = 250
    lthresh = 1500.0
    resp_min, beats_norm = db.calc_beat_phase(time, beat_ix, resp_s, order=order, norm=True, retpks=True)
    resp_count, beat_latency = db.calc_beat_latency(time, beat_ix, resp_s, order=order, resp_min=resp_min, lthresh=lthresh)
    
    hist0, bin_edges = np.histogram(beats_norm, bins=nr_bins)
    hist1, bin_edges1 = np.histogram(beat_latency, bins=nr_bins)

    hist0 = db.normalize(hist0)
    hist1 = db.normalize(hist1)

    out_dict = {'beats': np.array(hist0), 'lat': np.array(hist1)}
    out_dict['beats'] = np.array(out_dict['beats'],dtype=np.float)
    out_dict['lat'] = out_dict['lat'].tolist()
    out_dict['beats'] = out_dict['beats'].tolist()

    lat_peaks = db.find_peaks(np.array(out_dict['lat'],dtype=np.float), order=5)
    lat_modes = bin_edges1[lat_peaks]
    nr_clust = '\nFound {:d} clusters; modes: {:s}'.format(len(lat_modes),str(lat_modes.tolist()))
    print(nr_clust)
    
    HR = np.float(len(beat_ix)) / (time[beat_ix[-1]]-time[beat_ix[0]])
    resp_rate = resp_count / (time[-1]-time[0])
    rotn = HR / resp_rate
    rot_nr = '\nHR: {:f} ; Resp: {:f} ; Rotation number: {:f}'.format(HR,resp_rate,rotn,)
    print(rot_nr)
    if ave_bp: return [HR, resp_rate, ave_bp]
    return [HR, resp_rate]


if __name__ == '__main__':

    repo_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-2])
    print("repo_dir: {}".format(repo_dir,))

    db = DB('fantasia')


    plot_ = True
    filtword = 'ALL'
    subjects = [kk for kk in db().keys() if filtword in kk or (filtword=='ALL')]

    dstats = {}
    BP = []
    HR = []
    resp = []
    for nr_bins in [50,]:
        for si, subject in enumerate(subjects):
            print(str(si)+' : '+subject+'\n')
            cstats = calc_stats(subject, db)
            HR.append(cstats[0])
            resp.append(cstats[1])
            if len(cstats) > 2: BP.append(cstats[2])
    dstats['HR'] = np.mean(HR)
    dstats['BP'] = np.mean(BP)
    dstats['resp'] = np.mean(resp)
    with open(os.path.join(repo_dir, 'stats.json'), 'w') as sfile:
        json.dump(dstats, sfile)
    print("saved stats to stats.json")
            
