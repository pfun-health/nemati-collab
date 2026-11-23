from process import *

"""
    process_bp.py: Find how BP affects the HB latency within a resp cycle
"""

def hist_from_ranges(subject, nr_bins, smin, smax, db):
    out_dict = {}
    filtword = 'BPrng'
    
    time = db('time')[subject].copy()[smin:smax]
    resp = db('RESP')[subject].copy()[smin:smax]
    beat_ix = db('beat')[subject].copy()
    beat_ix = beat_ix[(beat_ix >= smin) & (beat_ix < smax)]
    beat_ix = beat_ix - smin
    ave_bp = None
    if db('BP')[subject] is not None: ave_bp = db('BP')[subject].copy()[smin:smax].mean()
    ann = db('ann')[subject].aux_note[smin:smax]
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

    # print('\nNotes: {:s}'.format(str(list(set(ann))),))
    
    # Plot Phase
    figh = plt.figure()
    plt.suptitle("Cardioventilatory coupling")
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    width = 0.0003*(100-nr_bins)
    bars = plt.bar(bin_centers, out_dict['beats'], width=width,
                   edgecolor='k', color='k')
    figsfn = './figs/{:s}hist_{:s}_{:d}_{:d}-{:d}_{:f}.png'.format(filtword,subject,nr_bins,smin,smax,lthresh)
    print("Saved figure as... {:s}".format(figsfn,))
    plt.xlabel('Respiratory phase (I -> E)')
    plt.ylabel('Heart Beats')
    plt.savefig(figsfn)
    plt.close()

    out_dict['bin_centers'] = bin_centers.copy().tolist()

    # Plot latency
    figh1 = plt.figure()
    plt.suptitle('Cardioventilatory coupling')
    bin_centers = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
    width = 0.5*(100-nr_bins)
    bars1 = plt.bar(bin_centers, out_dict['lat'], width=width,
                       edgecolor='k', color='r')
    figsfn = './figs/{:s}latencyHist_{:s}_{:d}_{:d}-{:d}_{:f}.png'.format(filtword,subject,nr_bins,smin,smax,lthresh)
    print("Saved figure as... {:s}\n".format(figsfn,))
    plt.xlabel('Latency to inspiration (ms)')
    plt.ylabel('Heart Beats')
    plt.savefig(figsfn)
    plt.close()

    out_dict['bin_centers_lat'] = bin_centers.copy().tolist()

    try:
        dname = 'data{:s}_{:s}_{:d}_{:d}-{:d}_{:f}.json'.format(filtword,subject,nr_bins,smin,smax,lthresh)
        with open(dname,'w') as jf:
            json.dump(out_dict, jf)
        print('Data saved as... {:s}\n\n'.format(dname,))
    except Exception as E:
        print(E)


if __name__ == '__main__':
    db = DB('fantasia')

    plot_ = True
    filtword = 'ALL'
    subjects = [kk for kk in db().keys() if filtword in kk or (filtword=='ALL')]
    subjects = [kk for kk in subjects if db()[kk]['BP'] is not None]
    subjects = [kk for kk in subjects if os.path.isfile('./fantasia/{:s}_bp.rng'.format(kk,))]
    bp_ranges = [np.loadtxt('fantasia/{:s}_bp.rng'.format(kk,),ndmin=2).astype('int') for kk in subjects]

    for nr_bins in [50,]:
        beats_old = []
        beats_young = []
        for si, subject in enumerate(subjects):
            print(str(si)+' : '+subject+'\n')
            
            ranges = bp_ranges[si]
            smin0, smax0 = ranges[0,0], ranges[0,1]
            smin1, smax1 = ranges[1,0], ranges[1,1]
    
            hist_from_ranges(subject, nr_bins, smin0, smax0, db)
            hist_from_ranges(subject, nr_bins, smin1, smax1, db)
