from process import *

"""
    process_resp.py: Count HBs, latency of HB within the respiratory cycle.
"""

if __name__ == '__main__':
    db = DB('fantasia')

    plot_ = True
    std_ = False

    filtword = 'f1o01'
    subjects = [kk for kk in db().keys() if (filtword in kk) or (filtword == 'ALL')]
    out_dict = {}

    for nr_bins in [25,]:

        print('Creating histogram with {:d} bins...\n'.format(nr_bins,))

        beats_old = []
        beats_young = []
        lat_old = []
        lat_young = []
        for subject in subjects:
            print(subject+'\n')

            time = db('time')[subject].copy()
            resp = db('RESP')[subject].copy()
            beat_ix = db('beat')[subject].copy()

            other_sigs, time, resp_s = db.smooth_signal(time, resp, wavelet_size=250, other_sigs=[beat_ix,])
            beat_ix = other_sigs[0]

            if plot_:
                fig = plt.gcf()
                ax = plt.gca()
                raw_resp_lines = plt.plot(time, resp, 'b')
                plt.draw()
                smooth_resp_lines = plt.plot(time, resp_s, 'k')
                pks_ix = signal.argrelmin(resp_s, order=250)[0]
                pks_lines = plt.plot(time[pks_ix], resp_s[pks_ix], 'rx')
                other_sigs, time, resp_s2 = db.smooth_signal(time, resp, wavelet_size=250, other_sigs=[beat_ix,])
                smooth2 = plt.plot(time, resp_s2, 'g')
                pks_ix2 = signal.argrelmin(resp_s, order=250)[0]
                plt.plot(time[pks_ix2], resp_s2[pks_ix2], 'rx')
                plt.draw()
                plt.show()

            order = 250
            resp_min, beats_norm = db.calc_beat_phase(time, beat_ix, resp_s, order=order, norm=True, retpks=True)
            beat_latency = db.calc_beat_latency(time, beat_ix, resp_s, order=order, resp_min=resp_min, lthresh=1500.00)

            hist0, bin_edges = np.histogram(beats_norm, bins=nr_bins)
            hist1, bin_edges1 = np.histogram(beat_latency, bins=nr_bins)

            if 'o' in subject:
                beats_old.append(hist0)
                lat_old.append(hist1)
            elif 'y' in subject:
                beats_young.append(hist0)
                lat_young.append(hist1)

        out_dict['old'] = {'beats': np.array(beats_old), 'lat': np.array(lat_old)}
        out_dict['young'] = {'beats': np.array(beats_young), 'lat': np.array(lat_young)}
        
        for ka, va in out_dict.items():
            if ka in ['old', 'young']:
                good_beats = [b for b in va['beats'] if not np.flatnonzero(np.isnan(b)).any()]
                out_dict[ka]['beats'] = np.array(good_beats,dtype=float).reshape((-1,nr_bins))
                out_dict[ka]['lat'] = out_dict[ka]['lat']
                out_dict[ka]['heights'] = out_dict[ka]['beats'].mean(0).tolist()
                out_dict[ka]['errors'] = out_dict[ka]['beats'].std(0).tolist()
                out_dict[ka]['lheights'] = out_dict[ka]['lat'].mean(0).tolist()
                out_dict[ka]['lerrors'] = out_dict[ka]['lat'].std(0).tolist()

                out_dict[ka]['lat'] = out_dict[ka]['lat'].tolist()
                out_dict[ka]['beats'] = out_dict[ka]['beats'].tolist()

        figh = plt.figure()
        plt.suptitle('Heart Beat Count vs Respiratory Phase')
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        width = 0.0003*(100-nr_bins)
        if std_:
            old_bars = plt.bar(bin_centers, out_dict['old']['heights'], width=width,
                           edgecolor='k', color='grey', label='old',
                           yerr=out_dict['old']['errors'],
                           ecolor='red')
            young_bars = plt.bar(bin_centers+width, out_dict['young']['heights'], width=width,
                             edgecolor='k', color='blue', label='young',
                             yerr=out_dict['young']['errors'],
                             ecolor='k')
            figsfn = './figs/{:s}hist{:d}_std.png'.format(filtword,nr_bins)
        else:
            old_bars = plt.bar(bin_centers, out_dict['old']['heights'], width=width,
                           edgecolor='k', color='grey', label='old')
            young_bars = plt.bar(bin_centers+width, out_dict['young']['heights'], width=width,
                             edgecolor='k', color='blue', label='young')
            figsfn = './figs/{:s}hist{:d}.png'.format(filtword,nr_bins)
        print("Saved figure as... {:s}".format(figsfn,))
        plt.xlabel('Respiratory phase')
        plt.ylabel('Heart Beats')
        plt.legend()
        plt.savefig(figsfn)
        plt.close()

        out_dict['old']['bin_centers'] = bin_centers.copy().tolist()
        out_dict['young']['bin_centers'] = (bin_centers+width).copy().tolist()

        figh1 = plt.figure()
        plt.suptitle('Heart Beats E->I Transition')
        bin_centers = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
        width = 0.08*(100-nr_bins)
        if std_:
            old_bars1 = plt.bar(bin_centers, out_dict['old']['lheights'], width=width,
                           edgecolor='k', color='grey', label='old',
                           yerr=out_dict['old']['lerrors'],
                           ecolor='red')
            young_bars1 = plt.bar(bin_centers+width, out_dict['young']['lheights'], width=width,
                             edgecolor='k', color='blue', label='young',
                             yerr=out_dict['young']['lerrors'],
                             ecolor='k')
            figsfn = './figs/{:s}latencyHist{:d}_std.png'.format(filtword,nr_bins)
        else:
            old_bars1 = plt.bar(bin_centers, out_dict['old']['lheights'], width=width,
                           edgecolor='k', color='grey', label='old')
            young_bars1 = plt.bar(bin_centers+width, out_dict['young']['lheights'], width=width,
                             edgecolor='k', color='blue', label='young')
            figsfn = './figs/{:s}latencyHist{:d}.png'.format(filtword,nr_bins)
        print("Saved figure as... {:s}\n".format(figsfn,))
        plt.xlabel('Latency (ms)')
        plt.ylabel('Heart Beats')
        plt.legend()
        plt.savefig(figsfn)
        plt.close()

        out_dict['old']['bin_centers_lat'] = bin_centers.copy().tolist()
        out_dict['young']['bin_centers_lat'] = (bin_centers+width).copy().tolist()

        try:
            dname = 'data{:s}_{:d}.json'.format(filtword,nr_bins)
            with open(dname,'w') as jf:
                json.dump(out_dict, jf)
            print('Data saved as... {:s}\n\n'.format(dname,))
        except Exception as E:
            print(E)

    # plt.show()
