"""
    select_epochs.py: Use a GUI to look at the data and select epochs.
"""


from process import *
from matplotlib.widgets import SpanSelector
import sys

db = DB('fantasia')
db_dir = '/home/robertc/Git/nemati-collab/fantasia'
# for dk, dv in list(filter(lambda dx: dx[0]+'.rng' not in os.listdir(db_dir),db().items())):
for dk, dv in db().items():
    bp = None
    if dv['BP'] is None: f, axarr = plt.subplots(nrows=3, ncols=1, sharex=True)
    time = dv['time']
    ecg = dv['ECG']
    resp = dv['RESP']
    ann = dv['ann']
    if dv['BP'] is not None:
        bp = dv['BP']
        f, axarr = plt.subplots(nrows=4, ncols=1, sharex=True)
    
    ranges = []

    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(time, (xmin, xmax))
        indmax = min(len(time) - 1, indmax)
        thetime = time[indmin:indmax]
        theecg = ecg[indmin:indmax]
        theresp = resp[indmin:indmax]
        if dv['BP'] is not None: thebp = bp[indmin:indmax]
        ranges.append([indmin, indmax])
        span.active = False
        ix_title = '%s: %d indices' % ( dk, int(sum([rr[1]-rr[0] for rr in ranges])) )
        print('\nspan selected.')
        print(ix_title)
        plt.suptitle(ix_title)

    def press(event):
        sys.stdout.flush()
        if event.key == 'x':
            print('\nselect span.')
            span.active = True
        elif event.key == 's':
            np.savetxt(os.path.join('fantasia', dk+'.rng'), np.array(ranges))
            print('\nspans saved.')

    f.canvas.mpl_connect('key_press_event', press)
    
    plt.suptitle(dk)
    for ii, aux in enumerate(ann.aux_note):
        if aux != '': axarr[0].annotate(aux, [time[ann.sample[ii]], 0.25], size=24, color='green')
    axarr[0].set_ylim(0, 0.35)
    axarr[1].plot(time, ecg, 'k')
    axarr[2].plot(time, resp, 'b')
    if bp is not None: axarr[3].plot(time, bp, 'r')
    ax_use = axarr[2]
    if bp is not None: ax_use = axarr[3]
    span = SpanSelector(ax_use, onselect, 'horizontal', useblit=True,
                        span_stays=True, rectprops=dict(alpha=0.5, facecolor='red'))
    span.active = False
    plt.show()
    plt.clf()
    plt.close()
    if len(ranges) > 0:
        fname = os.path.join('fantasia', dk+'.rng')
        if bp is not None: fname = os.path.join('fantasia', dk+'_bp.rng')
        np.savetxt(fname, np.array(ranges))
        print('spans saved.\n')
