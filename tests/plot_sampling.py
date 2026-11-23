import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import get_named_colors_mapping
import numpy as np
import os, sys

def covar(ps,params):
    p = np.array([ps[pn] for pn in params]).reshape((len(params),-1))
    stoff, ioff = 1500, 25
    p = p[:,stoff::ioff]
    pix = np.random.shuffle(np.arange(p.shape[1]))
    p = p[:,pix].reshape((len(params),-1))
    aweights = np.array(ps['cost'][stoff::ioff])[pix].flatten().astype('int')
    aweights = np.power(aweights, 2)
    C = np.cov(p, fweights=aweights, aweights=aweights)
    u, s, v = np.linalg.svd(C)
    evecs = v.T
    evals = np.power(s, 2)
    return C, evals, evecs

def plot_sampling(subject_name, database, path, opt_feature): # plot the parameter space

    fnp = os.path.join(path, database)
    fn = os.path.join(fnp, 'paramSpace_{}_{}.npy'.format(opt_feature,subject_name))
    ps = np.load(fn, allow_pickle=True)[()]

    nkeys = ['defaultParams', 'result', 'cost', 'labels', 'opt_label', 'covar', 'cdf']
    params = sorted([pk for pk in ps.keys() if pk not in nkeys and len(ps[pk]) > 0])

    print('\nNumber of samples: {}'.format(len(ps['cost'])))

    C = None
    evals = None
    evecs = None
    try:
        C, evals, evecs = ps['covar'][int(np.argmin(np.array(ps['cost'])))]
        print('\ncovariance: {}\n'.format(repr(C),))
        print("evals: {}\n".format(repr(evals),))
        print("evecs: {}\n".format(repr(evecs),))
    except:
        pass

    # C_all = np.array([cv[0] for cv in ps['covar']])
    # C_all = C_all[not np.any(np.isnan(C_all))]
    # if True: return C_all
    # print(C.shape)

    # C, evals, evecs = covar(ps,params)
    

    small_ms, med_ms, large_ms = 5.0, 7.0, 10.0

    nr_labels = len(list(set(ps['labels'])))
    # base_colors = ['r', 'b', 'g', 'k', 'c', 'y', 'grey', 'm']
    colors = [v for k, v in get_named_colors_mapping().items() if 'xkcd' in k]
    
    print('plotting cost function...\n')
    plt.figure()
    plt.title('cost function')
    ix_start = 0
    for li in range(nr_labels):
        rgb_label = colors[li]
        if np.isclose(ps['labels'], li).any():
            c_cost = np.array(ps['cost'])[np.isclose(ps['labels'], li)]
            if len(c_cost) > 1:
                # c_cost = c_cost[c_cost < 1e5]
                print('label: ', li, 'nr_samples: ', len(c_cost))
                c_range = np.arange(ix_start, ix_start+len(c_cost))
                plt.plot(c_range, c_cost, c=rgb_label)
                ix_start = len(c_cost)+ix_start
    plt.savefig('figs/paramSpace_{}_{}_cost.svg'.format(opt_feature,subject_name,))
    plt.close()
    
    print('plotting parameter space...\n')

    done = []
    for pi0, pn0 in enumerate(sorted(params)):
        for pi1, pn1 in enumerate(sorted(params)):
            
            if pn1 != pn0 and not any(d in [[pn0, pn1], [pn1, pn0]] for d in done):

                print("plotting {} vs {}...".format(pn1,pn0))

                plt.figure(figsize=(15,13))
                plt.title('{} {} sampling'.format(subject_name,opt_feature))
                for li in range(nr_labels):
                    if np.isclose(ps['labels'], li).any():
                        rgb_label = colors[li]

                        glabel = np.isclose(ps['labels'],li)
                        p0_range = np.array(ps[pn0])[glabel]
                        p1_range = np.array(ps[pn1])[glabel]
                        if len(p0_range) > 1:
                            # grange = np.array(ps['cost'])[glabel]
                            # p0_range = p0_range[grange < 1e5]
                            # p1_range = p1_range[grange < 1e5]
                        
                            plt.plot(p0_range, p1_range,
                                     c=rgb_label, ms=small_ms, alpha=0.5, label='sample')
                            plt.plot(p0_range[0], p1_range[0], 'g.', ms=med_ms, label='start')
                            plt.plot(p0_range[-1], p1_range[-1], 'r.', ms=med_ms, label='end')

                plt.plot(ps['result'].x[pi0], ps['result'].x[pi1], 'm*',
                         ms=large_ms, label='optimal')

                # Plot optimum
                optx, opty = ps['result'].x[pi0], ps['result'].x[pi1]

                if evals is not None:    
                    dx0, dy0 = evals[pi0]*evecs[pi0,pi0], evals[pi0]*evecs[pi1,pi0]
                    dx1, dy1 = evals[pi1]*evecs[pi0,pi1], evals[pi1]*evecs[pi1,pi1]

                    plt.plot([optx-dx0, optx, optx+dx0], [opty-dy0, opty, opty+dy0], 'k')
                    plt.plot([optx-dx1, optx, optx+dx1], [opty-dy1, opty, opty+dy1], 'k')

                plt.xlabel(pn0)
                plt.ylabel(pn1)

                ax = plt.gca()
                ax.ticklabel_format(style='sci',scilimits=(-4,-4))

                plt.legend(loc=(1.04,0),framealpha=0.5)

                fname = 'figs/paramSpace_{}_{}v{}_{}.svg'.format(opt_feature,pn1,pn0,subject_name,)
                plt.savefig(fname)

                zoom_f = 5e-4
                plt.xlim([optx-zoom_f, optx+zoom_f])
                plt.ylim([opty-zoom_f, opty+zoom_f])

                fname2 = 'figs/paramSpace_{}_{}v{}_{}_zoomed.svg'.format(opt_feature,pn1,pn0,subject_name,)
                plt.savefig(fname2)
                
                plt.close()
                    
                print("plotted {} vs {}.\n".format(pn1,pn0))
                
                done.append([pn0, pn1])
                
    print('done plotting parameter space.')

    print('optimal: {}'.format(str(ps['result'].x),))
    
if __name__=='__main__':
    subject = 'f1y02'
    database = 'fantasia'
    path = os.path.expanduser('~/Public')
    opt_feature = 'ave_resp'
    
    if len(sys.argv) > 2:
        subject, opt_feature = sys.argv[1:]

    plot_sampling(subject, database, path, opt_feature)
