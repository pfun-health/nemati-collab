import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio
import ctypes as ct

def plot_data(subject_name, database, path, opt_feature): # plot the relevant data
    print('plotting data...')

    stats = pycardio.load_data.load_stats(subject_name, database, path=path)
    signals = stats['signals']

    out = []

    if opt_feature == 'cvc':
        stats['cvc'].update(stats['ave_resp'])
        stats['cvc'].update(stats['rsa'])
        out.append(stats['cvc'])

    if opt_feature in ['rsa', 'cvc']:
        plt.figure(figsize=(12,12),dpi=300)
        rsa = stats['rsa']
        phase = rsa['resp_cycle'] # Get the respiratory cycle (normalized as phase 0 to 1)
        rri_dev = rsa['rri_dev'] # Get the RRI deviation (in seconds)

        data_rri_str = 'data rri: '+ str(rsa['rri'].mean()) + " +- " + str(rsa['rri'].std()) + " sec" + '\n'
        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([data_rri_str,])
            
        print(data_rri_str)
    
        plt.title('RSA {}'.format(subject_name,))
        
        plt.plot(phase, rri_dev, 'k', label='data_mean', lw=3.0) # plot data interp

        data_std = np.sqrt(rsa['rsa_var'])
        plt.fill_between(phase, rri_dev-data_std, rri_dev+data_std,
                         alpha=0.75, lw=0, facecolor='r', label='data_std')

        plt.hlines(y=0, xmin=0, xmax=1, colors='k', linestyles='--', lw=0.5, alpha=0.5)
        
        plt.xlabel('Phase [Expiration -> Expiration]')
        plt.ylabel('RRI deviation (sec)')

    if opt_feature in ['ave_resp', 'cvc']:
        ave_resp = stats['ave_resp']
        exp_time = ave_resp['exp_time']
        exp_std = np.std(ave_resp['exp_times'])
        insp_time = ave_resp['insp_time']
        insp_std = np.std(ave_resp['insp_times'])
        data_insp_str = "(Data) Inspiration duration: {:.01f} +- {:.02f} sec\n".format(insp_time,insp_std)
        data_exp_str = "(Data) Expiration duration: {:.01f} +- {:.02f} sec\n".format(exp_time,exp_std)
        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([data_insp_str, data_exp_str])
        print(data_insp_str)
        print(data_exp_str)
        print("\n\n")

    if opt_feature in ['rri_ave',]:
        rsa = stats['rsa']
        
        data_rri_str = 'data rri: '+ str(rsa['rri'].mean()) + " +- " + str(rsa['rri'].std()) + " sec" + '\n'
        
        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([data_rri_str,])
            
        print(data_rri_str)

    out.append(signals['time'])
    
    return out


def plot_result(subject_name, database, path, opt_feature, data_out): # plot the model result next to the data
    fnp = os.path.join(path, database)
    fn = os.path.join(fnp, 'paramSpace_{}_{}.npy'.format(opt_feature,subject_name))
    ps = np.load(fn, allow_pickle=True)[()]

    print('\nplotting result...\n')

    nkeys = ['defaultParams', 'result', 'cost', 'labels', 'opt_label', 'covar', 'cdf']
    pnames = [pk for pk in sorted([pp for pp in ps.keys() if pp not in nkeys and len(ps[pp]) > 0])[:len(ps['result'].x)]]
    params = dict([(pnames[i], ps['result'].x[i]) for i in range(len(ps['result'].x))])

    with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
        infof.writelines([repr(params)+"\n",])
    print(repr(params)+"\n")
    
    m = pycardio.setupModel()
    p = pycardio.setParams(dParams=ps['defaultParams'], **params)
    
    time_scale = 1000
    max_time =  120 #int(250*(data_out[-1][-1] - data_out[-1][0])) # seconds
    vlen = time_scale * max_time
    ndim = 12
    solution = np.zeros((vlen, ndim+1))
    end_time = ct.c_double(max_time)

    m.integrate(ct.c_int(vlen), ct.c_int(ndim), solution, end_time, ct.byref(p))    
    tt = np.arange(vlen)/float(time_scale)

    model_output = pycardio.ModelOutput(tt, solution)

    tp = model_output('time')
    sp = model_output()

    if opt_feature == 'cvc': cvc_data = data_out[0]

    if opt_feature in ['rsa', 'cvc']:
        model_rsa = model_output('rsa')
        rphase = model_rsa['resp_cycle']
        rri_dev = model_rsa['rri_dev']
        
        rri_str = 'model rri ' + str(model_rsa['rri_ave']) + '\n'
        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([rri_str,])

        print(rri_str)
    
        plt.xlim([0, 1])
        plt.ylim([-0.1, 0.1])
 
        plt.plot(rphase, rri_dev, 'b.', label='model', ms=3.0)

        plt.legend(loc='best', framealpha=0.5)
        plt.savefig('figs/{}_{}_rsadev_result.svg'.format(opt_feature,subject_name))
        plt.close()

    if opt_feature in ['rri_ave',]:
        rri = model_output('rri_ave')
        rri_str = 'model rri ' + str(rri) + '\n'
        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([rri_str,])

        print(rri_str)

    if opt_feature in ['ave_resp', 'cvc']:
        aresp = model_output('ave_resp')
        insp_time = aresp['insp_time']
        exp_time = aresp['exp_time']
        insp_str = "(Model) Inspiration duration: {:.01f} sec\n".format(insp_time,)
        exp_str = "(Model) Expiration duration: {:.01f} sec\n".format(exp_time,)
        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([insp_str, exp_str])
        print(insp_str)
        print(exp_str)
        print("\n\n")

    if opt_feature == 'cvc':

        plt.figure(figsize=(12,12),dpi=300)

        plt.title('CVC CDF {}'.format(subject_name,))

        cvc_model = model_output('cvc')
        cdf_model = cvc_model['cdf']
        vec_model = cvc_model['beat_latency']

        cdf_data = cvc_data['cdf']
        vec_data = cvc_data['beat_latency']
        
        interp_vec, cdf_data, cdf_model = pycardio.utils.interp_cvc(vec_data, cdf_data, vec_model, cdf_model)

        cdiff = np.abs(cdf_data - cdf_model)
        mix = np.argmax(cdiff)
        D = cdiff.max()

        cmaxdiff = np.abs(cdf_data - interp_vec)
        bix = np.argmax(cmaxdiff)
        Dmax = cmaxdiff.max()

        Dstr = "\nD: {:f}; Dmax: {:f}\n".format(D, Dmax)
        print(Dstr)
        
        plt.plot(interp_vec, cdf_model, 'b',label='model')
        plt.plot(interp_vec, cdf_data, 'k', label='data')
        plt.plot(interp_vec, interp_vec, 'k--', alpha=0.5, label='bisector')
        plt.plot([interp_vec[mix], interp_vec[mix]], [cdf_data[mix], cdf_model[mix]], 'r-x', label='D')
        plt.plot([interp_vec[bix], interp_vec[bix]], [interp_vec[bix], cdf_data[bix]], 'c-x', label='Dmax')
        # plt.plot(vec_model, cvc_model['cdf'], 'g', label='model_orig')
            
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.xlabel('Phase [Inspiration -> Inspiration]')
        plt.ylabel('CDF')

        plt.legend(loc='best', framealpha=0.5)
        plt.savefig('figs/{}_{}_cvc_result.svg'.format(opt_feature,subject_name))
        plt.close()

        cvc_model = model_output('cvc')
        model_rot = int(pycardio.calc_rot(cvc_model['rri_ave'], cvc_model['exp_time'], cvc_model['insp_time']))
        data_rot = int(pycardio.calc_rot(cvc_data['rri'].mean(), cvc_data['exp_time'], cvc_data['insp_time']))

        data_rot_str = "Data Rotation nr: {:d}\n".format(data_rot)
        model_rot_str = "Model Rotation nr: {:d}\n".format(model_rot)

        print(data_rot_str)
        print(model_rot_str)

        with open("figs/{}_{}.txt".format(opt_feature,subject_name), "a") as infof:
            infof.writelines([data_rot_str, model_rot_str, Dstr])

    
    # plot model output
    plt.figure(figsize=(12,12),dpi=300)

    plt.suptitle("Neurons: {} {}".format(subject_name,opt_feature))
    
    plt.subplot(511)
    plt.plot(tp, sp[:,0])
    plt.ylabel('pre-i/i')

    plt.subplot(512)
    plt.plot(tp, sp[:,1])
    plt.ylabel('early-i')

    plt.subplot(513)
    plt.plot(tp, sp[:,2])
    plt.ylabel('aug-e')

    plt.subplot(514)
    plt.plot(tp, sp[:,3])
    plt.ylabel('post-i')

    plt.subplot(515)
    plt.plot(tp, sp[:,4])
    plt.ylabel('ramp-i')
    plt.savefig('figs/{}_{}_neurons.svg'.format(opt_feature,subject_name))
    plt.close()
    
    plt.figure(figsize=(12,12),dpi=300)
    plt.suptitle("Signals: {} {}".format(subject_name,opt_feature))
    
    plt.subplot(411)
    plt.plot(tp, sp[:,10]) # heart rate rate
    plt.ylabel('heart rate accn')

    plt.subplot(412)
    plt.plot(tp, model_output('bp')) # blood pressure
    plt.ylabel("blood pressure")

    plt.subplot(413)
    plt.plot(tp, model_output('tv')) # tidal volume
    plt.ylabel("tidal volume")

    plt.subplot(414)
    plt.plot(tp[model_output('hr')[:,0].astype('int')], model_output('hr')[:,1]) # heart rate
    plt.ylabel('Heart Rate (bps)')
    plt.savefig('figs/{}_{}_signals.svg'.format(opt_feature,subject_name))
    plt.close()
    
    print('saved result figures')

    return ps

def plot_bar(subject, database, path, opt_feature, ps):
    nkeys = ['defaultParams', 'result', 'cost', 'labels', 'opt_label', 'covar', 'cdf']
    pnames = [pk for pk in sorted([pp for pp in ps.keys() if pp not in nkeys and len(ps[pp]) > 0])[:len(ps['result'].x)]]
    params = dict([(pnames[i], ps['result'].x[i]) for i in range(len(ps['result'].x))])
    vals = [px[1] for px in sorted(params.items(), key=lambda x: x[0])]

    sep_params = {"rsa": ['hr', 'rsa1', 'rsa2'],
                  "baro": ['baro'],
                  "po": ['po'],
                  "taup": ['taup'],
                  "resp": ['d2', 'd4']}

    for sk, sp in sep_params.items():
        if any(spp in pnames for spp in sp):
            
            plt.figure(figsize=(12,12), dpi=300)
            plt.title('{} : subject {}'.format(sk,subject,))
            ind = np.arange(1, len(sp)+1)
            bars = plt.bar(ind, [params[spp] for spp in sp], width=0.5)

            ax = plt.gca()
            ax.set_xticks(ind)
            ax.set_xticklabels(sp)

            plt.savefig('figs/{}_{}_bar_{}.svg'.format(opt_feature,subject,sk))
            plt.close()

    print('saved bar figures')
    

if __name__=='__main__':
    subject = 'f1y02'
    database = 'fantasia'
    path = os.path.expanduser('~/Public')
    opt_feature = 'ave_resp'

    if len(sys.argv) > 2:
        subject, opt_feature = sys.argv[1:]

    with open("figs/{}_{}.txt".format(opt_feature,subject), "w+") as infof:
        infof.writelines(["\n",])
        
    phase_comp = plot_data(subject, database, path, opt_feature)
    ps = plot_result(subject, database, path, opt_feature, phase_comp)
    plot_bar(subject, database, path, opt_feature, ps)
