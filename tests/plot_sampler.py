
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt

import numpy as np
import sys,os

if __name__ == '__main__':

    subject = "f2y03"
    opt_feature = "rsa"
    
    if len(sys.argv) > 2:
        subject, opt_feature = sys.argv[1:]

    paramSpace = np.load(os.path.expanduser("~/Public/fantasia/sampleSpace_{}_{}.npy".format(opt_feature,subject)))[()]


    print('\n {} {} \n'.format(subject, opt_feature))
    print('\nplotting sample space\n')

    # print(paramSpace['samples'].std(0))

    print('std err: {}'.format(str(paramSpace['std'])))

    print(len(paramSpace['L']))
    print(paramSpace['samples'].shape)
    
    params = [k for k in paramSpace.keys() if k not in ['L', 'samples', 'std', 'accept',
                                                        'defaultParams', 'cov']]

    # for pruning samples
    pks = list(params)
    pks.extend(['L', 'samples'])
    pwhere = np.argwhere(np.array(paramSpace['L']) > 1.5e-07).flatten()
    for pk in pks:
        print(pk)
        if pk != 'samples':
            paramSpace[pk] = np.array(paramSpace[pk])[pwhere]
        elif pk == 'samples':
            paramSpace[pk] = np.array(paramSpace[pk])[pwhere,:]

    print('std err : {}'.format(str(paramSpace['samples'].std(0))))

    
    plt.figure(figsize=(12,12), dpi=300)
    plt.title("Likelihood {} {}".format(opt_feature, subject))
    iterline = np.arange(len(paramSpace['L']))
    plt.plot(iterline, paramSpace['L'], 'k', label='likelihood')
    plt.xlabel('samples')
    plt.ylabel('inverse cost')
    plt.legend(loc='best')
    plt.savefig('figs/ss_{}_{}_likelihood.svg'.format(subject, opt_feature))
    plt.close()

    plt.figure(figsize=(12,12), dpi=300)
    plt.title('Acceptance rate {} {}'.format(opt_feature, subject))
    aline = np.arange((len(paramSpace['accept'])))
    plt.plot(aline, paramSpace['accept'], 'k', label='acceptance rate')
    plt.xlabel('samples')
    plt.ylabel('acceptance rate')
    plt.legend(loc='best')
    plt.savefig('figs/ss_{}_{}_accept.svg'.format(subject, opt_feature))
    plt.close()


    params = [k for k in paramSpace.keys() if k not in ['L', 'samples', 'std', 'accept',
                                                        'defaultParams', 'cov']]

    
    mlix = np.argmax(paramSpace['L'])
    ml_x = paramSpace['samples'][mlix,:]
    print('x* = {} ; L* = {}'.format(str(ml_x), str(paramSpace['L'][mlix])))

    
    done = []
    for pk1 in params:
        for pk2 in params:
            if pk2 != pk1 and pk2 not in done:
                done.append(pk2)
                done.append(pk1)
                plt.figure(figsize=(12,12), dpi=300)
                plt.title("{} vs {} ; {} {}".format(pk1, pk2, opt_feature, subject))
                plt.plot(paramSpace[pk2], paramSpace[pk1], 'b-', lw=1.0, ms=2.0)
                plt.plot(paramSpace[pk2][0], paramSpace[pk1][0], 'g*', label='start', ms=10.0)
                plt.plot(paramSpace[pk2][-1], paramSpace[pk1][-1], 'r*', label='end', ms=10.0)
                plt.plot(paramSpace[pk2][mlix], paramSpace[pk1][mlix], 'm*', label='optimum', ms=10.0)
                plt.legend(loc='best')
                xmax = paramSpace[pk2][mlix]
                ymax = paramSpace[pk1][mlix]
                # plt.xlim([xmax-0.001*xmax, xmax+0.001*xmax])
                # plt.ylim([ymax-0.001*ymax, ymax+0.001*ymax])
                plt.savefig('figs/ss_{}_{}_{}vs{}.svg'.format(subject, opt_feature, pk1, pk2))
                plt.close()

    print('\ndone plotting sample space')
