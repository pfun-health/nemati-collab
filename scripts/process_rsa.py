from process import *

"""
    process_rsa.py: Calculate the RSA stats (RR interval deviation versus phase)
"""

def run_rsa(subject, db):
    time = db('time')[subject].copy()
    resp = db('RESP')[subject].copy()
    beat_ix = db('beat')[subject].copy()

    other_sigs, time, resp_s = db.smooth_signal(time, resp, wavelet_size=250, other_sigs=[beat_ix,])
    beat_ix = other_sigs[0]
    
    order = 250
    phase_spots, rri_dev = db.calc_RSA(time, beat_ix, resp_s, order=order)

    plt.figure()
    plt.plot(phase_spots, rri_dev)
    plt.show()


if __name__ == '__main__':

    print("Starting process_rsa.py")
    repo_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-2])
    print("repo_dir: {}".format(repo_dir,))

    db = DB('fantasia')

    plot_ = True
    filtword = 'ALL'
    subjects = [kk for kk in db().keys() if filtword in kk or (filtword=='ALL')]

    for si, subject in enumerate(subjects):
        print(str(si)+' : '+subject+'\n')
        run_rsa(subject, db)

    # print("saved RSA stats to rsa.json")
