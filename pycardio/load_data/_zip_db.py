import numpy as np
import zipfile as zipf
from tempfile import TemporaryFile
import io, os
from google.cloud import storage

class ZipDB(object):
    """
        ZipDB : The database class for importing a zipfile from google cloud
    """
    def __init__(self, database="", dpath="", **kwargs):

        if not database:
            database = 'sepsis-data' # name of the bucket
        self.database = database
        
        if not dpath:
            dpath = os.path.expanduser('~/Public') # path to database
        self.dpath = dpath
            
        self.client = storage.Client() # will throw an exception if authentication needs to be set
        self.bucket = self.client.get_bucket(database)

    def __call__(self, dtype="db"):
        pass

    def _downloadZips(self):
        for b in self.bucket.list_blobs():
            rootfn, extfn = os.path.splitext(b.name)
            if extfn == '.zip':
                blob = self.bucket.get_blob(b.name)
                # print(blob.path)
                print(blob.public_url)
                # tf = TemporaryFile()
                # blob.download_to_file(tf)
                # narray = np.fromfile(tf)
                # yield narray
                
                # blobf = io.BytesIO(blob.download_as_string())
                # with zipf.ZipFile(blobf, 'r') as zf:
                #     for ff in zf.filelist:
                #         fn = ff.filename
                #         if os.path.splitext(fn)[1] == '.npy':                            
                #             with zf.open(fn) as zfile:
                #                 narray = np.frombuffer(zfile.read()).reshape(-1,2)
                #                 yield narray

if __name__=='__main__':
    import matplotlib.pyplot as plt

    zdb = ZipDB()
    zdb._downloadZips()
    
    # for zar in zdb._extractZips():
    #     plt.figure()
    #     # time = np.linspace(0, zar[:,0].size/240, zar[:,0].size)
    #     plt.plot(zar[:,0])
    #     plt.show()

