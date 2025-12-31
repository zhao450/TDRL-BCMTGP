import KmeansWithMutGuideMTGP.GPFC as KmeansWithMutGuideMTGPmain
import sys
import os
os.environ["OMP_NUM_THREADS"] = '1'

sys.path

if __name__ == '__main__':
    dataset_name='HH'
    seed=5
    KmeansWithMutGuideMTGPmain.main(dataset_name, seed,10,0.8,0.2,1,2)


    


