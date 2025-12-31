import  pickle
import numpy as np
import sys



def load_top_inds_from_final_gen(randomSeeds, dataSetName): 
    with open(sys.path[0] + '/KmeansWithMutGuideMTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_kmeansWithMutguide_top_individuals_final_gen_' + dataSetName + '.pkl',
            "rb") as fileName_individual:
        dict = pickle.load(fileName_individual)
    return dict

