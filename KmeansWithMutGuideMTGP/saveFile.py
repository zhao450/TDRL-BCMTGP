import pickle
import numpy as np

def save_top_inds_final_gen_meng(randomSeeds, dataSetName, top_inds_fitness_final_gen,ps,pc,pm,elitism,iteration):
    individual_dict = {}

    for gen in range(len(top_inds_fitness_final_gen)):
        best_ind = top_inds_fitness_final_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    with open('./KmeansWithMutGuideMTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_kmeansWithMutguide_top_individuals_final_gen_' + dataSetName +'_'+ps+'_'+pc+'_'+pm+'_'+elitism+'_'+iteration+ '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return

