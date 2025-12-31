import random

import numpy as np
from deap import tools

from KmeansWithMutGuideMTGP import saveFile
from KmeansWithMutGuideMTGP.selection import selElitistAndTournament
from KmeansWithMutGuideMTGP.kmeans.getPcAndSc import getpcsc
from KmeansWithMutGuideMTGP.kmeans.kmeans import adaptive_clustering

from KmeansWithMutGuideMTGP.niching.niching import niching_clear


from KmeansWithMutGuideMTGP.importanceTree.decompose import extract_and_save_subtrees
from KmeansWithMutGuideMTGP.importanceTree.choicecorsspoint import importanceTree


def varAnd(population, toolbox, cxpb, mutpb, reppb,crosspoint):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb+reppb)
    new_mutpb=mutpb/(cxpb+mutpb+reppb)+new_cxpb
    i = 1
    while i < len(offspring):
        randomValue = random.random()
        if randomValue < new_cxpb: 
            if (offspring[i - 1] == offspring[i]) :
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                corss_one=crosspoint[i-1]
                corss_two=crosspoint[i]
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i],corss_one,corss_two)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb: 
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
        else: 
            del offspring[i].fitness.values
            i = i + 1
    return offspring

def varAnd2(population, toolbox, cxpb, mutpb, reppb,crosspoint,terminal_occurrences_probability):
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(0,int(len(offspring)/2)):
        rand=random.random()
        if rand<cxpb:
            cross_one=crosspoint[i*2]
            cross_two=crosspoint[i*2+1]
            offspring[i*2],offspring[i*2+1]=toolbox.mate(offspring[i*2],offspring[i*2+1],cross_one,cross_two)
            del offspring[i*2].fitness.values,offspring[i*2+1].fitness.values
        else:
            offspring[i*2],offspring[i*2+1]=offspring[i*2],offspring[i*2+1]
            del offspring[i*2].fitness.values,offspring[i*2+1].fitness.values
    for i in range(0,len(offspring)):
        rand=random.random()
        if rand<mutpb:
            if random.random()<0.7:
                utilization=False
                offspring[i],=toolbox.mutate(offspring[i],terminal_occurrences_probability,utilization)
            else:
                utilization=True
                offspring[i],=toolbox.mutate(offspring[i],terminal_occurrences_probability,utilization)
            del offspring[i].fitness.values
        else:
            offspring[i],=offspring[i],
            del offspring[i].fitness.values

    return offspring
def varAnd1(population, toolbox, cxpb, mutpb, reppb,crosspoint):
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(0,int(len(offspring)/2)):
        rand=random.random()
        if rand<cxpb:
            cross_one=crosspoint[i*2]
            cross_two=crosspoint[i*2+1]
            offspring[i*2],offspring[i*2+1]=toolbox.mate(offspring[i*2],offspring[i*2+1],cross_one,cross_two)
            del offspring[i*2].fitness.values,offspring[i*2+1].fitness.values
        else:
            offspring[i*2],offspring[i*2+1]=offspring[i*2],offspring[i*2+1]
            del offspring[i*2].fitness.values,offspring[i*2+1].fitness.values
    for i in range(0,len(offspring)):
        rand=random.random()
        if rand<mutpb:
            offspring[i],=toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
        else:
            offspring[i],=offspring[i],
            del offspring[i].fitness.values

    return offspring
def sortPopulation(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)

    for j in range(popsize):
        sign = False
        for i in range(popsize-1-j):
            sum_fit_i = np.sum(populationCopy[i].fitness.values)
            sum_fit_i_1 = np.sum(populationCopy[i+1].fitness.values)
            if sum_fit_i > sum_fit_i_1:
                populationCopy[i], populationCopy[i+1] = populationCopy[i+1], populationCopy[i]
                sign = True
        if not sign:
            break

    return populationCopy


def eaSimple(population, toolbox, cxpb, mutpb, reppb, elitism, ngen,pss ,seedRotate, rd, stats=None,halloffame=None, verbose=__debug__, seed = __debug__, dataset_name=__debug__):
    pop_size=len(population)
    randomSeed_ngen = []
    for i in range((ngen + 1)):
        randomSeed_ngen.append(np.random.randint(2000000000))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = [] 
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    rd['seed'] = randomSeed_ngen[0]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


        
    pop_fit = [ind.fitness.values[0] for ind in population]
    min_fitness.append(min(pop_fit))
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index]) 
    p_one = population[best_index]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    decisionSituations=None
    if rd['use_kmeans'] or rd['use_niching'] or rd['use_guide'] or rd['use_onlycrossguide']:
        nich=niching_clear(0,1)
        nich.initial_phenoCharacterisation(population[best_index])
        decisionSituations = nich.decisionSituations
        if rd['use_niching']:
            population=nich.clearPopulation(toolbox, population)
        if rd['use_kmeans']:
            ps=getpcsc()
            pop_fit_for_kmeans=[ind.fitness.values[0] for ind in population]
            best_index=np.argmin(pop_fit_for_kmeans)
            if decisionSituations:

                ps.setPhenoCharacterisationFromDecisionSituations(decisionSituations, population[best_index])
            else:
                ps.initial_phenoCharacterisation(population[best_index])
        if rd['use_guide']:
            it=importanceTree()
            it.setPhenoCharacterisationFromDecisionSituations(decisionSituations,population[best_index])
        if rd['use_onlycrossguide']:
            it=importanceTree()
            it.setPhenoCharacterisationFromDecisionSituations(decisionSituations,population[best_index])
    

    elite_archive = []
    elite_archive_maxsize=elitism*3


    for gen in range(1, ngen + 1):
        if seedRotate:
            rd['seed'] = randomSeed_ngen[gen]
        sorted_pop= sortPopulation(toolbox, population)
        sorted_elite =sorted_pop[:elitism]  

        for ind in sorted_elite:
            elitism_clone=toolbox.clone(ind)
            elite_archive.append(elitism_clone)
        if len(elite_archive)>elite_archive_maxsize:
            elite_archive=elite_archive[len(sorted_elite):]      
        other_ind= sorted_pop[elitism:]  
        offspring = toolbox.select(population, len(population)-elitism)

        if rd['use_guide']:
            pop_fit = [ind.fitness.values[0] for ind in offspring]
            best_index=np.argmin(pop_fit)
            it.calculate_phenoCharacterisation(offspring[best_index])
            terminal_occurrences_probability=it.getterminalprobability(toolbox,elite_archive)
            crosspoint=it.getcrossindex(toolbox,offspring)
        
        if rd['use_onlycrossguide']:
            pop_fit = [ind.fitness.values[0] for ind in offspring]
            best_index=np.argmin(pop_fit)
            it.calculate_phenoCharacterisation(offspring[best_index])
            crosspoint=it.getcrossindex(toolbox,offspring)

        offspring = varAnd1(offspring, toolbox, cxpb, mutpb, reppb,crosspoint)

        invalid_elite_ind = [ind for ind in sorted_elite]
        for ind in invalid_elite_ind:
            del ind.fitness.values
        fitnesses_elite = toolbox.map(toolbox.evaluate, invalid_elite_ind)
        for ind, fit in zip(invalid_elite_ind, fitnesses_elite):
            ind.fitness.values = fit
        invalid_other_ind=[ind for ind in other_ind]
        for ind in invalid_other_ind:
            del ind.fitness.values
        fitnesses_other=toolbox.map(toolbox.evaluate, invalid_other_ind)
        for ind, fit in zip(invalid_other_ind, fitnesses_other):
            ind.fitness.values = fit
        invalid_ind = [ind for ind in offspring]
        for ind in invalid_ind:
            del ind.fitness.values
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = invalid_elite_ind + invalid_other_ind +invalid_ind

        if halloffame is not None:
            halloffame.clear()  
            halloffame.update(population)
        pop_fit = [ind.fitness.values[0] for ind in population]
        best_index = np.argmin(pop_fit)
        best_ind_all_gen.append(population[best_index])
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)
        if rd['use_kmeans']:
            best_index=np.argmin(pop_fit)
            ps.calculate_phenoCharacterisation(population[best_index])  
            pop_onlyforkmeans=sortPopulation(toolbox,population)
            pcsc=ps.getpc(toolbox,pop_onlyforkmeans)
            population = adaptive_clustering(pop_onlyforkmeans, pcsc,toolbox ,pop_size)

        if rd['use_niching']:
            pop_fit = [ind.fitness.values[0] for ind in population]
            best_index=np.argmin(pop_fit)           
            nich.calculate_phenoCharacterisation(population[best_index])
            population = nich.clearPopulation(toolbox,population)

        pop_fit = [ind.fitness.values[0] for ind in population]  
        min_fitness.append(min(pop_fit))



        if gen == ngen:
            sorted_elite = sortPopulation(toolbox, population)
            top_inds_final_gen = []
            top_inds_fitness_final_gen = []
            for i in range(10):
                top_inds_final_gen.append(sorted_elite[i])
                top_inds_fitness_final_gen.append(sorted_elite[i].fitness.values[0])

    return population, logbook, min_fitness, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen


