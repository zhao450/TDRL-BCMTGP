import simpy
from deap import base
from deap import creator
from deap import gp
import KmeansWithMutGuideMTGP.multi_tree as mt
from KmeansWithMutGuideMTGP import ea_simple_elitism
from KmeansWithMutGuideMTGP.selection import *
import sys
from KmeansWithMutGuideMTGP import saveFile
import time
import random

import numpy as np
import job_creation
import KmeansWithMutGuideMTGP.kmeans.PhenoCharacterisation as PhenoCharacterisation
import KmeansWithMutGuideMTGP.kmeans.RoutingPhenoCharacterisation as RoutingPhenoCharacterisation
import KmeansWithMutGuideMTGP.kmeans.SequencingPhenoCharacterisation as SequencingPhenoCharacterisation

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, routing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] 
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) 
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i)
            exec(expr2)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) 
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) 
            exec(expr2)
            cum_m_idx += m_per_wc

        '''STEP 3: initialize the job creator'''
        if 'seed' in kwargs:
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.9, seed=kwargs['seed'], ifPrint = self.ifPrint) 
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5, 25], 3, 0.9, seed=kwargs['seed'], ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 2, 0.9, seed=kwargs['seed'], ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 3, 0.9, seed=kwargs['seed'], ifPrint=self.ifPrint)
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            wc.setJobRoutingTree(routing_tree)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        if 'sequencing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception
        if 'routing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception


    def simulation(self):
        self.env.run()

class shopfloorSeq:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint']
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) 
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i)
            exec(expr2)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) 
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) 
            exec(expr2)
            cum_m_idx += m_per_wc

        '''STEP 3: initialize the job creator'''
        if 'seed' in kwargs:
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) 
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        if 'sequencing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception

        if 'routing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception

    def simulation(self):
        self.env.run()


def connectedness(cluster):
    print(cluster)


def init_toolbox(toolbox, pset):
    REP.init_toolbox(toolbox, pset)
    toolbox.register("select", selElitistAndTournament, tournsize=TOURNAMENT_SIZE, elitism=ELITISM)


def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats

def evaluate(individual, toolbox, seed):
    if len(individual) == 2:
        dataset_name = rd['dataset_name']
        rule_R = 'GP_evolve_R'
        rule_S = 'GP_evolve_S'
        env = simpy.Environment()
        spf = shopfloor(env, span, m_no, wc_no, individual[0], individual[1], routing_rule=rule_R, sequencing_rule=rule_S,
                        seed=seed, ifPrint=False, dataset_name=dataset_name)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        fitness = cumulative_tard[-1]

        for i in range(ins_each_gen-1):
            seed = seed + 1000
            env = simpy.Environment()
            spf = shopfloor(env, span, m_no, wc_no, individual[0], individual[1], routing_rule=rule_R,
                            sequencing_rule=rule_S, seed=seed, ifPrint=False, dataset_name=dataset_name)
            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            fitness = fitness + cumulative_tard[-1]

        fitness = fitness/ins_each_gen
        scores = [fitness]
    else:
        dataset_name = rd['dataset_name']
        if only_sequencing_rule:
            rule_S = 'GP_evolve_S'
            rule_R = 'EA'
            env = simpy.Environment()
            spf = shopfloorSeq(env, span, m_no, wc_no, individual[0], routing_rule=rule_R,
                            sequencing_rule=rule_S,
                            seed=seed, ifPrint=False, dataset_name=dataset_name)
            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            fitness =  cumulative_tard[-1]

            for i in range(ins_each_gen - 1):
                seed = seed + 1000
                env = simpy.Environment()
                spf = shopfloorSeq(env, span, m_no, wc_no, individual[0], routing_rule=rule_R,
                                sequencing_rule=rule_S, seed=seed, ifPrint=False, dataset_name=dataset_name)
                spf.simulation()
                output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
                fitness = fitness + cumulative_tard[-1]

            fitness = fitness / ins_each_gen
            scores = [fitness]
        else:
            print("Error here!")

    return scores

def eval_wrapper(*args, **kwargs):
    return evaluate(*args, **kwargs, toolbox=rd['toolbox'], seed = rd['seed'])

def init_data(rundata):
    global rd
    rd = rundata



def GPFC_main(dataset_name, seed,ps,pc,pm,elitism,iteration):
    rd['use_niching'] = use_niching
    rd['use_kmeans']= use_kmeans
    rd['use_guide'] = use_guide
    rd['seed'] = seed
    rd['dataset_name'] = dataset_name
    rd['use_onlycrossguide']=use_onlycrossguide
    num_features = 0 
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    REP.init_primitives(pset)
    weights = (-1.,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    toolbox=base.Toolbox()
    init_toolbox(toolbox, pset)
    toolbox.register("evaluate", eval_wrapper)

    rd['toolbox'] = toolbox
    rd['only_sequencing_rule'] = only_sequencing_rule
    pop = toolbox.population(n=int(float(ps)))
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True 
    pop, logbook, min_fitness, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen = ea_simple_elitism.eaSimple(pop, toolbox, float(pc), float(pm), REPPB, int(float(elitism)), int(float(iteration)),int(float(ps)) ,seedRotate, rd, stats, halloffame=hof, verbose=True, seed =seed, dataset_name=dataset_name)
    best = hof[0]
    return min_fitness,best, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen




POP_SIZE =40
NGEN = 5
CXPB = 0.8
MUTPB = 0.15
REPPB = 0.05
ELITISM = 3
TOURNAMENT_SIZE = 4
MAX_HEIGHT = 8 #8
REP = mt  # 
REP.MAX_HEIGHT = MAX_HEIGHT
N_TREES = 2
# N_TREES = 1
REP.N_TREES = N_TREES
only_sequencing_rule = False
rd = {}
use_niching = True
use_kmeans= True
use_guide=False
use_onlycrossguide=True

span = 500
m_no = 6
wc_no = 3
ins_each_gen = 2 
def main(dataset_name, seed,ps,pc,pm,elitism,iteration):

    random.seed(int(seed))
    np.random.seed(int(seed))
    start = time.time()
    min_fitness,p_one,best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen = GPFC_main(dataset_name,seed,str(ps),str(pc),str(pm),str(elitism),str(iteration))
    end = time.time()
    running_time = end - start
    saveFile.save_top_inds_final_gen_meng(seed, dataset_name, top_inds_final_gen,str(ps),str(pc),str(pm),str(elitism),str(iteration))
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')



