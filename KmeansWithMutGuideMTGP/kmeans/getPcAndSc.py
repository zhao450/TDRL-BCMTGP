import KmeansWithMutGuideMTGP.kmeans.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import job_creation
import KmeansWithMutGuideMTGP.kmeans.RoutingPhenoCharacterisation as RoutingPhenoCharacterisation
import KmeansWithMutGuideMTGP.kmeans.SequencingPhenoCharacterisation as SequencingPhenoCharacterisation
import simpy
from deap import gp

class shopfloor_kemas:
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
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) 
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) 
            cum_m_idx += m_per_wc

        '''STEP 3: initialize the job creator'''
        if 'seed' in kwargs:
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) 
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5, 25], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            wc.setJobRoutingTree(routing_tree)
            wc.setGetDecisionSituation(self.routingDecisionSituations)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)
            m.setGetDecisionSituation(self.sequencingDecisionSituations)


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

    def getDecisionSituations(self):
        decisionSituations = []
        if len(self.sequencingDecisionSituations) < 20:
            print("Error in get enough number of self.sequencingDecisionSituations)")

        if len(self.routingDecisionSituations) < 20:
            print("Error in get enough number of self.routingDecisionSituations)")

        np.random.shuffle(self.sequencingDecisionSituations)
        subset_sequencingDecisionSituations = []
        tryTimes = 0
        while len(subset_sequencingDecisionSituations) < 20 and tryTimes < len(self.sequencingDecisionSituations):
            if len(self.sequencingDecisionSituations[tryTimes].getData()[3]) == 3: 
                subset_sequencingDecisionSituations.append(self.sequencingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1

        print("tryTimes for sequencing = " + str(tryTimes))
        print("size for sequencing = " + str(len(subset_sequencingDecisionSituations)))

        np.random.shuffle(self.routingDecisionSituations)
        subset_routingDecisionSituations = []
        tryTimes = 0
        while len(subset_routingDecisionSituations) < 20 and tryTimes < len(self.routingDecisionSituations):
            if len(self.routingDecisionSituations[tryTimes].getData()[1]) == 3:
                subset_routingDecisionSituations.append(self.routingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1
        print("tryTimes for routing = " + str(tryTimes))
        print("size for routing = " + str(len(subset_routingDecisionSituations)))

        decisionSituations.append(subset_sequencingDecisionSituations)
        decisionSituations.append(subset_routingDecisionSituations)
        return decisionSituations

class shopfloor_kmeans_seq:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] 
        self.sequencingDecisionSituations = []
        self.routingDecisionSituations = []
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
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) 
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) 
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        if 'seed' in kwargs:
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) 
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)
            m.setGetDecisionSituation(self.sequencingDecisionSituations)


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

    def getDecisionSituations(self):
        decisionSituations = []
        if len(self.sequencingDecisionSituations) < 20:
            print("Error in get enough number of self.sequencingDecisionSituations)")

        np.random.shuffle(self.sequencingDecisionSituations)
        subset_sequencingDecisionSituations = []
        tryTimes = 0
        while len(subset_sequencingDecisionSituations) < 20 and tryTimes < len(self.sequencingDecisionSituations):
            if len(self.sequencingDecisionSituations[tryTimes].getData()[3]) == 3: 
                subset_sequencingDecisionSituations.append(self.sequencingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1

        print("tryTimes for sequencing = " + str(tryTimes))
        print("size for sequencing = " + str(len(subset_sequencingDecisionSituations)))

        decisionSituations.append(subset_sequencingDecisionSituations)
        return decisionSituations

class getpcsc:
    def __init__(self, **kwargs):
        self.decisionSituations = []
        self.phenotypic_characristics = []


    def initial_phenoCharacterisation(self, isissimulation,decisionSituations,individual):
        self.phenotypic_characristics = []
        env = simpy.Environment()
        dataset_name = 'HH'
        rule_R = 'GP_evolve_R'
        rule_S = 'GP_evolve_S'
        seed = 777777
        span = 100000
        m_no = 9
        wc_no = 3
        if len(individual) == 2:
            if isissimulation:
                spf = shopfloor_kemas(env, span, m_no, wc_no, individual[0], individual[1], routing_rule=rule_R,
                                sequencing_rule=rule_S,seed=seed, ifPrint=False, dataset_name=dataset_name)
                spf.simulation()
                self.decisionSituations = spf.getDecisionSituations()
            else:
                self.decisionSituations=decisionSituations
            sequencingPhenoCharacterisation = SequencingPhenoCharacterisation.SequencingPhenoCharacterisation(individual[0], self.decisionSituations[0])
            routingPhenoCharacterisation = RoutingPhenoCharacterisation.RoutingPhenoCharacterisation(individual[1], self.decisionSituations[1])
            self.phenotypic_characristics.append(sequencingPhenoCharacterisation)
            self.phenotypic_characristics.append(routingPhenoCharacterisation)
        else:
            rule_R = 'EA'
            if isissimulation:
                spf = shopfloor_kmeans_seq(env, span, m_no, wc_no, individual[0], routing_rule=rule_R,
                                        sequencing_rule=rule_S, seed=seed, ifPrint=False, dataset_name=dataset_name)
                spf.simulation()
                self.decisionSituations = spf.getDecisionSituations()
            else:
                self.decisionSituations=decisionSituations
            sequencingPhenoCharacterisation = SequencingPhenoCharacterisation.SequencingPhenoCharacterisation(
                individual[0], self.decisionSituations[0])
            self.phenotypic_characristics.append(sequencingPhenoCharacterisation)


    def calculate_phenoCharacterisation(self, individual):
        if len(individual) == 2:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])
            self.phenotypic_characristics[1].setReferenceRule(individual[1])
        else:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])

    def extract_and_save_subtrees(self,individual):

        subtree_indiv=[]
        for idx,node in enumerate(individual):
            if isinstance(node,gp.Primitive):
                subtree=individual[individual.searchSubtree(idx)]       
                subtree_gp=gp.PrimitiveTree(subtree)
                subtree_indiv.append(subtree_gp)
        return subtree_indiv

    #add by zhaoc
    def get_structure_characteristics(self,rout_tree,deq_tree,population,toolbox):

        structure_characteristics = []
        subtree_best_rout = self.extract_and_save_subtrees(rout_tree)
        subtree_best_rout.sort(key=lambda x: x.height,reverse=True)
        subtree_best_seq = self.extract_and_save_subtrees(deq_tree)
        subtree_best_seq.sort(key=lambda x: x.height,reverse=True)
        for ind in population:
            curr_subtree_rout=self.extract_and_save_subtrees(ind[1])
            curr_subtree_rout.sort(key=lambda x: x.height,reverse=True)
            curr_subtree_seq=self.extract_and_save_subtrees(ind[0])
            curr_subtree_seq.sort(key=lambda x: x.height,reverse=True)
            curr_tree_height_rout=ind[1].height   
            curr_tree_height_seq=ind[0].height  
            max_common_subtree_rout_height,min_common_subtree_rout_height=0,0
            for subtree in subtree_best_rout:
                if subtree in curr_subtree_rout:
                    max_common_subtree_rout_height=subtree.height
                    break
            for subtree in reversed(subtree_best_rout):
                if subtree in curr_subtree_rout:
                    min_common_subtree_rout_height=subtree.height
                    break
            structure_characteristics.append([curr_tree_height_rout-max_common_subtree_rout_height,min_common_subtree_rout_height])
            max_common_subtree_seq_height,min_common_subtree_seq_height=0,0
            for subtree in subtree_best_seq:
                if subtree in curr_subtree_seq:
                    max_common_subtree_seq_height=subtree.height
                    break
            for subtree in reversed(subtree_best_seq):
                if subtree in curr_subtree_seq:
                    min_common_subtree_seq_height=subtree.height
                    break
            structure_characteristics.append([curr_tree_height_seq-max_common_subtree_seq_height,min_common_subtree_seq_height])
        return structure_characteristics




    # add by zhaoc
    def calculate_structureCharacterisation(self,toolbox,sorted_pop):
        structure_characteristics = []
        best_ind=sorted_pop[0]
        structure_characteristics=self.get_structure_characteristics(best_ind[1],best_ind[0],sorted_pop,toolbox)

        return structure_characteristics


    def getpc(self,toolbox,population):
        if len(population[0]) == 2:

            phenotypic_characristics_pop = []
            sorted_pop = self.sortPopulation(toolbox, population)
            for idx in range(len(sorted_pop)):
                ind = sorted_pop[idx]
                sequencing_charList = self.phenotypic_characristics[0].characterise(ind[0])
                routing_charList = self.phenotypic_characristics[1].characterise(ind[1])
                all_charList = []
                for ref in sequencing_charList:
                    all_charList.append(ref)
                for ref in routing_charList:
                    all_charList.append(ref)
                phenotypic_characristics_pop.append(all_charList)               

        return phenotypic_characristics_pop


    #add by zhaoc
    def setPhenoCharacterisationFromDecisionSituations(self, decisionSituations, individual):
        self.decisionSituations = decisionSituations
        issimulation=False
        self.initial_phenoCharacterisation(issimulation,decisionSituations,individual)
    def sortPopulation(self, toolbox, population):
        populationCopy = [toolbox.clone(ind) for ind in population]
        popsize = len(population)

        for j in range(popsize):
            sign = False
            for i in range(popsize - 1 - j):
                sum_fit_i = np.sum(populationCopy[i].fitness.values)
                sum_fit_i_1 = np.sum(populationCopy[i + 1].fitness.values)
                if sum_fit_i > sum_fit_i_1:
                    populationCopy[i], populationCopy[i + 1] = populationCopy[i + 1], populationCopy[i]
                    sign = True
            if not sign:
                break
        return populationCopy