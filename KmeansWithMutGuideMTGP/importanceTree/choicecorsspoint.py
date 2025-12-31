import KmeansWithMutGuideMTGP.importanceTree.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import job_creation
import KmeansWithMutGuideMTGP.importanceTree.RoutingPhenoCharacterisation as RoutingPhenoCharacterisation
import KmeansWithMutGuideMTGP.importanceTree.SequencingPhenoCharacterisation as SequencingPhenoCharacterisation

from scipy.stats import spearmanr
from deap import gp  
from KmeansWithMutGuideMTGP.importanceTree.decompose import extract_and_save_subtrees

import simpy

class shopfloor_importsubtree:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, routing_tree, **kwargs):
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
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
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
            #self.job_creator.output()
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
                # order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
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

class shopfloor_importsubtree_seq:
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

class importanceTree:
    def __init__(self,  **kwargs):
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
                spf = shopfloor_importsubtree(env, span, m_no, wc_no, individual[0], individual[1], routing_rule=rule_R,
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
                spf = shopfloor_importsubtree_seq(env, span, m_no, wc_no, individual[0], routing_rule=rule_R,
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


    def safe_spearmanr(self,a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.3
        corr, _ = spearmanr(a, b)
        return corr if not np.isnan(corr) else 0.3 

    def getimportancefactor(self,toolbox,population):
        if len(population[0]) == 2:
            populationbydecompose=[]
            for i in range(len(population)):
                decomposeseq=extract_and_save_subtrees(population[i][0])
                decomposerou=extract_and_save_subtrees(population[i][1])
                populationbydecompose.append([decomposeseq])
                populationbydecompose.append([decomposerou])


            seq_subtree_rank=[]
            rou_subtree_rank=[]
            seq_spearman=[]
            rou_spearman=[]
            temp_seq_spearman=[]
            temp_rou_spearman=[]
            
            for idx in range(int(len(populationbydecompose)/2)):
                seq_spearman.append([])
                rou_spearman.append([])
                indseq =populationbydecompose[idx*2]
                indrou = populationbydecompose[idx*2+1] 
                for i in range(len(indseq[0])):
                    temp_seq_spearman.append([])
                    sequencing_charList = self.phenotypic_characristics[0].characterise_returnall(indseq[0][i])
                    seq_subtree_rank.append(sequencing_charList)              

                for i in range(len(seq_subtree_rank[0])):
                    temp_seq_spearman[0].append(1)
                    for j in range(1,len(indseq[0])):
                        correlaion=self.safe_spearmanr(seq_subtree_rank[0][i],seq_subtree_rank[j][i])
                        temp_seq_spearman[j].append(np.abs(correlaion))
                for i in range(len(indseq[0])):
                    seq_spearman[idx].append(np.mean(temp_seq_spearman[i]))
                for j in range(len(indrou[0])):
                    temp_rou_spearman.append([])
                    routing_charList = self.phenotypic_characristics[1].characterise_returnall(indrou[0][j])
                    rou_subtree_rank.append(routing_charList)
                for i in range(len(rou_subtree_rank[0])):
                    temp_rou_spearman[0].append(1)
                    for j in range(1,len(indrou[0])):
                        correlaion=self.safe_spearmanr(rou_subtree_rank[0][i],rou_subtree_rank[j][i])
                        temp_rou_spearman[j].append(np.abs(correlaion))
                for i in range(len(indrou[0])):
                    rou_spearman[idx].append(np.mean(temp_rou_spearman[i]))
                
                temp_seq_spearman=[] 
                temp_rou_spearman=[]
                seq_subtree_rank=[]
                rou_subtree_rank=[]

        return seq_spearman,rou_spearman,populationbydecompose
    

    def getterminalprobability(self,toolbox,population):
        
        terminal_occurrences_num = [0] * 10
        terminal_occurrences_num= [0] * 10
        
        for idx in range(len(population)):
            seq_subtree=population[idx][0]
            rou_subtree=population[idx][1]
            for i,node in enumerate(seq_subtree):
                if isinstance(node,gp.Terminal):
                    if node.name=="NIQ":
                        terminal_occurrences_num[0]+=1
                    elif node.name=="WIQ":
                        terminal_occurrences_num[1]+=1
                    elif node.name=="MWT":
                        terminal_occurrences_num[2]+=1
                    elif node.name=="PT":
                        terminal_occurrences_num[3]+=1
                    elif node.name=="NPT":
                        terminal_occurrences_num[4]+=1
                    elif node.name=="OWT":
                        terminal_occurrences_num[5]+=1
                    elif node.name=="WKR":
                        terminal_occurrences_num[6]+=1  
                    elif node.name=="NOR":
                        terminal_occurrences_num[7]+=1
                    elif node.name=="TIS":
                        terminal_occurrences_num[8]+=1
                    elif node.name=="SLACK":
                        terminal_occurrences_num[9]+=1
            for i,node in enumerate(rou_subtree):
                if isinstance(node,gp.Terminal):
                    if node.name=="NIQ":
                        terminal_occurrences_num[0]+=1
                    elif node.name=="WIQ":
                        terminal_occurrences_num[1]+=1
                    elif node.name=="MWT":
                        terminal_occurrences_num[2]+=1
                    elif node.name=="PT":
                        terminal_occurrences_num[3]+=1
                    elif node.name=="NPT":
                        terminal_occurrences_num[4]+=1
                    elif node.name=="OWT":
                        terminal_occurrences_num[5]+=1
                    elif node.name=="WKR":
                        terminal_occurrences_num[6]+=1  
                    elif node.name=="NOR":
                        terminal_occurrences_num[7]+=1
                    elif node.name=="TIS":
                        terminal_occurrences_num[8]+=1
                    elif node.name=="SLACK":
                        terminal_occurrences_num[9]+=1
        all_num_add = sum(terminal_occurrences_num)
        terminal_occurrences_probability = [x / all_num_add for x in terminal_occurrences_num]

        return terminal_occurrences_probability
        

    def getcrossindex(self,toolbox,population):
        seq_factor,rou_factor,_=self.getimportancefactor(toolbox,population)
        importance_seq_probability=[]
        unimportance_seq_probability=[]
        importance_rou_probability=[]
        unimportance_rou_probability=[]
        for i in range(len(seq_factor)):
            importance_seq_probability.append([])
            count_seq_importance=0
            for j in range(len(seq_factor[i])):
                count_seq_importance+=seq_factor[i][j]
            for j in range(len(seq_factor[i])):
                probability=seq_factor[i][j]/count_seq_importance
                importance_seq_probability[i].append(probability)
            unimportance_seq_probability.append([])
            count_seq_unimportance=0
            for j in range(len(seq_factor[i])):
                count_seq_unimportance+=1-seq_factor[i][j]
            for j in range(len(seq_factor[i])):
                if count_seq_unimportance==0:
                    probability=1/len(seq_factor[i])
                else:
                    probability=(1-seq_factor[i][j])/count_seq_unimportance
                unimportance_seq_probability[i].append(probability)


        for i in range(len(rou_factor)):
            importance_rou_probability.append([])
            count_rou_importance=0
            for j in range(len(rou_factor[i])):
                count_rou_importance+=rou_factor[i][j]
            for j in range(len(rou_factor[i])):
                probability=rou_factor[i][j]/count_rou_importance
                importance_rou_probability[i].append(probability)
            unimportance_rou_probability.append([])
            count_rou_unimportance=0
            for j in range(len(rou_factor[i])):
                count_rou_unimportance+=1-rou_factor[i][j]
            for j in range(len(rou_factor[i])):
                if count_rou_unimportance==0:
                    probability=0
                else:
                    probability=(1-rou_factor[i][j])/count_rou_unimportance
                unimportance_rou_probability[i].append(probability)
        select_subtree_index=[]
        for i in range(len(population)):
            indices={}
            imp_seq_idx=self.roulette_wheel_selection(importance_seq_probability[i])
            indices['imp_seq_idx']=imp_seq_idx
            unimp_seq_idx=self.roulette_wheel_selection(unimportance_seq_probability[i])
            indices['unimp_seq_idx']=unimp_seq_idx
            imp_rou_idx=self.roulette_wheel_selection(importance_rou_probability[i])
            indices['imp_rou_idx']=imp_rou_idx
            unimp_rou_idx=self.roulette_wheel_selection(unimportance_rou_probability[i])
            indices['unimp_rou_idx']=unimp_rou_idx
            select_subtree_index.append(indices)


        return select_subtree_index
    
    def setPhenoCharacterisationFromDecisionSituations(self, decisionSituations, individual):
        self.decisionSituations = decisionSituations
        issimulation=False
        self.initial_phenoCharacterisation(issimulation,decisionSituations,individual)


    def roulette_wheel_selection(self, probabilities):
        if sum(probabilities) == 0:
            return np.random.randint(0, len(probabilities))
        r = np.random.random()
    
        c = 0
        for i, p in enumerate(probabilities):
            c += p
            if r <= c:
                return i
        return len(probabilities) - 1

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