import random
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tabulate import tabulate
import routing
import os
import shutil

class routing_brain:
    def __init__(self, env, job_creator, m_list, wc_list, warm_up, span, *args, **kwargs):
        self.env = env
        self.job_creator = job_creator
        self.m_list = m_list
        self.wc_list = wc_list
        self.m_per_wc = len(self.wc_list[0].m_list)
        print(self.wc_list[0].m_list,self.m_per_wc)
        for m in m_list:
            m.routing_learning_event.succeed()
        self.path = sys.path[0]
        self.input_size = self.m_per_wc*3 + 3

        self.save_intermediate = False

        self.single_agent=False
        if 'GPrule_action' in kwargs and kwargs['GPrule_action']:
            self.GPrule_action = True
            self.GPrule_ensemble = False
            self.func_list = [0, 1, 2, 3]
            self.vetor_rule=[]
            self.routing_GPtree_list = []
            if 'Single_agent' in kwargs and kwargs['Single_agent']:
                self.sequencing_GPtree_list = []
                self.single_agent = True
            for idx in self.func_list:
                dict_best_MTGP_individuals = kwargs['GPrules']
                individual = dict_best_MTGP_individuals.get(idx)
                routing_rule_tree = individual[1]
                self.routing_GPtree_list.append(routing_rule_tree)
                self.vetor_rule.append(kwargs['vector_rule'][idx])
                if 'Single_agent' in kwargs and kwargs['Single_agent']:
                    sequencing_rule_tree = individual[0]
                    self.sequencing_GPtree_list.append(sequencing_rule_tree)
            for m in self.wc_list:
                m.GPrule_action = True
                m.GPrule_ensemble = False
                if 'Single_agent' in kwargs and kwargs['Single_agent']:
                    m.single_agent = True
            self.output_size = len(self.routing_GPtree_list)
        elif 'GPrule_ensemble' in kwargs and kwargs['GPrule_ensemble']:
            self.GPrule_action = False
            self.GPrule_ensemble = True
            self.func_list = [0,1,2,3]
            self.func_weights_list = [0, 1, 2, 3]
            self.routing_GPtree_list = []
            for idx in self.func_list:
                dict_best_MTGP_individuals = kwargs['GPrules']
                individual = dict_best_MTGP_individuals.get(idx)
                routing_rule_tree = individual[1]
                self.routing_GPtree_list.append(routing_rule_tree)
            for m in self.wc_list:
                m.GPrule_action = False
                m.GPrule_ensemble = True
            if 'End_to_end' in kwargs and kwargs['End_to_end']:
                self.output_size = self.m_per_wc
            else:
                self.output_size = len(self.routing_GPtree_list)
        else:
            self.GPrule_action = False
            self.GPrule_ensemble = False
            for m in self.wc_list:
                m.GPrule_action = False
                m.GPrule_ensemble = False
            self.output_size = self.m_per_wc

        self.use_build_experience_strategy = False
        if 'use_build_experience_strategy' in kwargs and kwargs['use_build_experience_strategy']:
            self.use_build_experience_strategy = True
            for m in self.wc_list:
                m.use_build_experience_strategy = True

        self.lr = 0.01
        print("---> DEFAULT mode ON <---")
        self.routing_action_NN = build_network_small(self.input_size, self.output_size)
        self.seed = kwargs['seed']
        self.dataset_name = kwargs['dataset_name']
        self.address_seed = "{}/routing_models/scenario_" + self.dataset_name + "/run_" + str(
            self.seed) + "_small_state_dict" + '{}wc{}m_treelstm'.format(len(wc_list),
                                                                len(m_list))
        self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
        self.build_state = self.state_deeper
        self.train = self.train_DDQN
        for wc in self.wc_list:
            wc.build_state = self.state_deeper

        self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr=self.lr, momentum = 0.9)
        self.minibatch_size = 128
        self.rep_memo_size = 512
        self.discount_factor = 0.99
        self.epsilon = 0.3
        self.warm_up = warm_up
        self.span = span
        self.routing_action_NN_training_interval = 2
        self.routing_action_NN_training_time_record = []
        self.routing_target_NN_sync_interval = 250
        self.routing_target_NN_update_time_record = []
        self.rep_memo = []
        self.data_memory = []
        self.exploration_record = []
        self.time_record = []
        self.loss_record = []
        self.delete_obsolete_experience = self.env.event()
        
        self.env.process(self.training_process_parameter_sharing())
        self.env.process(self.update_rep_memo_parameter_sharing_process())
        self.build_initial_rep_memo = self.build_initial_rep_memo_parameter_sharing
        self.env.process(self.update_learning_rate_process())
        self.address_seed += ".pt"
        self.env.process(self.warm_up_process())
        self.env.process(self.update_training_setting_process())

    def saveLoss(self):
        fileName = './routing_models/scenario_' + str(self.dataset_name) + '/' + str(self.run_seed) + '_training_loss'
        np.save(fileName, self.loss_record)

    def warm_up_process(self):
        print('+++ Take over the routing function of target workcenter +++')
        for wc in self.wc_list:
            wc.job_routing = self.action_random_exploration
        yield self.env.timeout(self.warm_up - 1)
        self.build_initial_rep_memo()
        for wc in self.wc_list:
            wc.job_routing = self.action_DRL

    def EA(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args, **kwargs):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        rank = np.argmin(routing_data, axis=0)
        a_t = torch.tensor(rank[1])
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        return a_t

    def CT(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args, **kwargs):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        completion_time = np.array(routing_data)[:,1].clip(0) + np.array(job_pt)
        rank = completion_time.argmin()
        a_t = torch.tensor(rank)
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        return a_t

    def normalise_meng(self, vector):
        sum = np.sum(vector)
        vector_normalised = []
        for value in vector:
            vector_normalised.append(value/sum)
        return np.array(vector_normalised)

    def action_default(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args, **kwargs):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        if self.GPrule_action:
            a_t = 0
            if self.single_agent:
                for m in self.m_list:
                    m.job_sequencing_tree = self.sequencing_GPtree_list[0]
        elif self.GPrule_ensemble:
            for i in range(len(self.func_weights_list)):
                weight_i = 1
                self.func_weights_list[i] = weight_i
        else:
            print("error here, action_default")
        if self.GPrule_action:
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                machine_position = routing.GP_pair_R_test(self.routing_GPtree_list[a_t], job_idx, routing_data_GPrule_action, job_pt,
                                                          kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'], kwargs['NOR'], kwargs['weight_list'],
                                                          kwargs['waiting_time'],job_slack)
                a_t = torch.tensor(a_t)
                self.build_experience(job_idx, s_t, a_t, wc_idx)
                self.time_record.append(self.env.now)
                print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx, self.wc_list[wc_idx].m_list[
                    machine_position].m_idx))
                return machine_position
        elif self.GPrule_ensemble:
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                ensemble_priority = 0
                for i in range(len(self.func_weights_list)):
                    machine_priority = self.func_weights_list[i] * self.normalise_meng(routing.GP_pair_ensemble_R_test(self.routing_GPtree_list[i], job_idx,
                                                                  routing_data_GPrule_action, job_pt,
                                                                  kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                                  kwargs['NOR'], kwargs['weight_list'],
                                                                  kwargs['waiting_time'], job_slack))
                    ensemble_priority = ensemble_priority + machine_priority
                machine_position = ensemble_priority.argmin()
                a_t = machine_position
                a_t = torch.tensor(a_t)
                self.build_experience(job_idx, s_t, a_t, wc_idx)
                self.time_record.append(self.env.now)
                return machine_position
        else:
            machine_position = a_t
            self.build_experience(job_idx, s_t, machine_position, wc_idx)
            self.time_record.append(self.env.now)
            print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx, self.wc_list[wc_idx].m_list[
                machine_position].m_idx))
            return machine_position

    def action_random_exploration(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args, **kwargs):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        if self.GPrule_action:
            a_t = torch.tensor(np.random.randint(0, self.output_size))
            if self.single_agent:
                for m in self.m_list:
                    m.job_sequencing_tree = self.sequencing_GPtree_list[a_t]
        elif self.GPrule_ensemble:
            for i in range(len(self.func_weights_list)):
                weight_i = np.random.random()
                self.func_weights_list[i] = weight_i
        else:
            a_t = torch.tensor(np.random.randint(0, self.m_per_wc))
        if self.GPrule_action:
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                machine_position = routing.GP_pair_R_test(self.routing_GPtree_list[a_t], job_idx, routing_data_GPrule_action, job_pt,
                                                          kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'], kwargs['NOR'], kwargs['weight_list'],
                                                          kwargs['waiting_time'],job_slack)
                if self.use_build_experience_strategy:
                    routing_GPtree_list_decision_True = []
                    for i in range(len(self.routing_GPtree_list)):
                        machine_position_i = routing.GP_pair_R_test(self.routing_GPtree_list[i], job_idx,
                                                                    routing_data_GPrule_action, job_pt,
                                                                    kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                                    kwargs['NOR'], kwargs['weight_list'],
                                                                    kwargs['waiting_time'], job_slack)

                        routing_GPtree_list_decision_True.append(machine_position_i)

                    for i in range(len(self.routing_GPtree_list)):
                        if i != a_t:
                            if routing_GPtree_list_decision_True[a_t] == routing_GPtree_list_decision_True[i]:
                                i_torch = torch.tensor(i)
                                self.build_experience(job_idx, s_t, i_torch, wc_idx)

                a_t = torch.tensor(a_t)
                self.build_experience(job_idx, s_t, a_t, wc_idx)
                self.time_record.append(self.env.now)
                print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx, self.wc_list[wc_idx].m_list[
                    machine_position].m_idx))
                return machine_position
        elif self.GPrule_ensemble:
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                ensemble_priority = 0
                for i in range(len(self.func_weights_list)):
                    machine_priority = self.func_weights_list[i] * self.normalise_meng(routing.GP_pair_ensemble_R_test(self.routing_GPtree_list[i], job_idx,
                                                                  routing_data_GPrule_action, job_pt,
                                                                  kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                                  kwargs['NOR'], kwargs['weight_list'],
                                                                  kwargs['waiting_time'], job_slack))
                    ensemble_priority = ensemble_priority + machine_priority
                machine_position = ensemble_priority.argmin()
                a_t = machine_position
                a_t = torch.tensor(a_t)
                self.build_experience(job_idx, s_t, a_t, wc_idx)
                self.time_record.append(self.env.now)
                return machine_position
        else:
            machine_position = a_t
            self.build_experience(job_idx, s_t, machine_position, wc_idx)
            self.time_record.append(self.env.now)
            print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx, self.wc_list[wc_idx].m_list[
                machine_position].m_idx))
            return machine_position

    def action_DRL(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args, **kwargs):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        if random.random() < self.epsilon:
            if self.GPrule_action:
                a_t = torch.tensor(np.random.randint(0,self.output_size))
                if self.single_agent:
                    for m in self.m_list:
                        m.job_sequencing_tree = self.sequencing_GPtree_list[a_t]
            elif self.GPrule_ensemble:
                for i in range(len(self.func_weights_list)):
                    weight_i = np.random.random()
                    self.func_weights_list[i] = weight_i
            else:
                a_t = torch.tensor(np.random.randint(0,self.m_per_wc))
        else:
            if self.GPrule_ensemble:
                value = self.routing_action_NN.forward(s_t.reshape(1, 1, self.input_size), wc_idx)
                for i in range(len(self.func_weights_list)):
                    weight_i = float(value[0][i].float())
                    self.func_weights_list[i] = weight_i
            else:
                values=[]
                for vec in range(len(self.vetor_rule)):
                    s_t_combine=torch.cat([s_t, self.vetor_rule[vec]], dim=0)
                    value = self.routing_action_NN.forward(s_t_combine.reshape([1,1,41]),wc_idx)
                    values.append(value)
                values_tensor = torch.stack(values)
                a_t = torch.argmax(values_tensor)
                if self.single_agent:
                    for m in self.m_list:
                        m.job_sequencing_tree = self.sequencing_GPtree_list[a_t]
        if self.GPrule_action:
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']

                machine_position = routing.GP_pair_R_test(self.routing_GPtree_list[a_t], job_idx,
                                                          routing_data_GPrule_action, job_pt,
                                                          kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                          kwargs['NOR'], kwargs['weight_list'],
                                                          kwargs['waiting_time'], job_slack)
                if self.use_build_experience_strategy:
                    routing_GPtree_list_decision_True = []
                    for i in range(len(self.routing_GPtree_list)):
                        machine_position_i = routing.GP_pair_R_test(self.routing_GPtree_list[i], job_idx,
                                                                  routing_data_GPrule_action, job_pt,
                                                                  kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                                  kwargs['NOR'], kwargs['weight_list'],
                                                                  kwargs['waiting_time'], job_slack)

                        routing_GPtree_list_decision_True.append(machine_position_i)

                    for i in range(len(self.routing_GPtree_list)):
                        if i != a_t:
                            if routing_GPtree_list_decision_True[a_t] == routing_GPtree_list_decision_True[i]:
                                i_torch = torch.tensor(i)
                                self.build_experience(job_idx, s_t, i_torch, wc_idx)

                a_t = torch.tensor(a_t)
                self.build_experience(job_idx, s_t, a_t, wc_idx)
                self.time_record.append(self.env.now)
                return machine_position
        elif self.GPrule_ensemble:
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                ensemble_priority = 0
                for i in range(len(self.func_weights_list)):
                    machine_priority = self.func_weights_list[i] * self.normalise_meng(routing.GP_pair_ensemble_R_test(self.routing_GPtree_list[i], job_idx,
                                                                  routing_data_GPrule_action, job_pt,
                                                                  kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                                  kwargs['NOR'], kwargs['weight_list'],
                                                                  kwargs['waiting_time'], job_slack))
                    ensemble_priority = ensemble_priority + machine_priority
                machine_position = ensemble_priority.argmin()
                a_t = machine_position
                a_t = torch.tensor(a_t)
                self.build_experience(job_idx, s_t, a_t, wc_idx)
                self.time_record.append(self.env.now)
                return machine_position
        else:
            machine_position = a_t
            self.build_experience(job_idx, s_t, machine_position, wc_idx)
            self.time_record.append(self.env.now)
            return machine_position

    def build_experience(self, job_idx, s_t, a_t, wc_idx):
        if self.use_build_experience_strategy:
            if job_idx in self.wc_list[wc_idx].incomplete_experience:
                self.wc_list[wc_idx].incomplete_experience[job_idx].append([s_t, a_t])
            else:
                self.wc_list[wc_idx].incomplete_experience[job_idx] = [[s_t, a_t]]
        else:
            self.wc_list[wc_idx].incomplete_experience[job_idx] = [s_t, a_t]

    def state_normalization(self, routing_data, job_pt, job_slack, wc_idx):
        coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]
        coming_job_no = coming_job_idx.size
        if coming_job_no:
            next_job = self.job_creator.release_time_list[coming_job_idx].argmin()
            coming_job_time = (self.job_creator.release_time_list[coming_job_idx] - self.env.now)[next_job]
            coming_job_slack = self.job_creator.arriving_job_slack_list[coming_job_idx][next_job]
        else:
            coming_job_time = 0
            coming_job_slack = 0
        m_state = [a[:2] for a in routing_data]
        s_t = torch.tensor(np.concatenate(m_state + [job_pt, [job_slack, coming_job_time, coming_job_slack]]), dtype=torch.float)
        return s_t

    def state_deeper(self, routing_data, job_pt, job_slack, wc_idx):
        coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]
        coming_job_no = coming_job_idx.size
        if coming_job_no:
            next_job = self.job_creator.release_time_list[coming_job_idx].argmin()
            coming_job_time = (self.job_creator.release_time_list[coming_job_idx] - self.env.now)[next_job]
            coming_job_slack = self.job_creator.arriving_job_slack_list[coming_job_idx][next_job]
        else:
            coming_job_time = 0
            coming_job_slack = 0
        m_state = [a[:2] for a in routing_data]
        state_meng = []
        for i in range(len(m_state)):
            for j in range(len(m_state[i])):
                state_meng.append(m_state[i][j])
        for i in range(len(job_pt)):
            state_meng.append(job_pt[i])
        state_meng.append(job_slack)
        state_meng.append(coming_job_time)
        state_meng.append(coming_job_slack)
        s_t = torch.tensor(state_meng, dtype=torch.float)
        return s_t

    def state_Lang2020(self, routing_data, job_pt, job_slack, wc_idx):
        m_state = [a[:1] for a in routing_data]
        s_t = torch.tensor(np.concatenate(m_state + [job_pt, [job_slack]]), dtype=torch.float)
        return s_t

    def build_initial_rep_memo_parameter_sharing(self):
        for wc in self.wc_list:
            self.rep_memo += wc.rep_memo.copy()
            wc.replay_memory = []
        print('INITIALIZATION - replay_memory is:\n',len(self.rep_memo),\
        tabulate(self.rep_memo, headers = ['s_t','a_t','r_t','s_t+1']))
        print('input-size:', self.input_size, '\nreplay_memory_size:', len(self.rep_memo))
        print('---------------------------initialization accomplished-----------------------------')

    def build_initial_rep_memo_independent(self):
        print('INITIALIZATION - replay_memory')
        for wc in self.wc_list:
            self.rep_memo[wc.wc_idx] += wc.rep_memo.copy()
            wc.replay_memory = []
            print(tabulate(self.rep_memo[wc.wc_idx], headers = ['s_t','a_t','s_t+1','r_t']))
            print('INITIALIZATION - size of replay memory:',len(self.rep_memo[wc.wc_idx]))
        print('---------------------------initialization accomplished-----------------------------')

    def update_rep_memo_parameter_sharing_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for wc in self.wc_list:
                self.rep_memo += wc.rep_memo.copy()
                wc.rep_memo = []
            if len(self.rep_memo) > self.rep_memo_size:
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
            yield self.env.timeout(self.routing_action_NN_training_interval*10)

    def update_rep_memo_independent_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for wc in self.wc_list:
                self.rep_memo[wc.wc_idx] += wc.rep_memo.copy()
                wc.rep_memo = []
                if len(self.rep_memo[wc.wc_idx]) > self.rep_memo_size:
                    truncation = len(self.rep_memo[wc.wc_idx])-self.rep_memo_size
                    perm = np.random.permutation(len(self.rep_memo[wc.wc_idx]))
                    index = 0
                    while len(self.rep_memo[wc.wc_idx]) > self.rep_memo_size:
                        idx = perm[index]
                        if self.rep_memo[wc.wc_idx][idx][2] <= 0:
                            del self.rep_memo[wc.wc_idx][idx]
            yield self.env.timeout(self.routing_action_NN_training_interval*10)

    def check_parameter(self):
        print('------------- Training Parameter Check -------------')
        print("Address seed:",self.address_seed)
        print('State Func.:',self.build_state.__name__)
        print('ANN:',self.routing_action_NN.__class__.__name__)
        print('------------- Training Parameter Check -------------')
        print('Discount rate:',self.discount_factor)
        print('Train feq: %s, Sync feq: %s'%(self.routing_action_NN_training_interval,self.routing_target_NN_sync_interval))
        print('Rep memo: %s, Minibatch: %s'%(self.rep_memo_size,self.minibatch_size))
        print('------------- Training Scenarios Check -------------')
        print("Configuration: {} work centers, {} machines".format(len(self.wc_list),len(self.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------')

    def training_process_parameter_sharing(self):
        yield self.env.timeout(self.warm_up)
        for i in range(20):
            self.train()
        while self.env.now < self.span:
            self.train()
            yield self.env.timeout(self.routing_action_NN_training_interval)
        print('Final replay_memory is:\n','size:',len(self.rep_memo),\
        tabulate(self.rep_memo, headers = ['s_t','a_t','r_t','s_t+1']))
        torch.save(self.routing_action_NN.state_dict(), self.address_seed.format(sys.path[0]))
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))

    def update_training_setting_process(self):
        yield self.env.timeout(self.warm_up+1)
        while self.env.now < self.span:
            self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
            if self.save_intermediate:
                self.save_intermediate_policy()
            print('--------------------------------------------------------')
            print('the target network and epsilion are updated at time %s' % self.env.now)
            print('--------------------------------------------------------')
            yield self.env.timeout(self.routing_target_NN_sync_interval)

    def update_training_parameters_process(self):
        yield self.env.timeout(self.warm_up)
        reduction = (self.routing_action_NN.lr - self.routing_action_NN.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            self.routing_action_NN.lr -= reduction
            self.epsilon -= 0.002
            if self.routing_action_NN.lr < 0.001:
                self.routing_action_NN.lr = 0.001
            if self.epsilon < 0.1:
                self.epsilon = 0.1
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.routing_action_NN.lr, self.env.now))
            print('--------------------------------------------------------')

    def update_learning_rate_process(self):
        yield self.env.timeout(self.warm_up)
        reduction = (self.lr - self.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            self.lr -= reduction
            if self.lr < 0.001:
                self.lr = 0.001
            self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr = self.lr, momentum = 0.9)
            self.epsilon -= 0.01
            if self.epsilon < 0.1:
                self.epsilon = 0.1
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.lr, self.env.now))
            print('--------------------------------------------------------')

    def train_DDQN(self):
        print(".............TRAINING .............%s"%(self.env.now))
        Q_0_temp=[]
        Q_1_action_temp=[]
        Q_1_target_temp=[]
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1)
        
        for vec in range(len(self.vetor_rule)):
            vetor_rule_expanded = self.vetor_rule[vec].T.unsqueeze(0).unsqueeze(1).repeat(sample_s0_batch.size(0), 1, 1)      
            sample_s0_batch_combined = torch.cat([sample_s0_batch, vetor_rule_expanded], dim=2)
            value = self.routing_action_NN.forward(sample_s0_batch_combined)
            Q_0_temp.append(value)  
        Q_0 = torch.cat(Q_0_temp, dim=1)    
        current_value = Q_0.gather(1, sample_a0_batch)
        
        for vec in range(len(self.vetor_rule)):
            vetor_rule_expanded = self.vetor_rule[vec].T.unsqueeze(0).unsqueeze(1).repeat(sample_s1_batch.size(0), 1, 1)
            sample_s1_batch_combined = torch.cat([sample_s1_batch, vetor_rule_expanded], dim=2)
            Q_1_action = self.routing_action_NN.forward(sample_s1_batch_combined).detach()
            Q_1_action_temp.append(Q_1_action)
            Q_1_target = self.routing_target_NN.forward(sample_s1_batch_combined).detach()
            Q_1_target_temp.append(Q_1_target)
        Q_1_action = torch.cat(Q_1_action_temp, dim=1)
        Q_1_target = torch.cat(Q_1_target_temp, dim=1)
        
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action,dim=1)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size, 1])
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        next_state_value *= self.discount_factor
        target_value = (sample_r0_batch + next_state_value)
        
        loss = self.routing_action_NN.loss_func(current_value, target_value)
        print('loss is:', loss)
        self.loss_record.append(float(loss))
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def save_intermediate_policy(self):
        path_seed = "{}/routing_models/scenario_" + self.dataset_name + "/run_" + str(self.seed) + "_policies/"
        path = path_seed.format(sys.path[0])
        isExists = os.path.exists(path)
        index = int((self.env.now-self.warm_up-1)/(self.routing_target_NN_sync_interval))
        if not isExists and index == 0:
            os.makedirs(path)
        elif isExists and index == 0:
            shutil.rmtree(path)
            os.makedirs(path)
        intermediate_address_seed = path + "/run_" + str(self.seed) + "_small_state_dict"+'{}wc{}m' +"_time" + str(index) + ".pt"
        address = intermediate_address_seed.format(len(self.wc_list),len(self.m_list))
        torch.save(self.routing_action_NN.state_dict(), address)

    def loss_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5.5))
        loss_record = fig.add_subplot(1,1,1)
        loss_record.set_xlabel('Iterations of training ('+r'$\times 10^3$'+')')
        loss_record.set_ylabel('Loss (error) of training')
        iterations = np.arange(len(self.loss_record))
        loss_record.scatter(iterations, self.loss_record,s=0.6,color='r', alpha=0.2)
        x = 50
        loss_record.plot(np.arange(x/2,len(self.loss_record)-x/2+1,1),np.convolve(self.loss_record, np.ones(x)/x, mode='valid'),color='k',label='moving average',zorder=3)
        ylim_upper = 0.2
        ylim_lower = 0
        loss_record.set_xlim(0,len(self.loss_record))
        loss_record.set_ylim(ylim_lower,ylim_upper)
        xtick_interval = 1000
        loss_record.set_xticks(np.arange(0,len(self.loss_record)+1,xtick_interval))
        loss_record.set_xticklabels(np.arange(0,len(self.loss_record)/xtick_interval,1).astype(int),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        loss_record.set_yticks(np.arange(ylim_lower, ylim_upper+0.01, 0.01))
        loss_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        loss_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        loss_record.legend()
        ax_time = loss_record.twiny()
        ax_time.set_xlabel('Time in simulation ('+r'$\times 10^3$'+', excluding warm up phase)')
        ax_time.set_xlim(self.warm_up,self.span)
        ax_time.set_xticks(np.arange(self.warm_up,self.span+1,xtick_interval*2))
        ax_time.set_xticklabels(np.arange(self.warm_up/xtick_interval,self.span/xtick_interval+1,2).astype(int),rotation=30, ha='left', rotation_mode="anchor", fontsize=8.5)
        loss_record.set_title("Routing Agent Training Loss / {}-machine per work centre test".format(int(len(self.m_list)/len(self.job_creator.wc_list))))
        fig.subplots_adjust(top=0.8, bottom=0.1, right=0.9)
        plt.show()
        if 'save' in kwargs and kwargs['save'] == 1:
            addressPNG = sys.path[0]+"/routing_models/scenario_" + kwargs['dataset_name'] + "/RA_loss_{}wc_{}m_Meng_{}.png".format(len(self.job_creator.wc_list),len(self.m_list),kwargs['seed'])
            fig.savefig(addressPNG, dpi=500, bbox_inches='tight')
            print('figure saved to'+addressPNG)
            addressPDF = sys.path[0] + "/routing_models/scenario_" + kwargs['dataset_name'] + "/RA_loss_{}wc_{}m_Meng_{}.pdf".format(len(self.job_creator.wc_list), len(self.m_list),kwargs['seed'])
            fig.savefig(addressPDF, dpi=500, bbox_inches='tight')
            print('figure saved to' + addressPDF)
        return

    def reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5))
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.rt_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.rt_reward_record).transpose()[1]
        reward_record.scatter(time, rewards, s=1,color='g', alpha=0.3, zorder=3)
        reward_record.set_xlim(0,self.span)
        reward_record.set_ylim(-1.1,1.1)
        xtick_interval = 2000
        reward_record.set_xticks(np.arange(0,self.span+1,xtick_interval))
        reward_record.set_xticklabels(np.arange(0,self.span+1,xtick_interval),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        reward_record.set_yticks(np.arange(-1, 1.1, 0.1))
        reward_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        reward_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        x = 50
        print(len(time))
        reward_record.plot(time[int(x/2):len(time)-int(x/2)+1],np.convolve(rewards, np.ones(x)/x, mode='valid'),color='k',label="moving average")
        reward_record.legend()
        plt.show()
        if 'save' in kwargs and kwargs['save']:
            fig.savefig(sys.path[0]+"/routing_models/scenario_" + kwargs['dataset_name'] + "/RA_reward_{}wc_{}m_Meng_{}.png".format(len(self.job_creator.wc_list),len(self.m_list),kwargs['seed']), dpi=500, bbox_inches='tight')
        return

    def culmulative_reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5))
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.rt_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.rt_reward_record).transpose()[1]
        culmulative_rewards = []
        for i in range(len(rewards)):
            if i == 0:
                culmulative_rewards.append(rewards[0])
            else:
                cumulative_reward_i = culmulative_rewards[i - 1]*self.discount_factor + rewards[i]
                culmulative_rewards.append(cumulative_reward_i)
        culmulative_rewards = np.array(culmulative_rewards)
        reward_record.scatter(time, culmulative_rewards, s=1,color='g', alpha=0.3, zorder=3)
        reward_record.set_xlim(0,self.span)
        xtick_interval = 2000
        reward_record.set_xticks(np.arange(0,self.span+1,xtick_interval))
        reward_record.set_xticklabels(np.arange(0,self.span+1,xtick_interval),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        reward_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        reward_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        x = 50
        print(len(time))
        reward_record.plot(time[int(x/2):len(time)-int(x/2)+1],np.convolve(culmulative_rewards, np.ones(x)/x, mode='valid'),color='k',label="moving average")
        reward_record.legend()
        plt.show()
        if 'save' in kwargs and kwargs['save']:
            fig.savefig(sys.path[0]+"/routing_models/scenario_" + kwargs['dataset_name'] + "/RA_culmulative_reward_{}wc_{}m_Meng_{}.png".format(len(self.job_creator.wc_list),len(self.m_list),kwargs['seed']), dpi=500, bbox_inches='tight')
        return

class build_network_small(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_small, self).__init__()
        layer_1 = 64
        layer_2 = 128
        layer_3 = 70
        layer_4 = 30
        layer_5 = 15
        self.fc1 = nn.Linear(41, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, 1)
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        x = self.instancenorm(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        return x

class build_network_medium(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_medium, self).__init__()
        layer_1 = 32
        layer_2 = 32
        layer_3 = 16
        layer_4 = 8
        layer_5 = 8
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, output_size)
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        x = self.instancenorm(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        return x

class build_network_large(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_large, self).__init__()
        layer_1 = 64
        layer_2 = 64
        layer_3 = 48
        layer_4 = 32
        layer_5 = 16
        layer_6 = 16
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, layer_6)
        self.fc7 = nn.Linear(layer_6, output_size)
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        x = self.instancenorm(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.tanh(x)
        x = self.fc7(x)
        return x

class build_network_TEST(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_TEST, self).__init__()
        layer_1 = 64
        layer_2 = 64
        layer_3 = 48
        layer_4 = 32
        layer_5 = 16
        layer_6 = 16
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, layer_6)
        self.fc7 = nn.Linear(layer_6, output_size)
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        x = self.instancenorm(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.tanh(x)
        x = self.fc7(x)
        return x