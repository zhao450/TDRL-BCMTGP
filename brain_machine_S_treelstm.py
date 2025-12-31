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
import sequencing
import os
import shutil

class sequencing_brain:
    def __init__(self, env, job_creator, all_machines, target_machines, warm_up, span, *args, **kwargs):
        self.env = env
        self.job_creator = job_creator
        self.m_list = all_machines
        self.m_no = len(self.m_list)
        self.target_m_list = target_machines 
        self.target_m_no = len(self.target_m_list)
        self.warm_up = warm_up
        self.span = span
        self.job_creator.build_sqc_experience_repository(self.target_m_list)
        print("+++ Take over all machines, activate learning mode +++")
        for m in self.m_list:
            m.sequencing_learning_event.succeed()
            m.job_sequencing = self.action_default
        print('+++ Take over sequencing / reward function of target machines +++')
        for m in self.target_m_list:
            m.job_sequencing = self.action_warm_up

        self.save_intermediate = False 
        self.GPrule_action = False
        if 'GPrule_action' in kwargs and kwargs['GPrule_action']: 
            self.GPrule_action = True
            self.func_list = [0,1,2,3]
            self.sequencing_GPtree_list = []
            self.vetor_rule=[]
            for idx in self.func_list:
                dict_best_MTGP_individuals = kwargs['GPrules']
                individual = dict_best_MTGP_individuals.get(idx)
                sequencing_rule_tree = individual[0]
                self.sequencing_GPtree_list.append(sequencing_rule_tree)
                self.vetor_rule.append(kwargs['vector_rule'][idx])
            for m in self.m_list:
                m.GPrule_action = True  
            print('+++ Take over sequencing / reward function of target machines +++')
            for m in self.target_m_list:
                m.GPrule_action = True 
        else:
            self.GPrule_action = False
            self.func_list = [sequencing.SPT, sequencing.WINQ, sequencing.MS, sequencing.CR] 
            for m in self.m_list:
                m.GPrule_action = False 
            for m in self.target_m_list:
                m.GPrule_action = False 
        self.output_size = len(self.func_list)

        self.use_build_experience_strategy = False
        if 'use_build_experience_strategy' in kwargs and kwargs['use_build_experience_strategy']:
            self.use_build_experience_strategy = True
            for m in self.m_list:
                m.use_build_experience_strategy = True
            for m in self.target_m_list:
                m.use_build_experience_strategy = True 

        if 'reward_function' in kwargs:
            order = 'm.reward_function = m.get_reward{}'.format(kwargs['reward_function'])
            for m in self.target_m_list:
                exec(order)
        else:
            print('WARNING: reward function is not specified')
            raise Exception
        if 'MC' in kwargs and kwargs['MC']:
            print("---> Multi-Channel (MC) mode ON <---")
            self.input_size = len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.sequencing_action_NN = network_validated(self.input_size, self.output_size)
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            self.seed = kwargs['seed']
            self.dataset_name = kwargs['dataset_name']
            self.address_seed = "{}/sequencing_models/scenario_" + self.dataset_name + "/run_" + str(self.seed) +"_MC_rwd_tree_lstm" + str(kwargs['reward_function']) + ".pt"
            self.build_state = self.state_multi_channel
            self.train = self.train_validated
            self.action_DRL = self.action_sqc_rule
            for m in self.target_m_list:
                m.build_state = self.state_multi_channel
        else:
            print("WARNING: ANN TYPE NOT SPECIFIED !!!!")

        if "trained_parameter" in kwargs:
            for m in self.target_m_list:
                import_address = "{}/sequencing_models/validated_"+kwargs["trained_parameter"]+".pt"
                self.sequencing_action_NN.network.load_state_dict(torch.load(import_address.format(sys.path[0])))
            print("IMPORT FROM:", import_address)

        if 'store_to' in kwargs:
            self.address_seed = "{}/sequencing_models/" + str(kwargs['address_seed']) + ".pt"
            print("New address seed:", self.address_seed)
        self.rep_memo = []
        self.minibatch_size = 64
        self.rep_memo_size = 256
        self.sequencing_action_NN_training_interval = 5 
        self.sequencing_action_NN_training_time_record = []
        self.sequencing_target_NN_update_interval = 500  
        self.sequencing_target_NN_update_time_record = []
        self.discount_factor = 0.8 
        self.epsilon = 0.3  
        self.loss_time_record = []
        self.loss_record = []
        if kwargs['IQL'] or kwargs['I_DDQN']:
            self.env.process(self.training_process_independent())
            self.env.process(self.update_rep_memo_independent_process())
            self.rep_memo = {} 
            for m in self.target_m_list:
                self.rep_memo[m.m_idx] = []
            self.build_initial_rep_memo = self.build_initial_rep_memo_independent
        else: 
            self.env.process(self.training_process_parameter_sharing())
            self.env.process(self.update_rep_memo_parameter_sharing_process())
            self.build_initial_rep_memo = self.build_initial_rep_memo_parameter_sharing
        self.env.process(self.warm_up_process())
        self.env.process(self.update_training_setting_process())
        self.env.process(self.update_learning_rate_process())

    def warm_up_process(self):
        for idx,func in enumerate(self.func_list):
            self.func_selection = idx
            print('set to rule {}'.format(func))
            yield self.env.timeout(self.warm_up/(2*(len(self.func_list)+1)))
        for m in self.target_m_list:
            m.job_sequencing = self.action_random_exploration
        print("start random exploration from time {} till time {}".format(self.env.now, self.warm_up))
        yield self.env.timeout(self.warm_up - self.env.now - 1)
        self.build_initial_rep_memo()
        for m in self.target_m_list:
            m.job_sequencing = self.action_DRL

    def action_default(self, sqc_data, **kwargs):
        m_idx = sqc_data[-1]
        if self.GPrule_action:
            if 'GPrule_action_data' in kwargs:
                sqc_data_GPrule_action = kwargs['GPrule_action_data']
                job_position = sequencing.GP_pair_S_test(sqc_data_GPrule_action,self.sequencing_GPtree_list[0])
        else:
            job_position = sequencing.FIFO(sqc_data)
        j_idx = sqc_data[-2][job_position]
        return job_position

    def action_warm_up(self, sqc_data, **kwargs):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        a_t = torch.tensor(self.func_selection)
        if self.GPrule_action:
            sqc_data_GPrule_action = kwargs['GPrule_action_data']
            job_position = sequencing.GP_pair_S_test(sqc_data_GPrule_action,self.sequencing_GPtree_list[self.func_selection])

            if self.use_build_experience_strategy:
                sequencing_GPtree_list_decision_True = []
                for i in range(len(self.sequencing_GPtree_list)):
                    job_position_i = sequencing.GP_pair_S_test(sqc_data_GPrule_action,self.sequencing_GPtree_list[i])
                    sequencing_GPtree_list_decision_True.append(job_position_i)
                for i in range(len(self.sequencing_GPtree_list)):
                    if i != self.func_selection:
                        if sequencing_GPtree_list_decision_True[a_t] == sequencing_GPtree_list_decision_True[i]:
                            i_torch = torch.tensor(i)
                            j_idx = sqc_data[-2][job_position]
                            self.build_experience(j_idx,m_idx,s_t,i_torch)
        else:
            job_position = self.func_list[self.func_selection](sqc_data)
            if self.use_build_experience_strategy:
                sequencing_GPtree_list_decision_True = []
                for i in range(len(self.func_list)):
                    job_position_i = self.func_list[i](sqc_data)
                    sequencing_GPtree_list_decision_True.append(job_position_i)
                for i in range(len(self.func_list)):
                    if i != self.func_selection:
                        if sequencing_GPtree_list_decision_True[a_t] == sequencing_GPtree_list_decision_True[i]:
                            i_torch = torch.tensor(i)
                            j_idx = sqc_data[-2][job_position]
                            self.build_experience(j_idx,m_idx,s_t,i_torch)
        j_idx = sqc_data[-2][job_position]
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    def action_random_exploration(self, sqc_data, **kwargs):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        self.func_selection = np.random.randint(len(self.func_list))
        a_t = torch.tensor(self.func_selection)
        if self.GPrule_action:
            sqc_data_GPrule_action = kwargs['GPrule_action_data']
            job_position = sequencing.GP_pair_S_test(sqc_data_GPrule_action,self.sequencing_GPtree_list[a_t])
            if self.use_build_experience_strategy:
                sequencing_GPtree_list_decision_True = []
                for i in range(len(self.sequencing_GPtree_list)):
                    job_position_i = sequencing.GP_pair_S_test(sqc_data_GPrule_action, self.sequencing_GPtree_list[i])
                    sequencing_GPtree_list_decision_True.append(job_position_i)
                for i in range(len(self.sequencing_GPtree_list)):
                    if i != self.func_selection:
                        if sequencing_GPtree_list_decision_True[a_t] == sequencing_GPtree_list_decision_True[i]:
                            i_torch = torch.tensor(i)
                            j_idx = sqc_data[-2][job_position]
                            self.build_experience(j_idx, m_idx, s_t, i_torch)
        else:
            job_position = self.func_list[a_t](sqc_data)
            if self.use_build_experience_strategy:
                sequencing_GPtree_list_decision_True = []
                for i in range(len(self.func_list)):
                    job_position_i = self.func_list[i](sqc_data)
                    sequencing_GPtree_list_decision_True.append(job_position_i)
                for i in range(len(self.func_list)):
                    if i != self.func_selection:
                        if sequencing_GPtree_list_decision_True[a_t] == sequencing_GPtree_list_decision_True[i]:
                            i_torch = torch.tensor(i)
                            j_idx = sqc_data[-2][job_position]
                            self.build_experience(j_idx,m_idx,s_t,i_torch)
        j_idx = sqc_data[-2][job_position]
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    def action_sqc_rule(self, sqc_data, **kwargs):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        if random.random() < self.epsilon:
            a_t = torch.tensor(np.random.randint(0, self.output_size))
        else:
            values=[]
            for vec in range(len(self.vetor_rule)):
                s_t_combine=torch.cat([s_t, self.vetor_rule[vec]], dim=0)
                value = self.sequencing_action_NN.forward(s_t_combine.reshape([1,1,50]),m_idx)
                values.append(value)
            values_tensor = torch.stack(values)
            a_t = torch.argmax(values_tensor)
        if self.GPrule_action:
            sqc_data_GPrule_action = kwargs['GPrule_action_data']
            job_position = sequencing.GP_pair_S_test(sqc_data_GPrule_action,self.sequencing_GPtree_list[a_t])

            if self.use_build_experience_strategy:
                sequencing_GPtree_list_decision_True = []
                for i in range(len(self.sequencing_GPtree_list)):
                    job_position_i = sequencing.GP_pair_S_test(sqc_data_GPrule_action,self.sequencing_GPtree_list[i])
                    sequencing_GPtree_list_decision_True.append(job_position_i)
                for i in range(len(self.sequencing_GPtree_list)):
                    if i != self.func_selection:
                        if sequencing_GPtree_list_decision_True[a_t] == sequencing_GPtree_list_decision_True[i]:
                            i_torch = torch.tensor(i)
                            j_idx = sqc_data[-2][job_position]
                            self.build_experience(j_idx, m_idx, s_t, i_torch)
        else:
            job_position = self.func_list[a_t](sqc_data)
            if self.use_build_experience_strategy:
                sequencing_GPtree_list_decision_True = []
                for i in range(len(self.func_list)):
                    job_position_i = self.func_list[i](sqc_data)
                    sequencing_GPtree_list_decision_True.append(job_position_i)
                for i in range(len(self.func_list)):
                    if i != self.func_selection:
                        if sequencing_GPtree_list_decision_True[a_t] == sequencing_GPtree_list_decision_True[i]:
                            i_torch = torch.tensor(i)
                            j_idx = sqc_data[-2][job_position]
                            self.build_experience(j_idx,m_idx,s_t,i_torch)
        j_idx = sqc_data[-2][job_position]
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    def state_multi_channel(self, sqc_data):
        in_system_job_no = self.job_creator.in_system_job_no
        local_job_no = len(sqc_data[0])
        arriving_jobs = np.where(self.job_creator.next_wc_list == sqc_data[-3])[0]
        arriving_job_no = arriving_jobs.size
        if arriving_job_no:
            arriving_job_time = (self.job_creator.release_time_list[arriving_jobs] - self.env.now).mean()
            arriving_job_slack = (self.job_creator.arriving_job_slack_list[arriving_jobs]).mean()
        else:
            arriving_job_time = 0
            arriving_job_slack = 0
        global_comp_rate = self.job_creator.comp_rate
        global_realized_tard_rate = self.job_creator.realized_tard_rate
        global_exp_tard_rate = self.job_creator.exp_tard_rate
        available_time = (self.job_creator.available_time_list - self.env.now).clip(0,None)
        rem_pt = []
        for m in self.m_list:
            for x in m.remaining_pt_list:
                rem_pt += x.tolist()
        pt_share = available_time[sqc_data[-1]] / sum(available_time)
        global_pt_CV = np.std(rem_pt) / np.mean(rem_pt)
        local_pt_sum = np.sum(sqc_data[0])
        local_pt_mean = np.mean(sqc_data[0])
        local_pt_min = np.min(sqc_data[0])
        local_pt_CV = np.std(sqc_data[0]) / local_pt_mean
        local_remaining_pt_sum = np.sum(sqc_data[1])
        local_remaining_pt_mean = np.mean(sqc_data[1])
        local_remaining_pt_max = np.max(sqc_data[1])
        local_remaining_pt_CV = np.std(sqc_data[1]) / local_remaining_pt_mean
        avlm_mean = np.mean(sqc_data[8])
        avlm_min = np.min(sqc_data[8])
        avlm_CV = np.std(sqc_data[8]) / avlm_mean
        time_till_due = sqc_data[5]
        realized_tard_rate = time_till_due[time_till_due<0].size / local_job_no
        ttd_sum = time_till_due.sum()
        ttd_mean = time_till_due.mean()
        ttd_min = time_till_due.min()
        ttd_CV = (time_till_due.std() / ttd_mean).clip(-2,2)
        slack = sqc_data[6]
        exp_tard_rate = slack[slack<0].size / local_job_no
        slack_sum = slack.sum()
        slack_mean = slack.mean()
        slack_min = slack.min()
        slack_CV = (slack.std() / slack_mean).clip(-2,2)
        no_info = [in_system_job_no, arriving_job_no, local_job_no]
        pt_info = [local_pt_sum, local_pt_mean, local_pt_min]
        remaining_pt_info = [local_remaining_pt_sum, local_remaining_pt_mean, local_remaining_pt_max, avlm_mean, avlm_min]
        ttd_slack_info = [ttd_mean, ttd_min, slack_mean, slack_min, arriving_job_slack]
        progression = [pt_share, global_comp_rate, global_realized_tard_rate, global_exp_tard_rate]
        heterogeneity = [global_pt_CV, local_pt_CV, ttd_CV, slack_CV, avlm_CV]
        s_t = np.nan_to_num(np.concatenate([no_info, pt_info, remaining_pt_info, ttd_slack_info, progression, heterogeneity]),nan=0,posinf=1,neginf=-1)
        s_t = torch.tensor(s_t, dtype=torch.float)
        return s_t

    def build_experience(self,j_idx,m_idx,s_t,a_t):
        if self.use_build_experience_strategy:
            if self.env.now in self.job_creator.incomplete_rep_memo[m_idx]:
                self.job_creator.incomplete_rep_memo[m_idx][self.env.now].append([s_t, a_t])
            else:
                self.job_creator.incomplete_rep_memo[m_idx][self.env.now] = [[s_t, a_t]]
        else:
            self.job_creator.incomplete_rep_memo[m_idx][self.env.now] = [s_t, a_t]

    def build_initial_rep_memo_parameter_sharing(self):
        for m in self.target_m_list:
            self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
            self.job_creator.rep_memo[m.m_idx] = []
        self.rep_memo_TDerror = torch.ones(len(self.rep_memo),dtype=torch.float)
        print('INITIALIZATION - replay_memory')
        print(tabulate(self.rep_memo, headers = ['s_t','a_t','s_t+1','r_t']))
        print('INITIALIZATION - size of replay memory:',len(self.rep_memo))
        print('---------------------------initialization accomplished-----------------------------')

    def update_rep_memo_parameter_sharing_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for m in self.m_list:
                self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
                self.rep_memo_TDerror = torch.cat([self.rep_memo_TDerror, torch.ones(len(self.job_creator.rep_memo[m.m_idx]),dtype=torch.float)])
                self.job_creator.rep_memo[m.m_idx] = []
            if len(self.rep_memo) > self.rep_memo_size:
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
                self.rep_memo_TDerror = self.rep_memo_TDerror[truncation:]
            yield self.env.timeout(self.sequencing_action_NN_training_interval*10)

    def check_parameter(self):
        print('------------- Training Parameter Check -------------')
        print("Address seed:",self.address_seed)
        print('Rwd.Func.:',self.target_m_list[0].reward_function.__name__)
        print('State Func.:',self.build_state.__name__)
        print('ANN:',self.sequencing_action_NN.__class__.__name__)
        print('Discount rate:',self.discount_factor)
        print('*** SCENARIO:')
        print("Configuration: {} work centers, {} machines".format(len(self.job_creator.wc_list),len(self.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('------------------------------------------------------------')

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
        loss_record.set_title("Sequencing Agent Training Loss / {}-machine per work centre test".format(int(len(self.m_list)/len(self.job_creator.wc_list))))
        fig.subplots_adjust(top=0.8, bottom=0.1, right=0.9)
        plt.show()
        if 'save' in kwargs and kwargs['save'] == 1:
            addressPNG = sys.path[0]+"/sequencing_models/scenario_" + kwargs['dataset_name'] + "/SA_loss_{}wc_{}m_Meng_{}.png".format(len(self.job_creator.wc_list),len(self.m_list), kwargs['seed'])
            fig.savefig(addressPNG, dpi=500, bbox_inches='tight')
            print('figure saved to'+addressPNG)
            addressPDF = sys.path[0] + "/sequencing_models/scenario_" + kwargs['dataset_name'] + "/SA_loss_{}wc_{}m_Meng_{}.pdf".format(len(self.job_creator.wc_list), len(self.m_list), kwargs['seed'])
            fig.savefig(addressPDF, dpi=500, bbox_inches='tight')
            print('figure saved to' + addressPDF)
        return

    def reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5))
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.sqc_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.sqc_reward_record).transpose()[1]
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
            fig.savefig(sys.path[0]+"/sequencing_models/scenario_" + kwargs['dataset_name'] + "/SA_reward_{}wc_{}m_Meng_{}.png".format(len(self.job_creator.wc_list),len(self.m_list),kwargs['seed']), dpi=500, bbox_inches='tight')
        return

    def culmulative_reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5))
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.sqc_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.sqc_reward_record).transpose()[1]
        culmulative_rewards = []
        for i in range(len(rewards)):
            if i == 0:
                culmulative_rewards.append(rewards[0])
            else:
                cumulative_reward_i = culmulative_rewards[i - 1] * self.discount_factor + rewards[i]
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
            fig.savefig(sys.path[0]+"/sequencing_models/scenario_" + kwargs['dataset_name'] + "/SA_culmulative_reward_{}wc_{}m_Meng_{}.png".format(len(self.job_creator.wc_list),len(self.m_list),kwargs['seed']), dpi=500, bbox_inches='tight')
        return

    def save_intermediate_policy(self):
        path_seed = "{}/sequencing_models/scenario_" + self.dataset_name + "/run_" + str(self.seed) + "_policies/"
        path = path_seed.format(sys.path[0])
        isExists = os.path.exists(path)
        index = int((self.env.now-self.warm_up-1)/self.sequencing_target_NN_update_interval)
        if not isExists and index == 0:
            os.makedirs(path)
        elif isExists and index == 0:
            shutil.rmtree(path)
            os.makedirs(path)
        intermediate_address_seed = path_seed + "/run_" + str(self.seed) + "_MC_rwd_time" + str(index) + ".pt"
        address = intermediate_address_seed.format(sys.path[0])
        torch.save(self.sequencing_action_NN.network.state_dict(), address)

    def training_process_parameter_sharing(self):
        yield self.env.timeout(self.warm_up)
        for i in range(10):
            self.train()
        while self.env.now < self.span:
            self.train()
            yield self.env.timeout(self.sequencing_action_NN_training_interval)
        print('FINAL- replay_memory')
        print(tabulate(self.rep_memo, headers = ['s_t','a_t','s_t+1','r_t']))
        print('FINAL - size of replay memory:',len(self.rep_memo))
        address = self.address_seed.format(sys.path[0])
        torch.save(self.sequencing_action_NN.network.state_dict(), address)
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))

    def update_training_setting_process(self):
        yield self.env.timeout(self.warm_up+1)
        while self.env.now < self.span:
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            if self.save_intermediate:
                self.save_intermediate_policy()
            print('--------------------------------------------------------')
            print('the target network and epsilion are updated at time %s' % self.env.now)
            print('--------------------------------------------------------')
            yield self.env.timeout(self.sequencing_target_NN_update_interval)

    def update_learning_rate_process(self):
        yield self.env.timeout(self.warm_up)
        reduction = (self.sequencing_action_NN.lr - self.sequencing_action_NN.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            self.sequencing_action_NN.lr -= reduction
            if self.sequencing_action_NN.lr < 0.001:
                self.sequencing_action_NN.lr = 0.001
            self.epsilon -= 0.01
            if self.epsilon < 0.1:
                self.epsilon = 0.1
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.sequencing_action_NN.lr, self.env.now))
            print('--------------------------------------------------------')

    def train_Double_DQN(self):
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        Q_0 = self.sequencing_action_NN.forward(sample_s0_batch)
        current_value = Q_0.gather(1, sample_a0_batch)
        Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch).detach()
        Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch).detach()
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        next_state_value *= self.discount_factor
        target_value = (sample_r0_batch + next_state_value)
        loss = self.sequencing_action_NN.loss_func(current_value, target_value).detach()
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))
        self.sequencing_action_NN.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.sequencing_action_NN.optimizer.step()

    def train_validated(self):
        Q_0_temp=[]
        Q_1_action_temp=[]
        Q_1_target_temp=[]
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        for vec in range(len(self.vetor_rule)):
            vetor_rule_expanded = self.vetor_rule[vec].T.unsqueeze(0).unsqueeze(1).repeat(sample_s0_batch.size(0), 1, 1)
            sample_s0_batch_combined = torch.cat([sample_s0_batch, vetor_rule_expanded], dim=2)
            value = self.sequencing_action_NN.forward(sample_s0_batch_combined)
            Q_0_temp.append(value)
        Q_0 = torch.cat(Q_0_temp, dim=1)
        current_value = Q_0.gather(1, sample_a0_batch)
        for vec in range(len(self.vetor_rule)):
            vetor_rule_expanded = self.vetor_rule[vec].T.unsqueeze(0).unsqueeze(1).repeat(sample_s1_batch.size(0), 1, 1)
            sample_s1_batch_combined = torch.cat([sample_s1_batch, vetor_rule_expanded], dim=2)
            Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch_combined).detach()
            Q_1_action_temp.append(Q_1_action)
            Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch_combined).detach()
            Q_1_target_temp.append(Q_1_target)
        Q_1_action = torch.cat(Q_1_action_temp, dim=1)
        Q_1_target = torch.cat(Q_1_target_temp, dim=1)
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        next_state_value *= self.discount_factor
        target_value = (sample_r0_batch + next_state_value)
        loss = self.sequencing_action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))
        self.sequencing_action_NN.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.sequencing_action_NN.optimizer.step()

class network_validated(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_validated, self).__init__()
        self.lr = 0.01
        self.input_size = input_size
        self.output_size = output_size
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        layer_1 = 128
        layer_2 = 128
        layer_3 = 256
        layer_4 = 128
        layer_5 = 64
        layer_6 = 20
        layer_7 = 5
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.subsequent_module = nn.Sequential(
                                nn.Linear(57, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, layer_7),
                                nn.Tanh(),
                                nn.Linear(layer_7, 1)
                                )
        self.loss_func = F.smooth_l1_loss
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x,*args):
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        x = self.network[4](x)
        return x

class network_value_based(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_value_based, self).__init__()
        self.lr = 0.01
        self.input_size = input_size
        self.output_size = output_size
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        self.norm_layer = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten()
                                )
        self.FC_layers = nn.Sequential(
                                nn.Linear(self.flattened_input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        self.loss_func = F.smooth_l1_loss
        self.network = nn.ModuleList([self.norm_layer, self.FC_layers])
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        x = self.network[0](x)
        x = self.network[1](x)
        return x