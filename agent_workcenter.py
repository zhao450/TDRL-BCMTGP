import simpy
import sys
sys.path
import random
import numpy as np
import torch
from tabulate import tabulate
import routing
import KmeansWithMutGuideMTGP.niching.RoutingDecisionSituation as RoutingDecisionSituation

class workcenter:
    def __init__(self, env, index, m_list, *args, **kwargs):
        self.env = env

        self.job_routing_tree = '' 

        self.GPrule_action = False
        if 'GPrule_action' in kwargs and kwargs['GPrule_action']:
            self.GPrule_action = True

        self.Single_agent = False
        if 'Single_agent' in kwargs and kwargs['Single_agent']: 
            self.Single_agent = True

        self.GPrule_ensemble = False
        if 'GPrule_ensemble' in kwargs and kwargs['GPrule_ensemble']: 
            self.GPrule_ensemble = True

        self.getDecisionSituation = False
        if 'getDecisionSituation' in kwargs and kwargs['getDecisionSituation']: 
            self.getDecisionSituation = True
            self.RoutingDecisionSituationList = kwargs['RoutingDecisionSituationList']

        self.getAllDecisions = False

        self.use_build_experience_strategy = False
        if 'use_build_experience_strategy' in kwargs and kwargs['use_build_experience_strategy']:
            self.use_build_experience_strategy = True

        self.m_list = m_list
        self.m_no = len(self.m_list)
        self.m_idx_list = [m.m_idx for m in m_list]
        self.wc_idx = index
        self.queue = []
        self.sequence_list = [] 
        self.pt_list = []
        self.remaining_pt_list = [] 
        self.due_list = [] 
        self.weight_list = [] 
        self.release_time_list = []  
        self.routing_data = []
        self.routing_data_generation_GPrule_action = [] 
        self.print_info = True
        self.routing_event = self.env.event()
        self.build_routing_experience = self.complete_experience_full
        self.incomplete_experience = {}
        self.rep_memo = []
        if 'rule' in kwargs:
            order = "self.job_routing = routing." + kwargs['rule']
            try:
                exec(order)
                print("workcenter {} uses {} routing rule".format(self.wc_idx, kwargs['rule']))
            except:
                print("Rule assigned to workcenter {} is invalid !".format(self.wc_idx))
                raise Exception
        else:
            self.job_routing = routing.EA

    def setAllRoutingDecisions(self, AllRoutingDecisions):
        self.getAllDecisions = True
        self.AllRoutingDecisions = AllRoutingDecisions

    def setGetDecisionSituation(self,RoutingDecisionSituationList):
        self.getDecisionSituation = True
        self.RoutingDecisionSituationList = RoutingDecisionSituationList

    def setJobRoutingTree(self, tree):
        self.job_routing_tree = tree

    def getRoutingDecisionSituations(self):
        return self.RoutingDecisionSituationList
    def initialization(self, job_creator):
        self.job_creator = job_creator
        self.dummy_pt = np.ones(self.m_no)*self.job_creator.avg_pt
        if self.print_info:
            print('work center {} contains machine {}'.format(self.wc_idx, self.m_idx_list))
            print('Initial %s jobs at workcenter %s are:'%(len(self.queue), self.wc_idx))
            job_info = [[self.queue[i],self.sequence_list[i], self.pt_list[i], self.due_list[i]] for i in range(len(self.queue))]
            print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','due']))
            print('************************************')
        for m in self.m_list:
            m.state_update_all()
        for i,m in enumerate(self.m_list):
            remaining_ptl = self.remaining_pt_list.pop(0)
            current_pt = remaining_ptl[0]
            estimated_slack_time = self.due_list[0] - self.env.now - np.sum(remaining_ptl.max(axis=1))
            remaining_ptl = np.delete(remaining_ptl, 0 ,axis=0) 
            self.m_list[i].queue.append(self.queue.pop(0))
            self.m_list[i].weight_list.append(self.weight_list.pop(0))
            self.m_list[i].release_time_list.append(self.release_time_list.pop(0)) 
            self.m_list[i].sequence_list.append(self.sequence_list.pop(0))
            self.m_list[i].pt_list.append(self.pt_list.pop(0))
            self.m_list[i].slack_upon_arrival.append(estimated_slack_time)
            self.m_list[i].remaining_pt_list.append(remaining_ptl)
            self.m_list[i].due_list.append(self.due_list.pop(0))
            self.m_list[i].arrival_time_list.append(self.env.now)
            self.m_list[i].state_update_after_job_arrival(15)
        self.state_update_before_routing()
        self.env.process(self.routing())
    def routing(self):
        while True:
            yield self.routing_event
            for j in range(len(self.queue)):
                self.state_update_before_routing()
                remaining_ptl = self.remaining_pt_list.pop(0)
                current_pt = remaining_ptl[0]
                next_pt = 0
                WKR = 0
                NOR = 0
                if len(remaining_ptl) > 1:
                    next_pt = remaining_ptl[1]
                    remaining_ptl_sum = np.sum(remaining_ptl, axis=0)
                    WKR = remaining_ptl_sum - current_pt
                    NOR = len(remaining_ptl) - 1
                estimated_slack_time = self.due_list[0] - self.env.now - self.least_waiting - np.sum(remaining_ptl.max(axis=1))
                remaining_ptl = np.delete(remaining_ptl, 0, axis=0)
                remaining_pt = remaining_ptl.sum()
                if self.job_routing.__name__ == 'GP_pair_R' or self.job_routing.__name__ == 'GP_pair_R_test' or self.job_routing.__name__ == 'GP_Evolved_R' or self.job_routing.__name__ == 'GP_evolve_R':
                    selected_machine_index = self.job_routing(self.job_routing_tree, self.queue[0], self.routing_data,
                                                              current_pt, next_pt, (self.env.now - self.release_time_list[0]), 
                                                              WKR, NOR, self.weight_list[0], (self.env.now - self.release_time_list[0]),
                                                              estimated_slack_time)
                    if self.getDecisionSituation:
                        routing_data_all = [self.queue[0], self.routing_data, current_pt, next_pt,
                                            (self.env.now - self.release_time_list[0]),
                                            WKR, NOR, self.weight_list[0], (self.env.now - self.release_time_list[0]),
                                            estimated_slack_time]
                        routingDecision = RoutingDecisionSituation.RoutingDecisionSituation(routing_data_all)
                        self.RoutingDecisionSituationList.append(routingDecision)
                elif self.GPrule_action or self.GPrule_ensemble:
                    selected_machine_index = self.job_routing(self.queue[0], self.routing_data,
                                                              current_pt, estimated_slack_time, self.wc_idx,
                                                              sum_remaining_ptl=np.sum(remaining_ptl.mean(axis=1)), len_remaining_ptl=len(remaining_ptl),
                                                              next_pt = next_pt, OWT=(self.env.now - self.release_time_list[0]),
                                                              WKR=WKR, NOR=NOR, weight_list=self.weight_list[0],
                                                              waiting_time=(self.env.now - self.release_time_list[0]),
                                                              GPrule_action_data=self.routing_data_generation_GPrule_action)
                else:
                    selected_machine_index = self.job_routing(self.queue[0], self.routing_data,
                                                              current_pt, estimated_slack_time, self.wc_idx,
                                                              np.sum(remaining_ptl.mean(axis=1)), len(remaining_ptl))

                if self.getAllDecisions:
                    self.AllRoutingDecisions.append(selected_machine_index)
                increased_available_time = current_pt[selected_machine_index]
                self.m_list[selected_machine_index].queue.append(self.queue.pop(0))
                self.m_list[selected_machine_index].sequence_list.append(self.sequence_list.pop(0))
                self.m_list[selected_machine_index].pt_list.append(self.pt_list.pop(0))
                self.m_list[selected_machine_index].slack_upon_arrival.append(estimated_slack_time)
                self.m_list[selected_machine_index].remaining_pt_list.append(remaining_ptl)
                self.m_list[selected_machine_index].due_list.append(self.due_list.pop(0))
                self.m_list[selected_machine_index].arrival_time_list.append(max(self.env.now,self.m_list[selected_machine_index].release_time))
                self.m_list[selected_machine_index].weight_list.append(self.weight_list.pop(0)) 
                self.m_list[selected_machine_index].release_time_list.append(self.release_time_list.pop(0))  
                self.m_list[selected_machine_index].state_update_after_job_arrival(increased_available_time)
                try:
                    self.m_list[selected_machine_index].sufficient_stock.succeed()
                except:
                    pass
            self.routing_event = self.env.event()
    def state_update_before_routing(self):
        self.routing_data = [machine.routing_data_generation() for machine in self.m_list]
        self.routing_data_generation_GPrule_action = [machine.routing_data_generation_GPrule_action() for machine in self.m_list] 
        self.least_waiting = np.min(self.routing_data, axis=0)[1]
        avg = np.average(np.array(self.routing_data).clip(0),axis=0)
        self.average_workcontent = avg[0]
        self.average_waiting = avg[1]
        self.machine_condition = np.array([machine.working_event.triggered*1 for machine in self.m_list])

    def complete_experience_full(self, job_idx, slack_change, critical_level_R):
        self.state_update_before_routing()
        s_t = self.build_state(self.routing_data, self.dummy_pt, 0, self.wc_idx)
        r_t = torch.tensor(np.clip(slack_change*critical_level_R/20, -1, 1),dtype=torch.float)
        self.job_creator.rt_reward_record.append([self.env.now, r_t])
        if self.use_build_experience_strategy:
            all_incomplete_experience_job_idx = self.incomplete_experience[job_idx]
            if len(all_incomplete_experience_job_idx) < 4 or len(self.rep_memo) == 0:
                for each in all_incomplete_experience_job_idx:
                    each += [r_t, s_t]
                    self.rep_memo.append(each)
            self.incomplete_experience.pop(job_idx)
        else:
            self.incomplete_experience[job_idx] += [r_t, s_t]
            self.rep_memo.append(self.incomplete_experience.pop(job_idx))

    def complete_experience_global_reward(self, job_idx, slack_change, critical_level_R):
        self.state_update_before_routing()
        s_t = self.build_state(self.routing_data, self.dummy_pt, 0, self.wc_idx)
        self.incomplete_experience[job_idx] += [s_t]

