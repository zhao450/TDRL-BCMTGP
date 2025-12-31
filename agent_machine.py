import simpy
import sys
sys.path
import random
import numpy as np
import torch
from tabulate import tabulate
import sequencing
import KmeansWithMutGuideMTGP.niching.SequencingDecisionSituation as SequencingDecisionSituation

class machine:
    def __init__(self, env, index, *args, **kwargs):
        self.env = env
        self.m_idx = index

        self.job_sequencing_tree = ''

        self.GPrule_action = False
        if 'GPrule_action' in kwargs and kwargs['GPrule_action']: 
            self.GPrule_action = True

        self.Single_agent = False
        if 'Single_agent' in kwargs and kwargs['Single_agent']: 
            self.Single_agent = True

        self.getDecisionSituation = False
        if 'getDecisionSituation' in kwargs and kwargs['getDecisionSituation']: 
            self.getDecisionSituation = True
            self.SequencingDecisionSituationList = kwargs['SequencingDecisionSituationList']

        self.getAllDecisions = False

        self.use_build_experience_strategy = False
        if 'use_build_experience_strategy' in kwargs and kwargs['use_build_experience_strategy']: 
            self.use_build_experience_strategy = True

        self.use_tree_lstm=False
        if 'use_tree_lstm' in kwargs and kwargs['use_tree_lstm']:
            self.use_tree_lstm = True

        self.queue = []
        self.sequence_list = [] 
        self.pt_list = [] 
        self.remaining_pt_list = [] 
        self.due_list = [] 
        self.arrival_time_list = [] 
        self.waited_time = [] 
        self.weight_list = [] 
        self.release_time_list = [] 
        self.slack_upon_arrival = [] 
        self.no_jobs_record = []
        self.decision_point = 0
        self.release_time = 0
        self.cumulative_run_time = 0
        self.global_exp_tard_rate = 0
        self.sufficient_stock = self.env.event()
        self.working_event = self.env.event()
        self.restart_time = 0
        self.count = 0
        self.count2 = 0
        if not len(self.queue):
            self.sufficient_stock.succeed()
        self.working_event.succeed()
        self.print_info = True
        self.routing_global_reward = False 
        self.breakdown_record = []
        self.EMA_slack_change = 0
        self.EMA_realized_tardiness = 0
        self.EMA_alpha = 0.1
        if 'rule' in kwargs:
            order = "self.job_sequencing = sequencing." + kwargs['rule']
            try:
                exec(order)
                print("machine {} uses {} sequencing rule".format(self.m_idx, kwargs['rule']))
            except:
                print("Rule assigned to machine {} is invalid !".format(self.m_idx))
                raise Exception
        else:
            self.job_sequencing = sequencing.FIFO
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()


    def setAllSequencingDecisions(self,AllSequencingDecisions):
        self.getAllDecisions = True
        self.AllSequencingDecisions = AllSequencingDecisions

    def setGetDecisionSituation(self,SequencingDecisionSituationList):
        self.getDecisionSituation = True
        self.SequencingDecisionSituationList = SequencingDecisionSituationList

    def setJobSequencingTree(self, tree):
        self.job_sequencing_tree = tree

    def getSequencingDecisionSituations(self):
        return self.SequencingDecisionSituationList


    def get_realized_tard_rate_before(self):
        time_till_due = np.array(self.due_list) - self.env.now
        tardy_jobs = (time_till_due < 0).sum()
        if len(self.queue) == 0:
            return 0
        return tardy_jobs / len(self.queue)
    
    def get_realized_tard_rate_after(self):
        time_till_due = np.array(self.due_list) - self.env.now
        tardy_jobs = (time_till_due < 0).sum()
        if len(self.queue) == 0:
            return 0
        return tardy_jobs / len(self.queue)


    def initialization(self, machine_list, workcenter_list, job_creator, assigned_wc):
        self.m_list = machine_list
        self.m_no = len(self.m_list)
        self.wc_list = workcenter_list
        self.wc = assigned_wc
        self.wc_idx = assigned_wc.wc_idx
        self.no_ops = len(self.wc_list)
        self.job_creator = job_creator
        if self.print_info:
            print('machine {} belongs to work center {}'.format(self.m_idx,assigned_wc.wc_idx))
            print('Initial %s jobs at machine %s are:'%(len(self.queue), self.m_idx))
            job_info = [[self.queue[i],self.sequence_list[i], self.pt_list[i], self.slack_upon_arrival[i], self.due_list[i]] for i in range(len(self.queue))]
            print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','slack','due']))
            print('************************************')
        self.state_update_all()
        self.update_global_info_progression()
        self.env.process(self.production())
    def production(self): 
        if not len(self.queue):
            yield self.env.process(self.starvation())
        self.state_update_all()
        while True:
            self.decision_point = self.env.now
            self.no_jobs_record.append(len(self.queue))
            if len(self.queue)-1>0:
                if self.job_sequencing.__name__ == 'GP_pair_S' or self.job_sequencing.__name__ == 'GP_pair_S_test' or self.job_sequencing.__name__ == 'GP_Evolved_S' or self.job_sequencing.__name__ == 'GP_evolve_S':
                    sequencing_data_generation = self.sequencing_data_generation()
                    self.position = self.job_sequencing(sequencing_data_generation, self.job_sequencing_tree)
                    if self.getDecisionSituation:
                        sequencingDecision = SequencingDecisionSituation.SequencingDecisionSituation(sequencing_data_generation)
                        self.SequencingDecisionSituationList.append(sequencingDecision)
                elif self.GPrule_action:
                    self.position = self.job_sequencing(self.sequencing_data_generation(), GPrule_action_data=self.sequencing_data_generation_GPrule_action())
                else:
                    self.position = self.job_sequencing(self.sequencing_data_generation())
                if self.getAllDecisions:
                    self.AllSequencingDecisions.append(self.position)
                self.job_idx = self.queue[self.position]
                self.before_operation()
                self.count += 1
                if len(self.queue)-2:
                    self.count2 += 1
            else:
                self.position = 0
                self.job_idx = self.queue[self.position]
            pt = self.pt_list[self.position][self.m_idx] 
            wait = self.env.now - self.arrival_time_list[self.position]
            self.update_global_info_progression()
            self.update_global_info_anticipation(pt)
            self.record_production(pt, wait)
            yield self.env.timeout(pt)
            self.cumulative_run_time += pt
            self.after_operation()
            if not len(self.queue):
                yield self.env.process(self.starvation())
                self.state_update_all()
    def starvation(self):
        self.sufficient_stock = self.env.event()
        yield self.sufficient_stock
        if not self.working_event.triggered:
            yield self.env.process(self.breakdown())

    def breakdown(self):
        print('********', self.m_idx, "breakdown at time", self.env.now, '********')
        start = self.env.now
        self.available_time = self.restart_time + self.cumulative_pt
        yield self.working_event
        self.breakdown_record.append([(start, self.env.now-start), self.m_idx])
        print('********', self.m_idx, 'brekdown ended, restart production at time', self.env.now, '********')


    def before_operation(self):
        self.waiting_jobs = len(self.queue)
        time_till_due = np.array(self.due_list) - self.env.now
        self.before_op_ttd = time_till_due
        self.before_op_ttd_chosen = self.before_op_ttd[self.position]
        self.before_op_ttd_loser = np.delete(self.before_op_ttd, self.position)
        tardy_jobs = len(time_till_due[time_till_due<0])
        self.before_op_realized_tard_rate =tardy_jobs/len(self.queue)
        initial_slack = self.slack_upon_arrival.copy()
        self.before_op_remaining_pt = self.remaining_job_pt + self.current_pt
        self.before_op_remaining_pt_chosen = self.before_op_remaining_pt[self.position]
        self.before_op_remaining_pt_loser = np.delete(self.before_op_remaining_pt, self.position)
        current_slack = time_till_due - self.before_op_remaining_pt
        exp_tardy_jobs = len(current_slack[current_slack<0])
        self.before_op_exp_tard = current_slack[current_slack<0]
        self.before_op_sum_exp_tard = self.before_op_exp_tard.sum()
        self.before_op_slack = current_slack
        self.before_op_sum_slack = self.before_op_slack.sum()
        self.critical_level = 1 - current_slack / (np.absolute(current_slack)+50)
        self.critical_level_chosen  = self.critical_level[self.position]
        self.pt_chosen = self.current_pt[self.position]
        self.initial_slack_chosen = initial_slack[self.position]
        self.before_op_slack_chosen = current_slack[self.position]
        self.before_op_exp_tard_chosen = min(0,self.before_op_slack_chosen)
        self.before_op_winq_chosen = self.winq[self.position]
        self.before_op_slack_loser = np.delete(current_slack, self.position) 
        self.critical_level_loser = np.delete(self.critical_level, self.position)
        self.before_op_sum_exp_tard_loser = self.before_op_slack_loser[self.before_op_slack_loser<0].sum()
        self.before_op_sum_slack_loser = self.before_op_slack_loser.sum()
        self.before_op_winq_loser = np.delete(self.winq, self.position)
        self.before_op_expected_tard_rate = exp_tardy_jobs/len(self.queue)
        self.act_trad_T=self.get_realized_tard_rate_before()

    def after_operation(self):
        if len(self.sequence_list[self.position]):
            next_wc = self.sequence_list[self.position][0]
            self.wc_list[next_wc].queue.append(self.queue.pop(self.position))
            self.wc_list[next_wc].weight_list.append(self.weight_list.pop(self.position))
            self.wc_list[next_wc].release_time_list.append(self.release_time_list.pop(self.position))
            self.wc_list[next_wc].sequence_list.append(np.delete(self.sequence_list.pop(self.position),0))
            self.wc_list[next_wc].pt_list.append(self.pt_list.pop(self.position))
            remaining_ptl = self.remaining_pt_list.pop(self.position)
            self.wc_list[next_wc].remaining_pt_list.append(remaining_ptl)
            current_slack = self.due_list[self.position] - self.env.now - np.sum(remaining_ptl.max(axis=1))
            self.wc_list[next_wc].due_list.append(self.due_list.pop(self.position))
            estimated_slack_time = self.slack_upon_arrival.pop(self.position)
            del self.arrival_time_list[self.position]
            self.slack_change = current_slack - estimated_slack_time
            self.critical_level_R = 1 - current_slack /(np.absolute(current_slack)+ 50)
            self.record_slack_tardiness()
            self.EMA_slack_change += self.EMA_alpha * (self.slack_change - self.EMA_slack_change)
            try:
                self.wc_list[next_wc].routing_event.succeed()
            except:
                pass
            self.state_update_all()
            self.update_global_info_after_operation()
            self.act_trad_T_t=self.get_realized_tard_rate_after()
            self.exp_trad_rate=self.act_trad_T_t

            if self.routing_learning_event.triggered:
                try:
                    self.wc.build_routing_experience(self.job_idx,self.slack_change, self.critical_level_R)
                except:
                    pass
            if self.sequencing_learning_event.triggered:
                self.complete_experience()

        else:
            self.tardiness = np.max([0, self.env.now - self.due_list[self.position]])
            self.job_creator.flowtime_list[self.position] = self.env.now - self.job_creator.flowtime_list[self.position]
            self.EMA_realized_tardiness += self.EMA_alpha * (self.tardiness - self.EMA_realized_tardiness)
            del self.queue[self.position]
            del self.weight_list[self.position]
            del self.release_time_list[self.position]
            del self.sequence_list[self.position]
            del self.pt_list[self.position]
            del self.remaining_pt_list[self.position]
            current_slack = self.due_list[self.position] - self.env.now 
            del self.due_list[self.position]
            estimated_slack_time = self.slack_upon_arrival.pop(self.position)
            del self.arrival_time_list[self.position]
            self.job_creator.record_job_departure()
            self.slack_change = current_slack - estimated_slack_time
            self.critical_level_R = 1 - current_slack /(np.absolute(current_slack)+ 50)
            self.record_slack_tardiness(self.tardiness)
            self.EMA_slack_change += self.EMA_alpha * (self.slack_change - self.EMA_slack_change)
            self.state_update_all()
            self.update_global_info_after_operation()
            self.act_trad_T_t=self.get_realized_tard_rate_after()
            self.exp_trad_rate=self.act_trad_T_t
            if self.routing_learning_event.triggered:
                try:
                    self.wc.build_routing_experience(self.job_idx,self.slack_change, self.critical_level_R)
                except:
                    pass
            if self.sequencing_learning_event.triggered:
                self.complete_experience()
            if self.routing_global_reward:
                self.add_global_reward_RA()
            
    def record_production(self, pt, wait):
        self.job_creator.production_record[self.job_idx][0].append((self.env.now,pt))
        self.job_creator.production_record[self.job_idx][1].append(self.m_idx)
        self.job_creator.production_record[self.job_idx][2].append(wait)

    def record_slack_tardiness(self, *args):
        self.job_creator.production_record[self.job_idx][4].append(self.slack_change)
        if len(args):
            self.job_creator.production_record[self.job_idx].append((self.env.now,args[0]))

    def state_update_all(self):
        self.current_pt = np.array([x[self.m_idx] for x in self.pt_list])
        self.cumulative_pt = self.current_pt.sum() 
        self.available_time = self.env.now + self.cumulative_pt
        self.remaining_job_pt = np.array([sum(x.mean(axis=1)) for x in self.remaining_pt_list])
        self.remaining_no_op = np.array([len(x) for x in self.remaining_pt_list])
        self.next_pt = np.array([x[0].mean() if len(x) else 0 for x in self.remaining_pt_list]) 
        self.completion_rate = np.array([(self.no_ops-len(x)-1)/self.no_ops for x in self.remaining_pt_list])
        self.que_size = len(self.queue) 
        self.time_till_due = np.array(self.due_list) - self.env.now
        self.slack = self.time_till_due - self.current_pt - self.remaining_job_pt
        self.waited_time = self.env.now - np.array(self.arrival_time_list) 
        self.winq = np.array([self.wc_list[x[0]].average_workcontent if len(x) else 0 for x in self.sequence_list])
        self.avlm = np.array([self.wc_list[x[0]].average_waiting if len(x) else 0 for x in self.sequence_list])

    def state_update_after_job_arrival(self, increased_available_time):
        self.current_pt = np.array([x[self.m_idx] for x in self.pt_list])
        self.cumulative_pt = self.current_pt.sum()
        self.available_time = max(self.available_time, self.env.now) + increased_available_time
        self.que_size = len(self.queue)

    def update_global_info_progression(self):
        realized = self.time_till_due.clip(0,1)
        exp = self.slack.clip(0,1)
        self.job_creator.comp_rate_list[self.m_idx] = self.completion_rate
        self.job_creator.comp_rate = np.concatenate(self.job_creator.comp_rate_list).mean()
        self.job_creator.realized_tard_list[self.m_idx] = realized
        self.job_creator.realized_tard_rate = 1 - np.concatenate(self.job_creator.realized_tard_list).mean()
        self.job_creator.exp_tard_list[self.m_idx] = exp
        self.job_creator.exp_tard_rate = 1 - np.concatenate(self.job_creator.exp_tard_list).mean()
        self.job_creator.available_time_list[self.m_idx] = self.available_time
    def update_global_info_anticipation(self,pt):
        current_j_idx = self.queue[self.position]
        self.job_creator.current_j_idx_list[self.m_idx] = current_j_idx
        next_wc = self.sequence_list[self.position][0] if len(self.sequence_list[self.position]) else -1 
        self.job_creator.next_wc_list[self.m_idx] = next_wc 
        self.release_time = self.env.now + pt
        self.job_creator.release_time_list[self.m_idx] = self.release_time 
        job_rempt = self.remaining_job_pt[self.position].sum() - pt
        self.job_creator.arriving_job_rempt_list[self.m_idx] = job_rempt
        job_slack = self.slack[self.position]
        self.job_creator.arriving_job_slack_list[self.m_idx] = job_slack 

    def update_global_info_after_operation(self):
        self.job_creator.next_wc_list[self.m_idx] = -1 

    def routing_data_generation(self):
        if self.job_sequencing.__name__ == 'GP_pair_S' or self.job_sequencing.__name__ == 'GP_pair_S_test' or self.job_sequencing.__name__ == 'GP_Evolved_S' or self.job_sequencing.__name__ == 'GP_evolve_S':
            self.routing_data = \
                [self.que_size, self.cumulative_pt, max(0, self.available_time - self.env.now)]
        else:
            self.routing_data = [self.cumulative_pt, max(0, self.available_time - self.env.now), self.que_size,
                                 self.cumulative_run_time]
        return self.routing_data

    def routing_data_generation_GPrule_action(self):
        if self.job_sequencing.__name__ == 'GP_pair_S' or self.job_sequencing.__name__ == 'GP_pair_S_test' or self.job_sequencing.__name__ == 'GP_Evolved_S' or self.job_sequencing.__name__ == 'GP_evolve_S' or self.GPrule_action:
            self.routing_data = \
                [self.que_size, self.cumulative_pt, max(0, self.available_time - self.env.now)]
        else:
            self.routing_data = [self.cumulative_pt, max(0, self.available_time - self.env.now), self.que_size,
                                 self.cumulative_run_time]
        return self.routing_data

    def sequencing_data_generation(self):
        if self.job_sequencing.__name__ == 'GP_pair_S' or self.job_sequencing.__name__ == 'GP_pair_S_test' or self.job_sequencing.__name__ == 'GP_Evolved_S' or self.job_sequencing.__name__ == 'GP_evolve_S':
            self.sequencing_data = \
                [self.que_size, self.cumulative_pt, max(0, self.env.now - self.available_time),
                 self.current_pt, self.next_pt, self.waited_time, self.remaining_job_pt,
                 self.remaining_no_op,
                 np.array(self.env.now - np.array(self.release_time_list)), self.slack]  
        else:
            self.sequencing_data = \
                [self.current_pt, self.remaining_job_pt, np.array(self.due_list), self.env.now, self.completion_rate, \
                self.time_till_due, self.slack, self.winq, self.avlm, self.next_pt, self.remaining_no_op, self.waited_time, \
                self.wc_idx, self.queue, self.m_idx]
        return self.sequencing_data

    def sequencing_data_generation_GPrule_action(self):
        if self.job_sequencing.__name__ == 'GP_pair_S' or self.job_sequencing.__name__ == 'GP_pair_S_test' or self.job_sequencing.__name__ == 'GP_Evolved_S' or self.job_sequencing.__name__ == 'GP_evolve_S' or self.GPrule_action:
            self.sequencing_data = \
                [self.que_size, self.cumulative_pt, max(0, self.env.now - self.available_time),
                 self.current_pt, self.next_pt, self.waited_time, self.remaining_job_pt,
                 self.remaining_no_op,
                 np.array(self.env.now - np.array(self.release_time_list)), self.slack] 
        else:
            self.sequencing_data = \
                [self.current_pt, self.remaining_job_pt, np.array(self.due_list), self.env.now, self.completion_rate, \
                self.time_till_due, self.slack, self.winq, self.avlm, self.next_pt, self.remaining_no_op, self.waited_time, \
                self.wc_idx, self.queue, self.m_idx]
        return self.sequencing_data
    def complete_experience(self):
        try:
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point]
            local_data = self.sequencing_data_generation()
            s_t = self.build_state(local_data)
            r_t = self.get_reward1() 
            self.job_creator.sqc_reward_record.append([self.env.now, r_t])

            if self.use_build_experience_strategy:
                all_incomplete_experience_decision = self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point]
                if len(all_incomplete_experience_decision) < 4: 
                    for each in all_incomplete_experience_decision:
                        each += [s_t, r_t]
                        self.job_creator.rep_memo[self.m_idx].append(each)
                self.job_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_point)
            else:
                self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point] += [s_t, r_t] 
                complete_exp = self.job_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_point) 
                self.job_creator.rep_memo[self.m_idx].append(complete_exp)
        except:
            pass
    def get_reward0(self):
        if self.pt_chosen <=self.before_op_remaining_pt_loser.mean():
            r_t = 1
        else:
            r_t = 0
        r_t = torch.tensor(r_t, dtype=torch.float)
        return r_t

    def get_reward1(self):
        slack = self.before_op_slack
        critical_level = 1 - slack / (np.absolute(slack) + 60)
        critical_level_chosen = critical_level[self.position]
        critical_level_loser = (np.delete(critical_level, self.position)).mean()
        pt_loser_mean=self.before_op_remaining_pt_loser.mean()
        next_winq_available_time_chosen = self.before_op_winq_chosen
        next_winq_available_time_lose=self.before_op_winq_loser.mean()
        r1=pt_loser_mean*critical_level_chosen-0.2*next_winq_available_time_chosen
        r2=self.pt_chosen*critical_level_loser-0.2*next_winq_available_time_lose
        rwd=((r1-r2)/20).clip(-1,1)
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t
    

    def get_reward2(self):
        if self.act_trad_T_t<self.act_trad_T:
            rwd= 1
        elif self.act_trad_T_t>self.act_trad_T:
            rwd = -1
        elif self.exp_trad_rate<self.before_op_expected_tard_rate:
            rwd = 1
        elif self.exp_trad_rate>self.before_op_expected_tard_rate:
            rwd = -1
        else:
            rwd = 0
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t

    def get_reward3(self):
        rwd=0
        if self.act_trad_T_t<=self.act_trad_T:
            rwd+= 1
        else:
            rwd = rwd-1
        
        if self.exp_trad_rate<=self.before_op_expected_tard_rate:
            rwd += 1
        else:
            rwd =rwd -1
        
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t

    def get_reward4(self):
        if self.act_trad_T_t<self.act_trad_T:
            rwd= 10
        else:
            rwd = 5
        
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t
    
    def get_reward5(self):
        slack = self.before_op_slack
        critical_level = 1 - slack / (np.absolute(slack) + 50)
        critical_level_chosen = critical_level[self.position]
        critical_level_loser = (np.delete(critical_level, self.position)).mean()

        if critical_level_chosen> critical_level_loser:
            rwd = 1
        else:
            rwd = 0

        r_t = torch.tensor(rwd , dtype=torch.float)

        return r_t

    def get_reward6(self):
        r1=self.before_op_remaining_pt_loser.mean()*self.critical_level_chosen-0.2*self.before_op_winq_chosen
        r2=self.pt_chosen*self.critical_level_loser.mean()-0.2*self.before_op_winq_loser.mean()

        rwd=(r1-r2)/20
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t

    def get_reward7(self):
        slack = self.before_op_slack
        critical_level = 1 - slack / (np.absolute(slack) + 64)
        critical_level_chosen = critical_level[self.position]
        critical_level_loser = np.delete(critical_level, self.position)
        earned_slack_chosen = np.mean(self.current_pt[:self.waiting_jobs-1])
        earned_slack_chosen *= critical_level_chosen
        consumed_slack_loser = self.pt_chosen*critical_level_loser.mean()
        rwd_slack = earned_slack_chosen - consumed_slack_loser
        rwd_winq = (self.before_op_winq_loser.mean() - self.before_op_winq_chosen) * 0.2
        rwd = ((rwd_slack + rwd_winq)/20).clip(-1,1)
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t



    def add_global_reward_RA(self): 
        job_record = self.job_creator.production_record[self.job_idx]
        path = job_record[1]
        queued_time = np.array(job_record[2])
        if self.tardiness and queued_time.sum():
            global_reward = - np.clip(self.tardiness / 64,0,1)
            reward = torch.ones(len(queued_time),dtype=torch.float)*global_reward
        else:
            reward = torch.ones(len(queued_time),dtype=torch.float)*0
        for i,m_idx in enumerate(path):
            r_t = reward[i]
            wc_idx = self.m_list[m_idx].wc_idx
            try:
                self.wc_list[wc_idx].incomplete_experience[self.job_idx].insert(2,r_t)
                self.wc_list[wc_idx].rep_memo.append(self.wc_list[wc_idx].incomplete_experience.pop(self.job_idx))
            except:
                pass
