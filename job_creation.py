import numpy as np
from tabulate import tabulate

class creation:
    def __init__ (self, env, span, machine_list, workcenter_list, pt_range, due_tightness, E_utliz, **kwargs):
        self.ifPrint = False
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            self.initialSeed = kwargs['seed']
            if 'ifPrint' in kwargs: 
                self.ifPrint = kwargs['ifPrint']
                if kwargs['ifPrint']:
                    print("Random seed of job creation is fixed, seed: {}".format(kwargs['seed']))
            else:
                print("Random seed of job creation is fixed, seed: {}".format(kwargs['seed']))
        self.env = env
        self.span = span
        self.m_list = machine_list
        self.wc_list = workcenter_list
        self.no_wcs = len(self.wc_list)
        self.no_machines = len(self.m_list)
        self.m_per_wc = int(self.no_machines / self.no_wcs)
        self.production_record = {}
        self.tardiness_record = {}
        self.sqc_reward_record = []
        self.rt_reward_record = []
        self.pt_range = pt_range
        self.avg_pt = np.average(self.pt_range) - 0.5
        self.tightness = due_tightness
        self.E_utliz = E_utliz
        self.sequence_seed = np.arange(self.no_wcs)
        self.in_system_job_no = 0
        self.in_system_job_no_dict = {}
        self.index_jobs = 0
        self.comp_rate_list = [[] for m in self.m_list]
        self.comp_rate = 0
        self.realized_tard_list = [[] for m in self.m_list]
        self.realized_tard_rate = 0
        self.exp_tard_list = [[] for m in self.m_list]
        self.available_time_list = np.array([0 for m in self.m_list])
        self.release_time_list = np.array([self.avg_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.no_machines)
        self.next_wc_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.avg_pt for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.next_ttd_list = np.array([self.avg_pt*self.no_wcs for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        self.sequence_list = []
        self.pt_list = []
        self.remaining_pt_list = []
        self.create_time = []
        self.due_list = []
        self.flowtime_list = [] 
        self.arrival_dict = {}
        self.departure_dict = {}
        self.mean_dict = {}
        self.std_dict = {}
        self.expected_tardiness_dict = {}
        self.beta = self.avg_pt / (self.m_per_wc * self.E_utliz)
        self.total_no = np.round(self.span/self.beta).astype(int)
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        if 'realistic_var' in kwargs and kwargs['realistic_var']:
            self.ptl_generation = self.ptl_generation_realistic
            self.realistic_var = kwargs['realistic_var']
        else:
            self.ptl_generation = self.ptl_generation_random
        if 'random_seed' in kwargs and kwargs['random_seed']:
            interval = self.span/50
            self.env.process(self.dynamic_seed_change(interval))
        if 'hetero_len' in kwargs and kwargs['hetero_len']:
            pass
        if 'even' in kwargs and kwargs['even']:
            print("EVEN mode ON")
            self.arrival_interval = np.ones(self.arrival_interval.size)*self.arrival_interval.mean()
        self.initial_job_assignment()
        self.env.process(self.new_job_arrival())

    def initial_job_assignment(self):
        sqc_seed = np.arange(self.no_wcs) 
        for wc_idx,wc in enumerate(self.wc_list): 
            np.random.shuffle(sqc_seed)
            sqc = np.concatenate([np.array([wc_idx]),sqc_seed[sqc_seed!=wc_idx]])
            for m_idx,m in enumerate(wc.m_list): 
                self.sequence_list.append(sqc)
                ptl = self.ptl_generation()
                self.pt_list.append(ptl)
                self.record_job_feature(self.index_jobs,ptl)
                remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[sqc]
                self.remaining_pt_list.append(remaining_ptl)
                avg_pt = ptl.mean()
                due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness))
                self.create_time.append(0)
                self.flowtime_list.append(0)
                self.due_list.append(due)
                self.record_job_arrival()
                self.production_record[self.index_jobs] = [[],[],[],{},[]]
                wc.queue.append(self.index_jobs)
                wc.sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
                wc.pt_list.append(self.pt_list[self.index_jobs])
                wc.remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
                wc.due_list.append(self.due_list[self.index_jobs])
                wc.weight_list.append(1)
                wc.release_time_list.append(0)
                self.index_jobs += 1
            wc.routing_event.succeed()

    def new_job_arrival(self):
        while self.index_jobs < self.total_no:
            time_interval = self.arrival_interval[self.index_jobs]
            yield self.env.timeout(time_interval)
            np.random.shuffle(self.sequence_seed)
            self.sequence_list.append(np.copy(self.sequence_seed))
            ptl = self.ptl_generation()
            self.pt_list.append(ptl)
            self.record_job_feature(self.index_jobs,ptl)
            remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[self.sequence_seed]
            self.remaining_pt_list.append(remaining_ptl)
            avg_pt = ptl.mean()
            due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness) + self.env.now)
            self.create_time.append(self.env.now)
            self.flowtime_list.append(self.env.now) 
            self.due_list.append(due)
            first_workcenter = self.sequence_seed[0]
            self.record_job_arrival()
            self.production_record[self.index_jobs] = [[],[],[],{},[]]
            self.wc_list[first_workcenter].queue.append(self.index_jobs)
            self.wc_list[first_workcenter].sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
            self.wc_list[first_workcenter].pt_list.append(self.pt_list[self.index_jobs])
            self.wc_list[first_workcenter].remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
            self.wc_list[first_workcenter].due_list.append(self.due_list[self.index_jobs])
            self.wc_list[first_workcenter].weight_list.append(1)
            self.wc_list[first_workcenter].release_time_list.append(self.create_time[self.index_jobs])
            self.index_jobs += 1
            try:
                self.wc_list[first_workcenter].routing_event.succeed()
            except:
                pass

    def ptl_generation_random(self):
        ptl = np.random.randint(self.pt_range[0], self.pt_range[1], size = [self.no_machines])
        return ptl

    def ptl_generation_realistic(self):
        base = np.random.randint(self.pt_range[0], self.pt_range[1], [self.no_wcs,1]) * np.ones([self.no_wcs, self.m_per_wc])
        variation = np.random.randint(-self.realistic_var,self.realistic_var,[self.no_wcs, self.m_per_wc])
        ptl = (base + variation).clip(self.pt_range[0], self.pt_range[1])
        ptl = np.concatenate(ptl)
        return ptl

    def dynamic_seed_change(self, interval):
        while self.env.now < self.span:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
            if self.ifPrint:
                print('change random seed to {} at time {}'.format(seed,self.env.now))

    def change_setting(self,pt_range):
        print('Heterogenity changed at time',self.env.now)
        self.pt_range = pt_range
        self.avg_pt = np.average(self.pt_range)-0.5
        self.beta = self.avg_pt / (2*self.E_utliz)

    def get_global_exp_tard_rate(self):
        x = []
        for m in self.m_list:
            x = np.append(x, m.slack)
        rate = x[x<0].size / x.size
        return rate
    def record_job_arrival(self):
        self.in_system_job_no += 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.arrival_dict[self.env.now] += 1
        except:
            self.arrival_dict[self.env.now] = 1

    def record_job_departure(self):
        self.in_system_job_no -= 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.departure_dict[self.env.now] += 1
        except:
            self.departure_dict[self.env.now] = 1

    def record_job_feature(self,idx,ptl):
        self.mean_dict[idx] = (self.env.now, ptl.mean())
        self.std_dict[idx] = (self.env.now, ptl.std())
    def get_expected_tardiness(self, ptl, due):
        sum_remaining_pt = sum([m.remaining_job_pt.sum() for m in self.m_list])
        expected_waiting_time = sum_remaining_pt / self.no_machines
        expected_processing_time = ptl.mean() * self.no_wcs
        expected_tardiness = expected_processing_time + expected_waiting_time + self.env.now - due
        self.expected_tardiness_dict[self.index_jobs] = max(0, expected_tardiness)

    def build_sqc_experience_repository(self,m_list):
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        for m in m_list:
            self.incomplete_rep_memo[m.m_idx] = {}
            self.rep_memo[m.m_idx] = []

    def output(self):
        print('job information are as follows:')
        job_info = [[i,self.sequence_list[i], self.pt_list[i], \
        self.create_time[i], self.due_list[i]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due']))
        print('--------------------------------------')
        return job_info

    def final_output(self):
        output_info = []
        for item in self.production_record:
            output_info.append(self.production_record[item][5])
        job_info = [[i,self.sequence_list[i], self.pt_list[i], self.create_time[i],\
        self.due_list[i], output_info[i][0], output_info[i][1]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due','out','tard.']))
        realized = np.array(output_info)[:,1].sum()
        exp_tard = sum(self.expected_tardiness_dict.values())

    def tardiness_output(self):
        tard_info = []
        for item in self.production_record:
            tard_info.append(self.production_record[item][5])
        dt = np.dtype([('output', float), ('tardiness', float)])
        tard_info = np.array(tard_info, dtype=dt)
        tard_info = np.array(tard_info.tolist())
        tard_info = np.array(tard_info)
        output_time = tard_info[:, 0]
        tard = tard_info[:, 1]
        cumulative_tard = np.cumsum(tard)
        tard_max = np.max(tard)
        tard_min = np.min(tard)
        tard_mean = np.cumsum(tard) / np.arange(1, len(cumulative_tard) + 1)
        tard_rate = tard.clip(0, 1).sum() / tard.size
        output_time = self.flowtime_list 
        return output_time, cumulative_tard, tard_mean, tard_max, tard_rate

    def record_printout(self):
        print(self.production_record)

    def timing_output(self):
        return self.arrival_dict, self.departure_dict, self.in_system_job_no_dict

    def feature_output(self):
        return self.mean_dict, self.std_dict

    def all_tardiness(self):
        tard = []
        for item in self.production_record:
            tard.append(self.production_record[item][5][1])
        tard = np.array(tard)
        mean_tardiness = tard.mean()
        tardy_rate = tard.clip(0,1).sum() / tard.size
        return mean_tardiness, tardy_rate
