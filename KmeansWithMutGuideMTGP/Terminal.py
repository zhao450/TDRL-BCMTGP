'''
Implemented by Mengxu 2022.10.13
to create terminals used for MTGP
'''

class Terminal:
    def __init__(self, name, **kwargs):
        self.name=name


    def getValue(self, name):
        if name == 'CPT':
            return self.cumulative_pt()
        elif name == 'AT':
            return self.available_time()
    def cumulative_pt(self):
        return 'cumulative_pt'

    def available_time(self):
        return 'available_time'

    def que_size(self):
        return 'que_size'

    def cumulative_run_time(self):
        return 'cumulative_run_time'

    def current_pt(self, agent_machine):
        return agent_machine.current_pt

    def remaining_job_pt(self, agent_machine):
        return agent_machine.remaining_job_pt

    def due_list(self, agent_machine):
        return agent_machine.due_list

    def completion_rate(self, agent_machine):
        return agent_machine.completion_rate

    def time_till_due(self, agent_machine):
        return agent_machine.time_till_due

    def slack(self, agent_machine):
        return agent_machine.slack

    def winq(self, agent_machine):
        return agent_machine.winq

    def avlm(self, agent_machine):
        return agent_machine.avlm

    def next_pt(self, agent_machine):
        return agent_machine.next_pt

    def remaining_no_op(self, agent_machine):
        return agent_machine.remaining_no_op

    def waited_time(self, agent_machine):
        return agent_machine.waited_time

    def wc_idx(self, agent_machine):
        return agent_machine.wc_idx

    def queue(self, agent_machine):
        return agent_machine.queue

    def m_idx(self, agent_machine):
        return agent_machine.m_idx

    def env_time(self, agent_machine):
        return agent_machine.env.now