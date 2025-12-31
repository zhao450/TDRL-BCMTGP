import numpy as np
import math
from deap import gp

def random_sequencing(data):
    job_position = np.random.randint(len(data[0]))
    return job_position

def SPT(data):
    job_position = np.argmin(data[0])
    return job_position

def LPT(data):
    job_position = np.argmax(data[0])
    return job_position

def LRO(data):
    job_position = np.argmax(data[10])
    return job_position

def LWKR(data): 
    job_position = np.argmin(data[0] + data[1])
    return job_position

def LWKRSPT(data):
    job_position = np.argmin(data[0]*2 + data[1])
    return job_position

def LWKRMOD(data):
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = np.argmin(data[0] + data[1] + MOD)
    return job_position

def EDD(data):
    job_position = np.argmin(data[2])
    return job_position

def COVERT(data): 
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    priority = (1 - cost / (0.05*average_pt)).clip(0,None) / data[0]
    job_position = priority.argmax()
    return job_position

def CR(data):
    time_till_due = data[5]
    CR = time_till_due / (data[0] + data[1])
    job_position = CR.argmin()
    return job_position

def CRSPT(data):
    CRSPT = data[5] / (data[0] + data[1]) + data[0]
    job_position = CRSPT.argmin()
    return job_position

def MS(data):
    slack = data[6]
    job_position = slack.argmin()
    return job_position

def MDD(data): 
    due = data[2]
    finish = data[1] + data[3]
    MDD = np.max([due,finish],axis=0)
    job_position = MDD.argmin()
    return job_position

def MON(data):
    due_over_pt = np.array(data[2])/np.sum(data[0])
    priority = due_over_pt/np.array(data[0])
    job_position = priority.argmax()
    return job_position

def MOD(data):
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = MOD.argmin()
    return job_position

def NPT(data): 
    job_position = np.argmin(data[9])
    return job_position

def ATC(data):
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    priority = np.exp( - cost / (0.05*average_pt)) / data[0]
    job_position = priority.argmax()
    return job_position

def AVPRO(data):
    AVPRO = (data[0] + data[1]) / (data[10] + 1)
    job_position = AVPRO.argmin()
    return job_position

def SRMWK(data): 
    SRMWK = data[6] / (data[0] + data[1])
    job_position = SRMWK.argmin()
    return job_position

def SRMWKSPT(data): 
    SRMWKSPT = data[6] / (data[0] + data[1]) + data[0]
    job_position = SRMWKSPT.argmin()
    return job_position

def WINQ(data): 
    job_position = data[7].argmin()
    return job_position

def PTWINQ(data):
    sum = data[0] + data[7]
    job_position = sum.argmin()
    return job_position

def PTWINQS(data): 
    sum = data[0] + data[6] + data[7]
    job_position = sum.argmin()
    return job_position

def DPTWINQNPT(data): 
    sum = data[0]*2 + data[7] + data[9]
    job_position = sum.argmin()
    return job_position

def DPTLWKR(data): 
    sum = data[0]*2 + data[1]
    job_position = sum.argmin()
    return job_position

def DPTLWKRS(data): 
    sum = data[0]*2 + data[1] + data[6]
    job_position = sum.argmin()
    return job_position

def FIFO(dummy): 
    job_position = 0
    return job_position

def GP_S1(data): 
    sec1 = data[0] + data[1]
    sec2 = (data[7]*2-1) / data[0]
    sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position

def GP_S2(data): 
    NIQ = len(data[0])
    sec1 = NIQ * (data[0]-1)
    sec2 = data[0] + data[1] * np.max([data[0],data[7]],axis=0)
    sec3 = np.max([data[7],NIQ+data[7]],axis=0)
    sec4 = (data[8]+1+np.max([data[1],np.ones_like(data[1])*(NIQ-1)],axis=0)) * np.max([data[7],data[1]],axis=0)
    sum = sec1 * sec2 + sec3 * sec4
    job_position = sum.argmin()
    return job_position

def GP_S3(data): 
    sec1 = data[0] + data[1]
    sec2 = (data[7]*2-1) / data[0]
    sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position



def GP_pair_S_test(data, tree_S):
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3, len(data)):
        new_data.append(data[i])
    individualvalue,length = treeNode_S_test(tree_S, 0, new_data) 
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0  
    job_position = individualvalue.argmin()
    return job_position


def treeNode_S_test(tree, index, data):
    if tree[index] == 'add':
        left,length_left = treeNode_S_test(tree, index+1, data)
        right,length_right = treeNode_S_test(tree, index+length_left+1, data)
        return left+right,length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index+length_left+1, data)
        return left - right, length_left + length_right + 1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index+length_left+1, data)
        return left * right, length_left + length_right + 1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return protected_div(left,right), length_left + length_right + 1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return np.maximum(left,right), length_left + length_right + 1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return np.minimum(left,right), length_left + length_right + 1
    elif tree[index] == 'lf': 
        ref,length_ref = treeNode_S_test(tree, index+1, data)
        if isinstance(ref, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-ref)),length_ref+1
        else:
            for i in range(len(ref)):
                ref[i] = 1 / (1 + np.exp(-ref[i]))
            return ref,length_ref+1
    elif tree[index] == 'NIQ':
        return data[0],1
    elif tree[index] == 'WIQ':
        return data[1],1
    elif tree[index] == 'MWT':
        return data[2],1
    elif tree[index] == 'PT':
        return data[3],1
    elif tree[index] == 'NPT':
        return data[4],1
    elif tree[index] == 'OWT':
        return data[5],1
    elif tree[index] == 'WKR':
        return data[6],1
    elif tree[index] == 'NOR':
        return data[7],1
    elif tree[index] == 'TIS':
        return data[8],1
    elif tree[index] == 'SLACK':
        return data[9],1

def GP_pair_S_ranks(data, tree_S):
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3, len(data)):
        new_data.append(data[i])
    individualvalue,length = treeNode_S_test(tree_S, 0, new_data) 
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return [0]  
    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        job_position = individualvalue.argmin()
        if job_position > len(ranks) - 1:
            print("Error!")
        ranks[job_position] = i
        individualvalue[job_position] = 10000000
    return ranks 

def GP_evolve_S_ranks(data, tree_S): 
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3,len(data)):
        new_data.append(data[i])
    individualvalue,length = treeNode_S(tree_S, 0, new_data)
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return [0] 
    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        job_position = individualvalue.argmin()
        ranks[job_position] = i
        individualvalue[job_position] = 10000000
    return ranks 

def GP_evolve_S(data, tree_S): 
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3, len(data)):
        new_data.append(data[i])
    individualvalue,length = treeNode_S(tree_S, 0, new_data) 
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0 
    job_position = individualvalue.argmin()
    return job_position

def treeNode_S(tree, index, data):
    if tree[index].arity == 2:
        left,length_left = treeNode_S(tree, index+1, data)
        right,length_right = treeNode_S(tree, index+length_left+1, data)
        if tree[index].name == 'add':
            return left+right,length_left+length_right+1
        elif tree[index].name == 'subtract':
            return left-right,length_left+length_right+1
        elif tree[index].name == 'multiply':
            return left*right,length_left+length_right+1
        elif tree[index].name == 'protected_div':
            return protected_div(left,right),length_left+length_right+1
        elif tree[index].name == 'maximum':
            return np.maximum(left,right),length_left+length_right+1
        elif tree[index].name == 'minimum':
            return np.minimum(left,right),length_left+length_right+1
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': 
            ref,length_ref = treeNode_S(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref)),length_ref+1
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                return ref,length_ref+1
    elif tree[index].arity == 0:
        if tree[index].name == 'NIQ':
            return data[0],1
        elif tree[index].name == 'WIQ':
            return data[1],1
        elif tree[index].name == 'MWT':
            return data[2],1
        elif tree[index].name == 'PT':
            return data[3],1
        elif tree[index].name == 'NPT':
            return data[4],1
        elif tree[index].name == 'OWT':
            return data[5],1
        elif tree[index].name == 'WKR':
            return data[6],1
        elif tree[index].name == 'NOR':
            return data[7],1
        elif tree[index].name == 'TIS':
            return data[8],1
        elif tree[index].name == 'SLACK':
            return data[9],1
def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x