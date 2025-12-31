import math

import simpy
import random
import numpy as np
import torch


def random_routing(idx, data, job_pt, job_slack, wc_idx, *args):
    machine_idx = np.random.randint(len(job_pt))
    return machine_idx

def TT(idx, data, job_pt, job_slack, wc_idx, *args): 
    rank = np.argmin(data, axis=0)
    machine_idx = rank[0]
    return machine_idx

def ET(idx, data, job_pt, job_slack, wc_idx, *args): 
    machine_idx = np.argmin(job_pt)
    return machine_idx

def EA(idx, data, job_pt, job_slack, wc_idx, *args): 
    rank = np.argmin(data, axis=0)
    machine_idx = rank[1]
    return machine_idx

def SQ(idx, data, job_pt, job_slack, wc_idx, *args): 
    rank = np.argmin(data, axis=0)
    machine_idx = rank[2]
    return machine_idx

def CT(idx, data, job_pt, job_slack, wc_idx, *args): 
    completion_time = np.array(data)[:,1].clip(0) + np.array(job_pt)
    machine_idx = completion_time.argmin()
    return machine_idx

def UT(idx, data, job_pt, job_slack, wc_idx, *args): 
    rank = np.argmin(data, axis=0)
    machine_idx = rank[3]
    return machine_idx

def GP_R1(idx, data, job_pt, job_slack, wc_idx, *args): 
    data = np.transpose(data)
    sec1 = min(2 * data[2] * np.max([data[2]*job_pt/data[1] , job_pt*data[0]*data[0]], axis=0))
    sec2 = data[2] * job_pt - data[1]
    sum = sec1 + sec2
    machine_idx = sum.argmin()
    return machine_idx

def GP_R2(idx, data, job_pt, job_slack, wc_idx, *args): 
    data = np.transpose(data) 
    sec1 = data[2]*data[2], (data[2]+job_pt)*data[2]
    sec2 = np.min([data[1],args[0]/(data[1]*args[0]-1)],axis=0)
    sec3 = -data[2] * args[0]
    sec4 = data[2] * job_pt * np.max([data[0], np.min([data[1],job_pt],axis=0)/(args[0])],axis=0)
    sec5 = np.max([data[2]*data[2], np.ones_like(data[2])*(args[1]-args[0]-1), (data[2]+job_pt)*np.min([data[2],np.ones_like(data[2])*args[1]],axis=0)],axis=0)
    sum = sec1 - sec2 * np.max([sec3+sec4/sec5],axis=0)
    machine_idx = sum.argmin()
    return machine_idx

def GP_pair_R_test(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): 
    data = np.transpose(data) 
    individualvalue,length = treeNode_R_test(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W,
                                 TIS, SLACK) 
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0  
    machine_idx = individualvalue.argmin()
    return machine_idx

def GP_pair_ensemble_R_test(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args):
    data = np.transpose(data) 
    individualvalue,length = treeNode_R_test(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W,
                                 TIS, SLACK) 
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        print("Error here: GP_pair_ensemble_R_test!")
        return 0 
    return individualvalue

def treeNode_R_test(tree, index, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK):
    if tree[index] == 'add':
        left,length_left = treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right, length_right = treeNode_R_test(tree, index+length_left+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        return left + right, length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_R_test(tree, index + 1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data, current_pt, next_pt, OWT, WKR, NOR,
                                              W, TIS, SLACK)
        return left - right, length_left+length_right+1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_R_test(tree, index + 1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data, current_pt, next_pt, OWT, WKR, NOR,
                                              W, TIS, SLACK)
        return left * right, length_left+length_right+1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_R_test(tree, index + 1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data, current_pt, next_pt, OWT, WKR, NOR,
                                              W, TIS, SLACK)
        return protected_div(left, right), length_left+length_right+1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_R_test(tree, index + 1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data, current_pt, next_pt, OWT, WKR, NOR,
                                              W, TIS, SLACK)
        return np.maximum(left, right), length_left+length_right+1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_R_test(tree, index + 1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data, current_pt, next_pt, OWT, WKR, NOR,
                                              W, TIS, SLACK)
        return np.minimum(left, right), length_left + length_right + 1
    elif tree[index] == 'lf':
        ref,length_ref = treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
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
        return current_pt,1
    elif tree[index] == 'NPT':
        return next_pt,1
    elif tree[index] == 'OWT':
        return OWT,1
    elif tree[index] == 'WKR':
        return WKR,1
    elif tree[index] == 'NOR':
        return NOR,1
    elif tree[index] == 'W':
        return W,1
    elif tree[index] == 'TIS':
        return TIS,1
    elif tree[index] == 'SLACK':
        return SLACK,1

def GP_pair_R_ranks(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args):
    data = np.transpose(data) 
    individualvalue,length = treeNode_R_test(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W,
                                 TIS, SLACK)  
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return [0]  
    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        machine_idx = individualvalue.argmin()
        ranks[machine_idx] = i
        individualvalue[machine_idx] = 10000000
    return ranks  
def GP_evolve_R_ranks(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): 
    data = np.transpose(data) 
    individualvalue,length = treeNode_R(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)  
    if isinstance(individualvalue, (np.int64, np.float64, float,  np.int32,int)):
        return [0] 

    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        machine_idx = individualvalue.argmin()
        ranks[machine_idx] = i
        individualvalue[machine_idx] = 10000000
    return ranks  
def GP_evolve_R_ranks_change(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): 
    data = np.transpose(data)  
    individualvalue,length = treeNode_R(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) 
    if isinstance(individualvalue, (np.int64, np.float64, float,  np.int32,int)):
        return [0,0,0]

    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        machine_idx = individualvalue.argmin()
        ranks[machine_idx] = i
        individualvalue[machine_idx] = 10000000
    return ranks  


def GP_evolve_R(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): 
    data = np.transpose(data)  
    individualvalue,length = treeNode_R(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)  
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0 
    machine_idx = individualvalue.argmin()
    return machine_idx


def treeNode_R(tree, index, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK):
    if tree[index].arity == 2: 
        left,length_left=treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        right,length_right=treeNode_R(tree, index+length_left+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        if tree[index].name == 'add':
            return left + right, length_left+length_right+1
        elif tree[index].name == 'subtract':
            return left - right, length_left+length_right+1
        elif tree[index].name == 'multiply':
            return left * right, length_left+length_right+1
        elif tree[index].name == 'protected_div':
            return protected_div(left, right), length_left+length_right+1
        elif tree[index].name == 'maximum':
            return np.maximum(left, right), length_left+length_right+1
        elif tree[index].name == 'minimum':
            return np.minimum(left, right), length_left+length_right+1
    elif tree[index].arity == 1:
        if tree[index].name == 'lf':
            ref,length_ref = treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
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
            return current_pt,1
        elif tree[index].name == 'NPT':
            return next_pt,1
        elif tree[index].name == 'OWT':
            return OWT,1
        elif tree[index].name == 'WKR':
            return WKR,1
        elif tree[index].name == 'NOR':
            return NOR,1
        elif tree[index].name == 'W':
            return W,1
        elif tree[index].name == 'TIS':
            return TIS,1
        elif tree[index].name == 'SLACK':
            return SLACK,1

def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x