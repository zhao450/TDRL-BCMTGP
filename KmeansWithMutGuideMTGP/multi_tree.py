import copy
import random
import numpy as np

from deap import gp, creator
from deap import tools
from collections import defaultdict


def roulette_wheel_selection(probabilities):

    if sum(probabilities) == 0:
        return np.random.randint(0, len(probabilities))
    r = np.random.random()
    c = 0
    for i, p in enumerate(probabilities):
        c += p
        if r <= c:
            return i
    return len(probabilities) - 1


def init_primitives(pset):
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addTerminal(str('NIQ'))  
    pset.addTerminal(str('WIQ'))  
    pset.addTerminal(str('MWT')) 
    pset.addTerminal(str('PT'))  
    pset.addTerminal(str('NPT')) 
    pset.addTerminal(str('OWT')) 
    pset.addTerminal(str('WKR')) 
    pset.addTerminal(str('NOR')) 
    pset.addTerminal(str('TIS')) 
    pset.addTerminal(str('SLACK')) 


def lf(x): 
    return 1 / (1 + np.exp(-x))


def init_toolbox(toolbox, pset):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6) 
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate",lim_xmate)
    toolbox.register("mutate",lim_xmut,expr=toolbox.expr_mut)



def maxheight(v):
    return max(i.height for i in v)


def wrap(func, *args, **kwargs):
    if func.__name__ == "xmate":
        keep_inds = [copy.deepcopy(ind) for ind in args[:2]]
    if func.__name__ == "xmut":
        keep_inds = [copy.deepcopy(ind) for ind in args[:1]]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > MAX_HEIGHT:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def searchSubtree_index(ind, index):
    temp=0
    for idx,node in enumerate(ind):
        if isinstance(node,gp.Primitive):
            if temp==index:
                ind_subtree=ind.searchSubtree(idx)
                break
            else:
                temp+=1
    return ind_subtree


__type__ = object
def cxOnePoint(ind1, ind2,ind1_cross_index,ind2_cross_index):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        types1[__type__] = list(range(1, len(ind1)))
        types2[__type__] = list(range(1, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        
        ind1_imp_subtree=searchSubtree_index(ind1,ind1_cross_index[0])
        ind1_unimp_subtree=searchSubtree_index(ind1,ind1_cross_index[1])
        ind2_imp_subtree=searchSubtree_index(ind2,ind2_cross_index[0])
        ind2_unimp_subtree=searchSubtree_index(ind2,ind2_cross_index[1])
        ind1[ind1_unimp_subtree],ind2[ind2_unimp_subtree]=ind2[ind2_imp_subtree],ind1[ind1_imp_subtree]

    return ind1, ind2


def mutUniform(individual, expr, probability,pset):
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret


    temp_tree=expr(pset=pset, type_=type_)

    for idx,node in enumerate(temp_tree):
        if isinstance(node,gp.Terminal):
            pro=roulette_wheel_selection(probability)
            if pro==0:
                temp_tree[idx].value='NIQ'
                temp_tree[idx].name='NIQ'
            if pro==1:
                temp_tree[idx].value='WIQ'
                temp_tree[idx].name='WIQ'
            if pro==2:
                temp_tree[idx].value='MWT'
                temp_tree[idx].name='MWT'
            if pro==3:
                temp_tree[idx].value='PT'
                temp_tree[idx].name='PT'
            if pro==4:
                temp_tree[idx].value='NPT'
                temp_tree[idx].name='NPT'
            if pro==5:
                temp_tree[idx].value='OWT'
                temp_tree[idx].name='OWT'
            if pro==6:
                temp_tree[idx].value='WKR'
                temp_tree[idx].name='WKR'
            if pro==7:
                temp_tree[idx].value='NOR'
                temp_tree[idx].name='NOR'
            if pro==8:
                temp_tree[idx].value='TIS'
                temp_tree[idx].name='TIS'
            if pro==9:
                temp_tree[idx].value='SLACK'
                temp_tree[idx].name='SLACK'


    
    individual[slice_] = temp_tree
    return individual,


def xmate(ind1, ind2,corss_one,corss_two):
    if len(ind1)==2:
        i1 = random.randrange(len(ind1))
        ind1_cross_index=[]
        ind2_cross_index=[]
        if i1==0:
            ind1_cross_index.append(corss_one['imp_seq_idx'])
            ind1_cross_index.append(corss_one['unimp_seq_idx'])
            ind2_cross_index.append(corss_two['imp_seq_idx'])
            ind2_cross_index.append(corss_two['unimp_seq_idx'])
        else:
            ind1_cross_index.append(corss_one['imp_rou_idx'])
            ind1_cross_index.append(corss_one['unimp_rou_idx'])
            ind2_cross_index.append(corss_two['imp_rou_idx'])
            ind2_cross_index.append(corss_two['unimp_rou_idx'])

        ind1[i1], ind2[i1] = cxOnePoint(ind1[i1], ind2[i1],ind1_cross_index,ind2_cross_index)

        i2 = 1 - i1 
        ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    else:
        if len(ind1) == 2:
            ind1[0], ind2[0] =cxOnePoint(ind1[0], ind2[0])
    return ind1, ind2

def lim_xmate(ind1, ind2,corss_one,corss_two):
    return wrap(xmate, ind1, ind2,corss_one,corss_two)



def xmut(ind,expr):
    i1 = random.randrange(len(ind))
    indx=gp.mutUniform(ind[i1], expr,pset=ind.pset)
    ind[i1] = indx[0]
    return ind,

def lim_xmut(ind, expr):
    res = wrap(xmut, ind,expr=expr)
    return res


def add_abs(a, b):
    return np.abs(np.add(a, b))


def sub_abs(a, b):
    return np.abs(np.subtract(a, b))


def mt_if(a, b, c):
    return np.where(a < 0, b, c)


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

MAX_HEIGHT = 8
N_TREES = 2 