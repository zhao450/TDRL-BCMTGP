from deap import gp  

 #add by zhaoc1
def extract_and_save_subtrees(individual):

    subtree_indiv=[]
    for idx,node in enumerate(individual):
        if isinstance(node,gp.Primitive):
            subtree=individual[individual.searchSubtree(idx)]
         
            subtree_gp=gp.PrimitiveTree(subtree)
            subtree_indiv.append(subtree_gp)
    return subtree_indiv
