from numpy.random import choice


def biased_random_walk(G, idx, threshold=None):
    '''
    Calculate neighbours based on biased random walk.
    G: Input Graph
    idx: index of the interested node
    Return: list of idx's neighbourhood
    '''
    start_node = idx
    max_iter = 100
    remake_prob = 0.15
    p = 0.5 # 越小，越BFS，但有向图感觉难有退回的边，因此可以调大remake_prob来归位
    q = 2 # 越小，越DFS

    walk = [idx]
    curr_node = walk[0]
    curr_neighbor = sorted(G.neighbors(curr_node))
    if(len(curr_neighbor)==0):
        return None

    curr_node = choice(curr_neighbor)
    walk.append(curr_node)

    for step in range(max_iter):
        prev_node = walk[-2]
        curr_node = walk[-1]
        curr_neighbor = sorted(G.neighbors(curr_node))
        if(len(curr_neighbor)==0):
            walk.append(start_node) # 这里建议直接remake，从头再来
            continue

        curr_neighbor_weight = []

        for x in curr_neighbor:
            weight = G[curr_node][x]['weight']
            if x == prev_node:
                curr_neighbor_weight.append(weight/p) 
            elif G.has_edge(x,prev_node) or G.has_edge(prev_node,x):
                curr_neighbor_weight.append(weight)
            else:
                curr_neighbor_weight.append(weight/q)

        weight_sum = sum(curr_neighbor_weight)
        cur_neighbor_prob = [weight/weight_sum for weight in curr_neighbor_weight]
        next_node = choice(curr_neighbor,size=1,p=cur_neighbor_prob)[0]
        
        walk.append(next_node)
        if(p<remake_prob): # 这里建议直接remake，从头再来
            walk.append(start_node)

    return list(set(walk)-set([start_node]))

if __name__=="__main__":
    # usage example
    import networkx as nx
    WeightedGraph = nx.read_gpickle("./result/weight_graph.pickle")
    neighbor = biased_random_walk(WeightedGraph,idx=3)
    print(neighbor)
    print(len(neighbor))