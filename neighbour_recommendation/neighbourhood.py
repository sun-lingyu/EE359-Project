import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import matplotlib.pyplot as plt
from utils import get_cluster,get_rank


def ppr_neighbour(G, idx, threshold):
    '''
    Calculate neighbours based on PPR algorithm.
    G: Input Graph
    idx: index of the interested node
    Return: dict of idx's neighbourhood
    '''
    # Convert a MultiDiGraph to DiGraph.
    # This sentence do not sum the weights of multiple edges.
    # TODO: Sum the weights.
    G = nx.DiGraph(G)

    # Get Personalized Page Rank result
    res = pagerank(G, personalization={idx: 1})
    res = sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    # Get nodes rank top #threshold
    count = 0
    last = res[0][1]
    stop = -1
    for i,(k,v) in enumerate(res):
        if (v > last):
            count += 1
        if (count == threshold):
            stop = i
            break
        last = v
    return res[1:stop]  # except himself

def get_recommended(G, num, neighbours, idx, clusterid, method="indegree", distance_limit=4):
    '''
    Get recommendations for a given question.
    G: Input Graph
    num: number of recommendations
    neighbours: list of all neighbours
    idx: idex of the questioner
    clusterid: the cluster of the interested tag
    method: Currently only support "indegree".
    distance_limit: limit distance for BFS search 
    Return list of recommendations(sorted)
    '''
    # Get the questioner's clusterid
    mycluster = get_cluster(G, idx)  # may need to be modified
    
    # Compute rank of all neighbours
    rank = get_rank(G, idx, method)

    # Case: Target cluster is questioner's cluster
    # Recommend only based on rank
    if (clusterid == mycluster):
        sorted_rank = sorted(rank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        return [item[0] for item in sorted_rank[:num]]
    
    # Case: Target cluster is **not** questioner's cluster
    # Recommend based on rank and shortest path

    # First check: is there any neighbours lying in target cluster
    target_nodes = []
    other_nodes = []
    for neighbour in neighbours:
        if (get_cluster(G, neighbour) == clusterid):
            target_nodes.append(neighbour)
        else:
            other_nodes.append(neighbour)

    # Get sorted rank of all target_nodes
    target_rank = {key: rank.get(key) for key in target_nodes}
    sorted_target_rank = sorted(target_rank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    
    # If having enough nodes in target cluster
    if (len(target_nodes) >= num):
        return [item[0] for item in sorted_target_rank[:num]]

    # Otherwise
    result = [item[0] for item in sorted_target_rank]
    num_left = num - len(result)

    def search_for_cluster(children,clusterid):
        for child in children:
            if (get_cluster(G, child) == clusterid):
                return True
        return False

    distance_list = [[] for _ in range(distance_limit)]

    # Do BFS in other_nodes
    for node in other_nodes:
        #shortest_path = nx.single_source_shortest_path_length(G, node)
        for distance in range(1,distance_limit):
            children = nx.descendants_at_distance(G, node, distance)
            if search_for_cluster(children, clusterid):
                distance_list[distance-1].append(node)
                break
        else:
            distance_list[-1].append(node)
    
    for i in range(distance_limit):
        target_rank = {key: rank.get(key) for key in distance_limit[i]}
        sorted_target_rank = sorted(target_rank.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        result += sorted_target_rank[:num_left]
        if (len(result) == num):
            break
        num_left = num - len(result)

    return result




if __name__ == '__main__':
    G = nx.MultiDiGraph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 2, dict(weight=1)), (1, 2, dict(weight=2))\
                    , (1, 3, dict(weight=3)), (2, 3, dict(weight=1)) \
                    , (3, 4, dict(weight=2))])
    '''print(G[1])
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()'''
    print(G.edges.data('weight'))

    neighbours = ppr_neighbour(G, idx=1, threshold=3)  # threshold is an important parameter to be tuned
    get_recommended(G, num=2, neighbours=neighbours.keys(), clusterid=1, method="indegree")