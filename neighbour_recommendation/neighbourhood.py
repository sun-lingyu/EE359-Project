import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import matplotlib.pyplot as plt
from .utils import _load_comm, get_rank
import time
import pickle as pkl

get_cluster = _load_comm("./data/community.npy")

def ppr_neighbour(G, idx, threshold):
    '''
    Calculate neighbours based on PPR algorithm.
    G: Input Graph
    idx: index of the interested node
    Return: list of idx's neighbourhood
    '''

    # Get Personalized Page Rank result
    res = pagerank(G, personalization={idx: 1}, max_iter=30)
    res = sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    res = dict(res)

    #visualize pagerank_score
    '''pagerank_score = res.values()
    plt.plot(pagerank_score)
    plt.show()'''

    # Calculate conductance
    A=set()
    Cut = 0
    Vol = 0
    threshold = 10  # Need to be modified!(Too sensitive!)
    curr = 0
    #conductance_list = []
    conductance = pre_conductance = float("inf")
    for node in res:
        A.add(node)
        degree = G.degree(node, weight="weight")
        Vol += degree
        Cut += degree
        for neighbour in G.adj[node]:
            if (neighbour in A):
                Cut -= 2 * G[node][neighbour]['weight']
        conductance = Cut / Vol
        if (pre_conductance < conductance):
            curr += 1
            if (curr == threshold):
                break
        #conductance_list.append(conductance)
        pre_conductance = conductance
    '''plt.plot(conductance_list)
    plt.show()
    print(len(A))
    exit()'''

    # Get nodes rank top #threshold
    '''count = 0
    last = res[0][1]
    stop = -1
    for i,(k,v) in enumerate(res):
        if (v > last):
            count += 1
        if (count == threshold):
            stop = i
            break
        last = v
    return dict(res[1:stop])  # except himself'''
    return list(A)

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
    rank = get_rank(G, idx, method, neighbours)

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
    #print(target_nodes)
    # If having enough nodes in target cluster
    if (len(target_nodes) >= num):
        return [item for item in sorted_target_rank[:num]]

    # Otherwise
    result = [item for item in sorted_target_rank]
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
    #print(distance_list)
    for i in range(distance_limit):
        target_rank = {key: rank.get(key) for key in distance_list[i]}
        sorted_target_rank = sorted(target_rank.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        result += sorted_target_rank[:num_left]
        if (len(result) == num):
            break
        num_left = num - len(result)

    return result




if __name__ == '__main__':
    print("Loading graph...")
    start = time.time()
    G = nx.read_gpickle("../data/weight_graph_di.pickle")
    print(f"Graph loaded in {time.time() - start:.2f} s.")

    idx = 1000
    clusterid = get_cluster(G,idx)
    print("clusterid: ",clusterid)

    # Convert a MultiDiGraph to DiGraph.
    # This sentence do not sum the weights of multiple edges.
    # TODO: Sum the weights.
    '''print("Converting to Digraph...")
    start = time.time()
    G = nx.DiGraph(G)
    print(f"Converted in {time.time() - start:.2f} s.")
    nx.write_gpickle(G, "../data/weight_graph_di.pickle")
    exit()'''

    '''print("Finding neighbours...")
    start = time.time()
    neighbours = ppr_neighbour(G, idx=idx, threshold=3)  # threshold is an important parameter to be tuned
    print(f"Neighbours found in {time.time() - start:.2f} s.")
    print(f"{len(neighbours)} neighbours found.")
    with open("tmp.pkl", 'wb') as f:
        pkl.dump(neighbours,f)
    exit()'''
    with open("tmp.pkl", 'rb') as f:
        neighbours = pkl.load(f)

    while (True):
        clusterid = int(input("Please Input Cluster id:"))
        print("Get recommendations...")
        start = time.time()
        recommend = get_recommended(G, num=3, neighbours=neighbours, idx=idx, clusterid=clusterid, method="indegree")
        print(recommend)
        print(f"Recommendations got in {time.time() - start:.2f} s.")
