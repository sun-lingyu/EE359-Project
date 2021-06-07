import time
import networkx as nx
import numpy as np
from neighbour_recommendation.utils import _load_comm
from neighbour_recommendation.neighbourhood import get_recommended
from neighbour_recommendation.random_walk import biased_random_walk
from neighbour_recommendation.calc_global_pr import calc_global_pr


start = time.time()
# G = nx.read_gpickle("./data/weight_graph_di.pickle")
G = nx.read_gpickle("./data/weight_graph_di_py37.pickle")
print(f"Graph loaded in {time.time() - start:.2f} s.")

with open("./data/tag2comm.npy","rb") as f:
    cluster_info = np.load(f,allow_pickle=True).tolist()

get_cluster = _load_comm("./data/community.npy")

while (True):
    #try:
    #userid = int(input("Please enter user id: "))
    #tags = input("Please enter tags(space splitted): ").split()
    userid = 999
    tags = ["java", "python"]
    in_graph = False

    # Check whether in graph
    if (G.has_node(userid)):
        # Get neighbours
        neighbours = biased_random_walk(G, userid)
        in_graph = True


    print("Get global recommendations:")
    start = time.time()
    pr_list = calc_global_pr(tags)
    for user_id, user_score, user_comm, comm_target_tag_score in pr_list:
        print(f"\tUser ID: {user_id}, Score: {user_score}, in community {user_comm} with tag score {comm_target_tag_score}")
    print(f"Global recommendations got in {time.time() - start:.2f} s.")


    # Get recommendation for each tag
    for tag in tags:
        clusterid = cluster_info[tag]

        print("Get neighboring recommendations...")
        start = time.time()
        """
        # Get Global recommendation
        comm_file_name ="./data/comm_pr/{}.npy".format(clusterid)
        pagerank = np.load(comm_file_name, allow_pickle=True).item()
        pagerank_list = sorted(pagerank.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10]
        recommend_global = [kv[0] for kv in pagerank_list]
        print("Global recommendations: ", recommend_global)
        """

        if in_graph:
        # Get neighbouring recommendations
            recommend_neighbour = get_recommended(G, num=10, neighbours=neighbours, idx=userid, clusterid=clusterid, method="indegree", distance_limit=3)
            print(f"Neighboring recommendations got in {time.time() - start:.2f} s.")
            print("Neighbouring recommendations: ", recommend_neighbour)
        
        
        input("Continue?")
    '''except Exception as e:
        print(e)'''