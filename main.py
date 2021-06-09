import time
import networkx as nx
import numpy as np
from recommendation.neighbourhood import get_recommended
from recommendation.random_walk import biased_random_walk
from recommendation.calc_global_pr import calc_global_pr

Neighbouring_recommendations_for_each_tag = 5
Distance_limit = 3

# Load Graph
start = time.time()
G = nx.read_gpickle("./data/weight_graph_di.pickle")
print(f"Graph loaded in {time.time() - start:.2f} s.")

# Load Tag-to-Community Info
with open("./data/tag2comm.npy","rb") as f:
    comm_info = np.load(f, allow_pickle=True).tolist()
    
# Load Node-to-Community Info
comm = np.load("./data/community.npy")
comm_dict = dict(zip(comm[:, 0], comm[:, 1]))

# Loop for Query
while (True):

    #userid = int(input("Please enter user id: "))
    #tags = input("Please enter tags(space splitted): ").split()
    userid = 744
    tags = ["google-vr", "unity3d"]
    in_graph = False

    print(f"User id: {userid}")
    print(f"Question Tags:{tags}")

    # Global recommendation、
    print("Get global recommendations:")
    start = time.time()
    pr_list = calc_global_pr(tags)
    for user_id, user_comm, user_score, comm_target_tag_score in pr_list:
        print(f"\tUser ID: {user_id:>7.0f}, Score: {user_score:.6f}, in community {user_comm:>4.0f} with tag score {comm_target_tag_score:.6f}")
    print(f"Global recommendations got in {time.time() - start:.2f} s.")

    # Check whether in graph
    if (not G.has_node(userid)):
        # If not in Graph, do not give neibouring recommendations
        print("No Neighbouring recommendations...")
        input("Continue?")
        continue

    # Get neighbors using biased_random_walk.
    start = time.time()
    print("Get neighboring recommendations...")
    neighbours = biased_random_walk(G, userid)

    recommend_tags = []
    # Get recommendation for each tag
    for i, tag in enumerate(tags):
        commid = comm_info[tag]
        # Get neighbouring recommendations
        recommend_tags.append(get_recommended(G, num=Neighbouring_recommendations_for_each_tag, neighbours=neighbours, idx=userid, commid=commid, comm_dict=comm_dict, method="indegree", distance_limit=Distance_limit))

    # Combine neighbouring recommendations for all tags:
    recommend_neighbour = {}
    # recommend_neighbour: anordered dict, used as ordered set. (Since set in python is unordered.)
    # Each element of recommend_neighbour represents a person. It is an ordered set(implemented as an ordered dict) of tags.
    for i in range(Neighbouring_recommendations_for_each_tag):
        for j, tag in enumerate(tags):
            recommendation, score, distance = recommend_tags[j][i]
            if (recommendation not in recommend_neighbour):
                if(distance == Distance_limit):
                    recommend_neighbour[recommendation] = {} # too far away from target tag. So do not need to remember the tag.
                else:
                    recommend_neighbour[recommendation] = {tag: 1}  # remember the tag of the recommendation.
            else:
                if (distance == Distance_limit): # too far away from target tag. So do not need to remember the tag.
                    continue
                else:
                    recommend_neighbour[recommendation][tag] = 1  # remember the tag of the recommendation.

    for recommendation, tags in recommend_neighbour.items():
        if (tags == {}):
            print(f"\tUser ID: {recommendation}")
        else:
            print(f"\tUser ID: {recommendation}, Recommend based on {list(tags.keys())}")
    print(f"Neighboring recommendations got in {time.time() - start:.2f} s.")
    
    input("Continue?")
