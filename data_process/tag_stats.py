import os
import time
import numpy as np
import networkx as nx



if __name__ == "__main__":
    print("Loading graph...")
    start = time.time()
    OriginalGraph = nx.read_gpickle("./result/qa_graph.pickle")
    WeightedGraph = nx.read_gpickle("./result/weight_graph.pickle")
    print(f"Graph loaded in {time.time() - start:.2f} s.")

    print(f"Current number of nodes: {OriginalGraph.number_of_nodes()}")
    print(f"Current number of edges: {OriginalGraph.number_of_edges()}")

    print("Loading communities...")
    start = time.time()
    comm = np.load("./result/community.npy")
    comm_dict = dict(zip(comm[:, 0], comm[:, 1]))
    comm_unique, comm_cnt = np.unique(comm[:, 1], return_counts=True)
    comm_cnt_dict = dict(zip(comm_unique, comm_cnt))
    print(f"Communities loaded in {time.time() - start:.2f} s.")

    # ============================================================
    print(f"Current number of communities: {comm_unique.shape[0]}")
    print(f"Largest size of communities: {comm_cnt.max()}")
    print(f"Average size of communities: {comm_cnt.mean()}")

    print(f"Number of communities equal or below size    1: {np.count_nonzero(comm_cnt <= 1):>5}, with in total {np.sum(comm_cnt[comm_cnt <= 1]):>6} nodes.")
    print(f"Number of communities equal or below size    3: {np.count_nonzero(comm_cnt <= 3):>5}, with in total {np.sum(comm_cnt[comm_cnt <= 3]):>6} nodes.")
    print(f"Number of communities equal or below size    5: {np.count_nonzero(comm_cnt <= 5):>5}, with in total {np.sum(comm_cnt[comm_cnt <= 5]):>6} nodes.")
    print(f"Number of communities equal or below size   10: {np.count_nonzero(comm_cnt <= 10):>5}, with in total {np.sum(comm_cnt[comm_cnt <= 10]):>6} nodes.")
    print(f"Number of communities equal or below size  100: {np.count_nonzero(comm_cnt <= 100):>5}, with in total {np.sum(comm_cnt[comm_cnt <= 100]):>6} nodes.")
    print(f"Number of communities equal or below size 1000: {np.count_nonzero(comm_cnt <= 1000):>5}, with in total {np.sum(comm_cnt[comm_cnt <= 1000]):>6} nodes.")

    print(f"Number of communities above size 100: {np.count_nonzero(comm_cnt > 100)}, with in total {np.sum(comm_cnt[comm_cnt > 100])} nodes.")


    # ============================================================
    print("Removing nodes in communities with size lower than 100...")
    start = time.time()
    node_list = np.random.choice(list(OriginalGraph.nodes), 10, replace=False)
    for node in node_list:
        print(f"Community for node {node} is {comm_dict[node]}")
    removed_node_list = [node for node in OriginalGraph.nodes if comm_cnt_dict[comm_dict[node]] <= 100]
    OriginalGraph.remove_nodes_from(removed_node_list)
    print(f"Nodes removed in {time.time() - start:.2f} s.")

    print(f"Current number of nodes after removal: {OriginalGraph.number_of_nodes()}")
    print(f"Current number of edges after removal: {OriginalGraph.number_of_edges()}")


    # ============================================================
    print(f"Overall tag statistics:")
    overall_tag_cnt = {}
    for (u, v, tags) in OriginalGraph.edges.data('tags'):
        for tag in tags:
            overall_tag_cnt[tag] = overall_tag_cnt.get(tag, 0) + 1
    top_5_cnt = sorted(list(overall_tag_cnt.items()), key=lambda x: x[1], reverse=True)[:5]
    print(f"In total {len(list(overall_tag_cnt.keys()))} tags.")
    print(f"In total {np.sum(list(overall_tag_cnt.values()))} tag count.")
    print("Top 5 tags:")
    print(top_5_cnt)

    comm_list = comm_unique[comm_cnt > 100]
    print(f"Tag statistics for each community:")
    if os.path.exists("./result/subgraph_tag_scores.npy"):
        subgraph_tag_scores = np.load("./result/subgraph_tag_scores.npy", allow_pickle=True).tolist()
    else:
        subgraph_tag_scores = {}
        for comm in comm_list:
            subgraph = OriginalGraph.copy()
            removed_node_list = [node for node in subgraph.nodes if comm_dict[node] != comm]
            subgraph.remove_nodes_from(removed_node_list)

            tag_cnt = {}
            for (u, v, tags) in subgraph.edges.data('tags'):
                for tag in tags:
                    tag_cnt[tag] = tag_cnt.get(tag, 0) + 1
            top_5_cnt = sorted(list(tag_cnt.items()), key=lambda x: x[1], reverse=True)[:5]
            edge_cnt = subgraph.number_of_edges()
            print(f"\n\tIn total {len(list(tag_cnt.keys()))} tags for community {comm}.")
            print(f"\n\tIn total {edge_cnt} edges for community {comm}.")
            print(f"\tIn total {np.sum(list(tag_cnt.values()))} tag count for community {comm}.")
            print(f"\tTop 5 tags for community {comm}:")
            print(f"\t{top_5_cnt}")

            # Tag density: Community tag count / Community edge count
            # Tag coverage: Community tag count / Overall tag count
            # Tag factor: 1 + log10(Community tag count)
            # Tag score: Tag density * Tag coverage * Tag factor
            tag_score = {
                tag: (n_tag / edge_cnt) * (n_tag / overall_tag_cnt[tag]) * (np.log10(n_tag) + 1) 
                for tag, n_tag in tag_cnt.items()
            }
            subgraph_tag_scores[comm] = tag_score
        np.save("./result/subgraph_tag_scores.npy", subgraph_tag_scores)

    print("Calculating tag2comm:")
    tag2comm = {}
    for tag in overall_tag_cnt.keys():
        tag_score_list = sorted(
            [(comm, tag_score.get(tag, -1)) for comm, tag_score in subgraph_tag_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        if overall_tag_cnt[tag] > 10000:
            print("\t", tag, tag_score_list[:5])
        tag2comm[tag] = tag_score_list[0][0]
    np.save("./result/tag2comm.npy", tag2comm)

    print("Stating tag2comm:")
    comms = list(tag2comm.values())
    comms_unique, comms_cnt = np.unique(comms, return_counts=True)
    print(comms_unique, comms_cnt)
    print(comms_unique.shape[0])
