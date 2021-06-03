import os
import time
import numpy as np
import networkx as nx
import community as community_louvain


def build_graph(qa_file_path, max_edge_num=None):
    print("Loading QA files...")
    start = time.time()
    pair_list = []
    weighted_pair_list = []
    
    with open(qa_file_path, "r") as qa:
        edge_cnt = 0
        for line in qa.readlines():
            if max_edge_num is not None and edge_cnt >= max_edge_num:
                break
            qid, aid, qowner, aowner, tags, qscore, qview, qfavor, ascore = line.strip().split("\t")
            qowner, aowner, qscore, qview, qfavor, ascore = \
                list(map(int, [qowner, aowner, qscore, qview, qfavor, ascore]))
            tags = tags.split(";")

            pair_list.append(
                (qowner, aowner, {"tags": tags, "qscore": qscore, "view": qview, "favor": qfavor, "ascore": ascore})
            )

            weight = np.log(qscore + 150) + np.log(qview / 1000 + 10) + np.log(qfavor + 5) + np.log(ascore + 180)
            weighted_pair_list.append(
                (qowner, aowner, weight)
            )

            edge_cnt += 1

    # pair: (source, target, {"attr_key": attr_value})
    print(f"QA file loaded in {time.time() - start:.2f} s.")

    print("Building original graph...")
    start = time.time()
    oG = nx.MultiDiGraph()
    oG.add_edges_from(pair_list)
    print(f"Original graph built in {time.time() - start:.2f} s.")

    print("Building weighted graph...")
    start = time.time()
    wG = nx.Graph()
    for qowner, aowner, weight in weighted_pair_list:
        if wG.has_edge(qowner, aowner):
            wG.edges[qowner, aowner]["weight"] += weight
        else:
            wG.add_edge(qowner, aowner, weight=weight)
    print(f"Weighted graph built in {time.time() - start:.2f} s.")

    return oG, wG


if __name__ == "__main__":
    if os.path.exists("../data/qa_graph.pickle"):
        print("Loading graph...")
        start = time.time()
        OG = nx.read_gpickle("../data/qa_graph.pickle")
        WG = nx.read_gpickle("../data/weight_graph.pickle")
        print(f"Graph loaded in {time.time() - start:.2f} s.")
    else:
        OG, WG = build_graph("../data/qa.csv")

        print("Saving graph...")
        start = time.time()
        nx.write_gpickle(OG, "../data/qa_graph.pickle")
        nx.write_gpickle(WG, "../data/weight_graph.pickle")
        print(f"Graph saved in {time.time() - start:.2f} s.")

    print(f"Current number of nodes: {OG.number_of_nodes()}")
    print(f"Current number of edges: {OG.number_of_edges()}")

    print("Performing Louvain algorithm on graph...")
    partition = community_louvain.best_partition(WG)
    partition = np.array(list(partition.items()))
    np.save("../data/community.npy", partition)
    print(f"Louvain completed in {time.time() - start:.2f} s.")

    """
    print("Visualizing graph...")
    start = time.time()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(60, 40), dpi=320)
    plt.subplot(111)
    pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    plt.tight_layout()
    plt.savefig("./test.png")
    print(f"Graph Visualization saved in {time.time() - start:.2f} s.")
    """
