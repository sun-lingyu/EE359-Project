import numpy as np
import pandas as pd
import os


def calc_global_pr(target_tag_list, recommend_num=10):
    comm_tag_score = pd.read_csv("./data/comm_tag_score.csv", index_col=0)
    comm_target_tag_score = comm_tag_score[target_tag_list].mean(axis=1)

    pr_list = []
    for file_name in os.listdir("./data/comm_pr/"):
        comm_id = int(file_name.split(".")[0])
        pagerank = np.load(f"./data/comm_pr/{file_name}", allow_pickle=True).item()
        pagerank = [(k, v * comm_target_tag_score[comm_id], comm_id, comm_target_tag_score[comm_id]) for k, v in pagerank.items()]
        pr_list.extend(pagerank)

    pr_list.sort(key=lambda x: x[1], reverse=True)
    return pr_list[:recommend_num]
