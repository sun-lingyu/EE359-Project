import numpy as np
import pandas as pd
import os


"""
def calc_global_pr(target_tag_list, recommend_num=10):
    comm_tag_score = pd.read_csv("./data/comm_tag_score.csv", index_col=0)
    comm_target_tag_score = comm_tag_score[target_tag_list].mean(axis=1)

    pr_list = []
    pr_tmp_list = []
    for file_name in os.listdir("./data/comm_pr/"):
        comm_id = int(file_name.split(".")[0])
        pagerank = np.load(f"./data/comm_pr/{file_name}", allow_pickle=True).item()
        pr_temp = [(k, comm_id, v) for k, v in pagerank.items()]
        pagerank = [(k, v * comm_target_tag_score[comm_id], comm_id, comm_target_tag_score[comm_id]) for k, v in pagerank.items()]

        pr_list.extend(pagerank)
        pr_tmp_list.extend(pr_temp)

    pr_tmp_list = np.array(pr_tmp_list)
    np.save("./data/pr_score.npy", pr_tmp_list)

    import time; s = time.time(); print("Start sorting")
    pr_list.sort(key=lambda x: x[1], reverse=True)
    t = time.time() - s; print(f"Comsuming: {t:.4f} s")
    return pr_list[:recommend_num]
"""


def calc_global_pr(target_tag_list, recommend_num=10):
    import time; s = time.time(); print("Start load community-tag score")
    comm_tag_score = pd.read_csv("./data/comm_tag_score.csv", index_col=0)
    t = time.time() - s; print(f"Comsuming: {t:.4f} s")

    s = time.time(); print("Start calculating target tag score")
    comm_target_tag_score = comm_tag_score[target_tag_list].mean(axis=1)
    t = time.time() - s; print(f"Comsuming: {t:.4f} s")

    s = time.time(); print("Start load PR score")
    pr_score = np.load("./data/pr_score.npy")
    t = time.time() - s; print(f"Comsuming: {t:.4f} s")

    s = time.time(); print("Start calculating weighted PR score")
    tag_score = comm_target_tag_score[pr_score[:, 1]]
    print(pr_score.shape, tag_score.shape)
    pr_score = np.hstack([pr_score, tag_score[:, np.newaxis]])
    pr_score[:, 2] *= pr_score[:, 3]
    pr_score = pr_score.tolist()
    t = time.time() - s; print(f"Comsuming: {t:.4f} s")

    import time; s = time.time(); print("Start sorting")
    pr_score.sort(key=lambda x: x[2], reverse=True)
    t = time.time() - s; print(f"Comsuming: {t:.4f} s")
    return pr_score[:recommend_num]
