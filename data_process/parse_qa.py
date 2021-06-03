import os
import sys

if __name__ == '__main__':
    questions_file = sys.argv[1]
    answers_file = sys.argv[2]
    
    save_file = open("../data/qa.csv", "w")
    sys.stdout = save_file

    af = open(answers_file, "r")
    ad = {}

    for line in af.readlines():
        # [id, score, owner]
        content = line.strip().split("\t")
        ad[content[0]] = content

    af.close()

    qf = open(questions_file, "r")

    for line in qf.readlines():
        # [id, accep, score, view, owner, tags, favor]
        content = line.strip().split("\t")
        ans_content = ad.get(content[1], "__none__")
        if ans_content != "__none__":
            qid, aid = content[0], ans_content[0]
            qowner, aowner = content[4], ans_content[2]
            tags = ";".join([tag[1:] for tag in content[5].split(">")[:-1]])
            if len(tags) == 0:
                continue
            qscore, qview, qfavor = content[2], content[3], content[6]
            ascore = ans_content[1]

            new_content = [qid, aid, qowner, aowner, tags, qscore, qview, qfavor, ascore]
            print("\t".join(new_content), flush=True)
