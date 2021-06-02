import sys
import xml.sax


QUESTION_ID = "1"
ANSWER_ID = "2"


class StackQuestionHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
 
    def startElement(self, name, attrs):
        if name != "row":
            return

        if attrs.get("PostTypeId", "__none__") != QUESTION_ID:
            return

        id = attrs.get("Id", "__none__")
        accep = attrs.get("AcceptedAnswerId", "__none__")
        score = attrs.get("Score", "__none__")
        view = attrs.get("ViewCount", "__none__")
        owner = attrs.get("OwnerUserId", "__none__")
        tags = attrs.get("Tags", "__none__")
        favor = attrs.get("FavoriteCount", "__none__")

        line = [id, accep, score, view, owner, tags, favor]
        if "__none__" in line:
            return

        print("\t".join(line), flush=True)


class StackAnswerHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
 
    def startElement(self, name, attrs):
        if name != "row":
            return

        if attrs.get("PostTypeId", "__none__") != ANSWER_ID:
            return

        id = attrs.get("Id", "__none__")
        score = attrs.get("Score", "__none__")
        owner = attrs.get("OwnerUserId", "__none__")

        line = [id, score, owner]

        if "__none__" in line:
            return

        print("\t".join(line), flush=True)


if __name__ == '__main__':
    fname = sys.argv[1]
    type = sys.argv[2]

    f = open(fname)
    stdout = sys.stdout

    if type == "question":
        with open("./result/questions.csv", "w") as questions_file:
            sys.stdout = questions_file
            xml.sax.parse(f, StackQuestionHandler())
    
    elif type == "answer":
        with open("./result/answers.csv", "w") as answers_file:
            sys.stdout = answers_file
            xml.sax.parse(f, StackAnswerHandler())
