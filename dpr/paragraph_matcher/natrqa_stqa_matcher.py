import json
from dpr.paragraph_matcher.tf_idf import PATH_NATURAL_QUES, convert_to_passage








if __name__ == '__main__':
    path = PATH_NATURAL_QUES
    counter = 0
    with open(path + '\\simplified-nq-train.jsonl','rb') as f:
        for line in f: 
            counter += 1
            # line = json.loads(line.decode('utf-8'))
            # long_passage = convert_to_passage(line)
            # if(len(long_passage)==0):
            #     continue
            # print(long_passage)
    print(counter)
    