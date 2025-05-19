#!/usr/bin/env python
__author__ = 'xinya'
import sys
sys.path.append('../')

from bleu1.bleu import Bleu
from meteor1.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize



# sys.setdefaultencoding('utf-8')

class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # print('gts',self.gts)
        # print('res',self.res)

        # =================================================
        # Compute scores
        # =================================================
        
        # gts = word_tokenize(self.gts['0'][0].decode())
        # res = word_tokenize(self.res['0'][0].decode())
        # print(meteor_score.meteor_score([gts], res))
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            # print(self.gts[0],self.res[0])
            score, scores = scorer.compute_score(self.gts, self.res)
            print('success return')
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f"%(m, sc)) 
                    output.append(sc)
            else:
                print("%s: %0.5f"%(method, score)) 
                output.append(score)
        # exit(0)
        # 45.22	29.94	22.01	16.76
        # Bleu_1: 0.26487
        # Bleu_2: 0.10851
        # Bleu_3: 0.05112
        # Bleu_4: 0.02147
        return output

def eval(out_file, src_file, tgt_file, isDIn = False, num_pairs = 500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)


    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]


    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction'].encode('utf-8')]

        ## gts 
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-type", dest="type", default="PFAI", help="squad or nqg")
    # parser.add_argument("-out", "--out_file", dest="out_file", default="/root/autodl-tmp/neural-question-generation/nqg/generated.txt", help="output file to compare")
    # parser.add_argument("-src", "--src_file", dest="src_file", default="/root/autodl-tmp/neural-question-generation/squad_nqg/para-test.txt", help="src file")
    # parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="/root/autodl-tmp/neural-question-generation/squad_nqg/tgt-test.txt", help="target file")
    args = parser.parse_args()

    if args.type == 'nqg':
        out_file = '../nqg/generated.txt'
        src_file = '../squad_nqg/para-test.txt'
        tgt_file = '../squad_nqg/tgt-test.txt'
    elif args.type == 'squad':
        out_file = '../ans_squad/generated.txt'
        src_file = '../ans_squad/src.txt' # ----
        tgt_file = '../ans_squad/golden.txt'
    elif args.type == 'PFAI':
        out_file = '../LLaMA-Factory/PFAI/res_xiaorui/data.txt'
        src_file = '../LLaMA-Factory/PFAI/res_xiaorui/src.txt'
        tgt_file = '../LLaMA-Factory/PFAI/res_xiaorui/input.txt'
    else:
        print('please input again')
    with open('../LLaMA-Factory/PFAI/scores.txt', 'w') as f:
        sys.stdout = f  # 重定向标准输出到文件
        print("scores: \n")
        eval(out_file, src_file, tgt_file)
    # 恢复标准输出到控制台
    sys.stdout = sys.__stdout__


"""
scores: 

success return
Bleu_1: 0.45180
Bleu_2: 0.29961
Bleu_3: 0.22055
Bleu_4: 0.16877
0.46562314246086783
success return
METEOR: 0.41957
success return
ROUGE_L: 0.44777
"""