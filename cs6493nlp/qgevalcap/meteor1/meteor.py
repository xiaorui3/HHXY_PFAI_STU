#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading
import numpy

from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', 
                '-norm',
                # '-t', 'adq' 
                # '-p', '0.85 0.2 0.6 0.75' # alpha beta gamma delta'',
                # '-a', 'data/paraphrase-en.gz', '-m', 'exact stem paraphrase']
                ]
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            # print(gts[i][0])
            hypothesis_tokens = word_tokenize(gts[i][0].decode())
            reference_answers = word_tokenize(res[i][0].decode())
            # print(hypothesis_tokens,reference_answers)
            score = meteor_score.meteor_score([reference_answers], hypothesis_tokens)
            scores.append(score)
            # stat = self._stat(res[i][0], gts[i][0])
            # print('success return ')
            # eval_line += ' ||| {}'.format(stat)

        # self.meteor_p.stdin.write('{}\n'.format(eval_line))
        # for i in range(0,len(imgIds)):
        #     scores.append(float(self.meteor_p.stdout.readline().strip()))
        print(score)
        meanscore = numpy.mean(scores)
        # print('meanscore',meanscore)
        self.meteor_p.__del__()
        # score = float(self.meteor_p.stdout.readline().strip())
        # self.lock.release()
        

        return meanscore, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        print(hypothesis_str)
        print(reference_list)
        hypothesis_str = hypothesis_str.decode().replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list.decode()), 
                                   hypothesis_str))
        score_line += '\n'
        # print score_line
        print(type(score_line),score_line)
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode('utf-8'))
        # return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats 
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score
 
    def __del__(self):
        # self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
