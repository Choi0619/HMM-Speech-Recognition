import numpy as np
from copy import deepcopy
import operator
import os
import header
from calc import *

# Give empty dict

words = {}
unigram = {}
bigram = {}

# Load dictionary, bigram, and unigram text files

with open('dictionary.txt', 'r') as dic:
    for line in dic:
        line = line.split()
        if line[0] == "<s>":
            continue
        else:
            words[line[0]] = line
            words[line[0]][0] = 'sil'

with open('bigram.txt', 'r') as bigr:
    for line in bigr:
        line = line.split()
        bigram[(line[0], line[1])] = float(line[2])

with open('unigram.txt', 'r') as unigr:
    for line in unigr:
        line = line.split()
        unigram[line[0]] = float(line[1])

# Bring the list of text files from tst
def file_list():
    test_files = []
    for root, _, files in os.walk("tst", topdown=False):
        for file in files:
            file_path = os.path.join(root, file).replace("\\", "/")
            test_files.append(file_path)
    return test_files


# Model and Structure

class HMM():
    def __init__(self, nstates):
        self.nstates = nstates
        self.tran = np.zeros((nstates, nstates))
        self.mean = {}
        self.variance = {}
        self.gconst = {}
        self.weight = {}
        
    # setter
    def set_state_num(self, val):
        self._nstates = val

    def set_tran(self, val):
        self._tran = val

    def set_mean(self, val):
        self._mean = val

    def set_variance(self, val):
        self._variance = val

    def set_gconst(self, val):
        self._gconst = val

    def set_weight(self, val):
        self._weight = val
        
    # getter
    def get_nstates(self):
        return self._nstates

    def get_tran(self):
        return self._tran

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_gconst(self):
        return self._gconst

    def get_weight(self):
        return self._weight
    

    # Property setting
    nstates = property(get_nstates, set_state_num)
    tran = property(get_tran, set_tran)
    mean = property(get_mean, set_mean)
    variance = property(get_variance, set_variance)
    gconst = property(get_gconst, set_gconst)
    weight = property(get_weight, set_weight)

    # Calculate gconst : Normalizing constant
    def calculate_gconst(self):
        for state, comp in self.weight.keys():
            self.gconst[state, comp] = np.prod(np.sqrt(self.variance[state, comp]))
            self.gconst[state, comp] = 1 / (np.sqrt(2 * np.pi) * self.gconst[state, comp])

    def emss(self, state, vec):
        output = MINUS_INF
        result = {}
        for a, b in self.weight.keys():
            if a != state: continue
            result[b] = 0
            for i in range(nDIMENSION):
                result[b] += np.power(vec[i] - self.mean[a, b][i], 2) / self.variance[a, b][i]
            result[b] = log(self.weight[a, b] * self.gconst[a, b] * exp(-0.5 * result[b]))
        for k in result:
            output = logsum(output, result[k])
        return output

# Building phone hmm

def build_phone_hmm(phones_hmm):
    for obj in range(len(header.phones)):
        phone = header.phones[obj][0]
        nstates = len(header.phones[obj][1])
        phones_hmm[phone] = HMM(nstates)

        # Transition probability
        phones_hmm[phone].tran = np.array(header.phones[obj][1])

        # mean, variance, weight setting
        for state in range(len(header.phones[obj][2])):
            for pdf in range(nPDF):
                phones_hmm[phone].mean[(state+1, pdf+1)] = header.phones[obj][2][state][pdf][1]
                phones_hmm[phone].variance[(state+1, pdf+1)] = header.phones[obj][2][state][pdf][2]
                phones_hmm[phone].weight[(state+1, pdf+1)] = header.phones[obj][2][state][pdf][0]

    for hmm in phones_hmm.keys():
        phones_hmm[hmm].calculate_gconst()


def connect_hmm(former, next):
    # connnecting two HMM
    nS1 = former.nstates
    nS2 = next.nstates
    if former.nstates == 0:
        return next;

    nstates = nS1 + nS2 - 2
    conn_hmm = HMM(nstates)

    # Add former_hmm weight, mean, variance
    conn_hmm.weight = deepcopy(former.weight)
    conn_hmm.mean = deepcopy(former.mean)
    conn_hmm.variance = deepcopy(former.variance)

    # Append next_hmm weight, mean, variance
    for state, comp in next.mean.keys():
        conn_hmm.weight[(state + nS1 - 2, comp)] = deepcopy(next.weight[state, comp])
        conn_hmm.mean[(state + nS1 - 2, comp)] = deepcopy(next.mean[state, comp])
        conn_hmm.variance[(state + nS1 - 2, comp)] = deepcopy(next.variance[state, comp])

    # Upadte transition matrix
    conn_hmm.tran[0:former.nstates-1, 0:former.nstates-1] = former.tran[0:former.nstates-1, 0:former.nstates-1]
    conn_hmm.tran[former.nstates-1:nstates, (former.nstates-1):nstates] = next.tran[1:next.nstates, 1:next.nstates]
    next.tran[0] *= former.tran[former.nstates - 2, former.nstates - 1]
    conn_hmm.tran[former.nstates - 2, former.nstates - 1:nstates] = next.tran[0][1:]
    conn_hmm.tran[former.nstates - 2] = normalize(conn_hmm.tran[former.nstates - 2])

    # cal gconst
    conn_hmm.calculate_gconst()

    return conn_hmm


# Viterbi Algorithm Construction

def viterbi(hmm, x):
    V = {}
    traj = {}
    L = len(x)
    for j in range(hmm.nstates):
        V[(1, j)] = logproduct(log(hmm.tran[0, j]), hmm.emss(j, x[1]))
        traj[(1, j)] = 0
    for t in range(2, L + 1):
        for j in range(hmm.nstates):
            V[(t, j)] = MINUS_INF
            traj[(t, j)] = 0
            for i in range(hmm.nstates):
                if V[(t, j)] < logproduct(V[(t - 1, i)], log(hmm.tran[i, j])):
                    V[(t, j)] = logproduct(V[(t - 1, i)], log(hmm.tran[i, j]))
                    traj[(t, j)] = i
            if j == hmm.nstates - 1:
                V[(t, j)] = MINUS_INF
            else:
                V[(t, j)] = logproduct(V[(t, j)], hmm.emss(j, x[t]))

    V[(L, hmm.nstates - 1)] = MINUS_INF
    for j in range(hmm.nstates):
        if V[(L, hmm.nstates - 1)] < logproduct(V[(L, j)], log(hmm.tran[j, hmm.nstates - 1])):
            V[(L, hmm.nstates - 1)] = logproduct(V[L, j], log(hmm.tran[j, hmm.nstates - 1]))
            traj[(L, hmm.nstates - 1)] = j

    q = [0 for i in range(L)]
    q[L - 1] = traj[L, hmm.nstates - 1]
    for t in range(L - 2, -1, -1):
        q[t] = traj[t + 1, q[t + 1]]
    return q

# define
nDIMENSION = 39
nPDF = 2
PARAM = 0.1
MINUS_INF = -1e+08

# Building phone HMM

phones_hmm = {}
build_phone_hmm(phones_hmm)


# Building word HMM

words_hmm = {}
for word in words:
    # First phone
    words_hmm[word] = deepcopy(phones_hmm[words[word][0]])
    # Append each phone
    for index in range(len(words[word]) - 1):
        next_phone = deepcopy(phones_hmm[words[word][index + 1]])
        words_hmm[word] = connect_hmm(words_hmm[word], next_phone)

# Building final structure

hmm_dict = {}
start = 1
end = 0

hmm = HMM(0)
for word in words:
    hmm = connect_hmm(hmm, words_hmm[word])
    end = hmm.nstates - 1
    hmm_dict[word] = (start, end)
    start = end + 1

combination = [(word_1, word_2) for word_1 in hmm_dict for word_2 in hmm_dict if word_1 != word_2]

for word_1, word_2 in combination:
    (stt1, end1) = hmm_dict[word_1]
    hmm.tran[0, stt1] = unigram[word_1]
    escape = words_hmm[word_1].nstates - 1
    (stt2, end2) = hmm_dict[word_2]
    hmm.tran[end1 - 1, stt2] = 0
    hmm.tran[end1, stt2] = 0
    
    if word_1 == "zero2" and word_1 != "zero2":
        hmm.tran[end1 - 1, stt2 + 3] = bigram[("zero", word_2)] * words_hmm[word_1].tran[escape - 2, escape] * PARAM
        hmm.tran[end1, stt2 + 3] = bigram[("zero", word_2)] * words_hmm[word_1].tran[escape - 1, escape] * PARAM
    elif word_2 == "zero2" and word_1 != "zero2":
        hmm.tran[end1 - 1, stt2 + 3] = bigram[(word_1, "zero")] * words_hmm[word_1].tran[escape - 2, escape] * PARAM
        hmm.tran[end1, stt2 + 3] = bigram[(word_1, "zero")] * words_hmm[word_1].tran[escape - 1, escape] * PARAM
    elif word_1 == "zero2" and word_2 == "zero2":
        hmm.tran[end1 - 1, stt2 + 3] = bigram[("zero", "zero")] * words_hmm[word_1].tran[escape - 2, escape] * PARAM
        hmm.tran[end1, stt2 + 3] = bigram[("zero", "zero")] * words_hmm[word_1].tran[escape - 1, escape] * PARAM
    elif word_1 != "zero2" and word_2 != "zero2":
        if (word_1, word_2) != ('oh', 'zero') and (word_1, word_2) != ('zero', 'oh'):
            hmm.tran[end1 - 1, stt2 + 3] = bigram[(word_1, word_2)] * words_hmm[word_1].tran[escape - 2, escape] * PARAM
            hmm.tran[end1, stt2 + 3] = bigram[(word_1, word_2)] * words_hmm[word_1].tran[escape - 1, escape] * PARAM

        
