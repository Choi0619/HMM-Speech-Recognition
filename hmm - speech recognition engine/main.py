import numpy as np
from copy import deepcopy
import operator
import os
from hmm import *

# prediction from Viterbi Algorithm
def state_word(state_seq):
    seq_word = []
    current_word = ""
    for i in range(len(state_seq) - 1):
        for word in hmm_dict:
            (a, b) = hmm_dict[word]
            if state_seq[i] <= b and state_seq[i] >= a + 3:
                if current_word != word and state_seq[i] > state_seq[i - 1]:
                    current_word = word
                    if current_word == "zero2":
                        seq_word.append("zero")
                    else:
                        seq_word.append(current_word)
                elif state_seq[i] < state_seq[i - 1]:
                    current_word = ""
    return seq_word


test_files_path = file_list()

if __name__ == "__main__":
    with open("recognized.txt", "w") as recog:
        recog.write("#!MLF!#" + '\n')
        for path in test_files_path:
            recog.write("\"" + path.replace("txt", 'rec') + "\"\n")
            test_file = open(path)
            x = {}
            for i in range(1, int(test_file.readline().split()[0]) + 1):
                x[i] = list(map(float, test_file.readline().split()))
            print(path, '- Viterbi Algorithm is running...')
            try:
                state_pred = viterbi(hmm, x)
                result = state_word(state_pred)
                for i in range(7):
                    recog.write(result[i] + "\n")
            except:
                recog.write(".\n")
                print('Error', path)
                continue
            recog.write(".\n")
            print(f"{path}... Done")
