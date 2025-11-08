"""
MP7: Viterbi 1
Implement an HMM-based spelling corrector using the basic Viterbi algorithm.
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial character probabilities, emission characters and transition character-to-character probabilities
    :param pairs:
    :return: intitial character probs, emission characters given character probs, transition of character to character probs
    """
    init_prob = defaultdict(lambda: 0) # {init char: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {output char: {input char: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {prev char: {next char: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    trans_cnt = {} 
    emit_cnt  = {}

    hidden_states = set()
    observed_set  = set()

    for typo, correct in sentences:
        for i in range(len(correct)):
            h = correct[i]
            o = typo[i]
            if h not in emit_cnt:
                emit_cnt[h] = {}
            emit_cnt[h][o] = emit_cnt[h].get(o, 0) + 1
            hidden_states.add(h)
            observed_set.add(o)

        if len(correct) > 0:
            first = correct[0]
            if '^' not in trans_cnt:
                trans_cnt['^'] = {}
            trans_cnt['^'][first] = trans_cnt['^'].get(first, 0) + 1

            for i in range(len(correct) - 1):
                if a not in trans_cnt:
                    trans_cnt[correct[i]] = {}
                trans_cnt[correct[i]][correct[i + 1]] = trans_cnt[correct[i]].get(correct[i + 1], 0) + 1
                hidden_states.add(correct[i])
                hidden_states.add(correct[i + 1])

            last = correct[len(correct) - 1]
            if last not in trans_cnt:
                trans_cnt[last] = {}
            trans_cnt[last]['$'] = trans_cnt[last].get('$', 0) + 1

    Vt = len(hidden_states) + 1
    for prev in trans_cnt:
        total = sum(trans_cnt[prev].values())
        trans_prob[prev] = {}
        for nxt in hidden_states | {'$'}: 
            count = trans_cnt[prev].get(nxt, 0)
            trans_prob[prev][nxt] = (count + 1.0) / (total + 1.0 * Vt)

    for h, row in emit_cnt.items():
        total = sum(row.values())
        Ve = len(row) + 1
        emit_prob[h] = {}
        for o, c in row.items():
            emit_prob[h][o] = (c + 1.0) / (total + 1.0 * Ve)
        emit_prob[h]['<UNSEEN>'] = 1.0 / (total + 1.0 * Ve)
        
    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, char, prev_prob, prev_predict_char_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param char: The i'th observed char
    :param prev_prob: A dictionary of chars to probs representing the max probability of getting to each character at in the
    previous column of the lattice
    :param prev_predict_char_seq: A dictionary representing the predicted character sequences leading up to the previous column
    of the lattice for each character in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each character, and the respective predicted character sequences
    """
    log_prob = {} # This should store the log_prob for all the characters at current column (i)
    predict_char_seq = {} # This should store the character sequence to reach each character at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    states = emit_prob.keys()

    for s in states:
        t = emit_prob[s].get('<UNSEEN>', emit_epsilon)
        e = emit_prob[s].get(char, t)
        v = math.log(e)

        best_prev = None
        best_val = None

        for p in prev_prob:
            tp = trans_prob[p].get(s, epsilon_for_pt)
            val = prev_prob[p] + math.log(tp) + v
            if best_val is None or val > best_val:
                best_val = val
                best_prev = p

        log_prob[s] = best_val
        predict_char_seq[s] = prev_predict_char_seq[best_prev] + [s]

    return log_prob, predict_char_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of pairs of typos and their correct spelling). E.g.,  [(typo1, word1), (typo2, word2), (typo3, word3)]
            test data (list of typos). E.g.,  [typo1, typo2, typo3]
    output: list of corrected typos.
            E.g., [corrected_typo1, corrected_typo2, corrected_typo3]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_char_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_char_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_char_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_char_seq, emit_prob,trans_prob)

        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        output_val = None
        output_state = None
        for s in log_prob:
            if '$' in trans_prob[s]:
                tp_end = trans_prob[s]['$']
            else:
                tp_end = 0.0
            val = log_prob[s] + math.log(tp_end)
            if (output_val is None) or (val > output_val):
                output_val = val
                output_state = s

        if output_state is None:
            boutput_state = max(log_prob, log_prob.get)

        best_path = predict_char_seq[output_state]
        predicts.append(''.join(best_path))

    return predicts