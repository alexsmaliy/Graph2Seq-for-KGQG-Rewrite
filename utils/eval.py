import copy
from collections import defaultdict
from json import encoder
import math

import numpy as np
import torch

def batch_decoded_index2word(decoded_tokens, vocab, oov_dict):
    decoded_batch = []
    if not isinstance(decoded_tokens, list):
        decoded_tokens = decoded_tokens.transpose(0, 1).tolist()
    for i, doc in enumerate(decoded_tokens):
        decoded_doc = []
        for word_idx in doc:
            if word_idx == vocab.SOS:
                continue
            if word_idx == vocab.EOS:
                break
            if word_idx >= len(vocab):
                word = oov_dict.index2word.get((i, word_idx), vocab.unk_token)
                word = " ".join(word)
            else:
                word = vocab.getWord(word_idx)
            decoded_doc.append(word)
        decoded_batch.append(" ".join(decoded_doc))
    return decoded_batch

def eval_batch_output(target_src, vocab, oov_dict, *pred_tensors):
  decoded_batch = [
      batch_decoded_index2word(pred_tensor, vocab, oov_dict)
      for pred_tensor in pred_tensors
  ]
  metrics = [evaluate_predictions(target_src, x) for x in decoded_batch]
  return metrics

def eval_decode_batch(batch, network, vocab, criterion=None, show_cover_loss=False):
    """Test the `network` on the `batch`, return the decoded textual tokens and the Output."""
    with torch.no_grad():
        ext_vocab_size = batch["oov_dict"].ext_vocab_size if batch["oov_dict"] else None
        if criterion is None:
            target_tensor = None
        else:
            target_tensor = batch["targets"]
        out = network(
            batch,
            target_tensor,
            criterion,
            ext_vocab_size=ext_vocab_size,
            include_cover_loss=show_cover_loss,
        )
        decoded_batch = batch_decoded_index2word(out.decoded_tokens, vocab, batch["oov_dict"])
    return decoded_batch, out

def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]
    qg_eval = QGEvalCap(eval_targets, eval_predictions)
    return qg_eval.evaluate()

def precook(s, n=4):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return len(words), counts

def cook_refs(refs, eff=None, n=4): ## lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them."""

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram,count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen))/len(reflen)

    return (reflen, maxcounts)

def cook_test(test, refs, eff=None, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it."""
    reflen, refmaxcounts = refs
    testlen, counts = precook(test, n)
    result = {}

    # Calculate effective reference sentence length.
    if eff == "closest":
        result["reflen"] = min((abs(l-testlen), l) for l in reflen)[1]
    else: ## i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen
    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]
    result['correct'] = [0] * n

    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result

class BleuScorer(object):
    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"
    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        """singleton instance"""
        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match
        self._score = None ## need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        """returns (bleu, len_ratio) pair"""
        return (self.fscore(option=option), self.ratio(option=option)) # where is this implemented?

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs))
        self._score = None
        return self

    def rescore(self, new_test):
        ''' replace test(s) with new test(s), and returns the new score.'''
        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None ## need to recompute

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens))/len(reflens)
        elif option == "closest":
            reflen = min((abs(l-testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option
        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15 ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}

        # for each sentence
        for comps in self.ctest:
            testlen = comps["testlen"]
            self._testlen += testlen

            if self.special_reflen is None: ## need computation
                reflen = self._single_reflen(comps["reflen"], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ["guess", "correct"]:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps["correct"][k]) + tiny) \
                        /(float(comps["guess"][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small) ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1/ratio)
            if verbose > 1:
                print(comps, reflen)

        totalcomps["reflen"] = self._reflen
        totalcomps["testlen"] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps["correct"][k] + tiny) \
                    / (totalcomps["guess"][k] + small)
            bleus.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small) ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1/ratio)
        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)
        self._score = bleus
        return self._score, bleu_list

def calc_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if(string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j] , lengths[i][j - 1])
    return lengths[len(string)][len(sub)]

class Bleu(object):
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        return score, scores

    def method(self):
        return "Bleu"

class Rouge():
    """
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    """

    def __init__(self):
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
            Compute ROUGE-L score given one candidate and references for an image
            :param candidate: str : candidate sentence to be evaluated
            :param refs: list of str : COCO reference sentences for the particular image to be evaluated
            :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate) == 1)
        assert(len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            token_r = reference.split(" ")
            lcs = calc_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"

class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self, verbose=False):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if verbose:
                        print("%s: %0.5f"%(m, sc))
                    # output.append(sc)
                    output[m] = sc
            else:
                if verbose:
                    print("%s: %0.5f"%(method, score))
                output[method] = score
        return output

def eval(out_file, src_file, tgt_file):
    """
        Given a filename, calculate the metric scores for that prediction file
        isDin: boolean value to check whether input file is DirectIn.txt
    """
    pairs = []
    with open(src_file, "r") as infile:
        for line in infile:
            pair = {}
            pair["tokenized_sentence"] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]["tokenized_question"] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, "r") as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)

    for idx, pair in enumerate(pairs):
        pair["prediction"] = output[idx]

    encoder.FLOAT_REPR = lambda o: format(o, ".4f")

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair["tokenized_sentence"]
        res[key] = [pair["prediction"].encode("utf-8")]
        gts[key].append(pair["tokenized_question"].encode("utf-8"))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()
