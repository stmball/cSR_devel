
import math

import numpy
from sklearn import metrics

from csr.Data import DataStream, parse_conditions

def dcg_at_k(k):
    def score_func_IMPL(truth, conf):
        order = numpy.argsort(conf)[::-1]
        truth = numpy.take(truth, order[:k])
        
        gain = truth
        discounts = numpy.log2(numpy.arange(len(truth)) + 2)
        return numpy.sum(gain / discounts)
    return score_func_IMPL

def ndcg_at_k(k):
    def score_func_IMPL(truth, conf):
        n_rel = n_positive_labels(truth, conf)
        idcg = dcg_at_k(n_rel)([True] * n_rel, [1] * n_rel)
        
        order = numpy.argsort(conf)[::-1]
        truth = numpy.take(truth, order[:k])
        
        gain = truth
        discounts = numpy.log2(numpy.arange(len(truth)) + 2)
        return numpy.sum(gain / discounts) / idcg
    return score_func_IMPL

def precision_at_k(k):
    def score_func_IMPL(truth, conf):
        ii = numpy.argsort(conf)
        count = sum(numpy.array(truth)[ii[-k:]])
        return 1.0*count / k
    return score_func_IMPL

def recall_at_k(k):
    def score_func_IMPL(truth, conf):
        ii = numpy.argsort(conf)
        count = sum(numpy.array(truth)[ii[-k:]])
        total = sum(numpy.array(truth))
        if total == 0: return float('NaN')
        return 1.0*count / total
    return score_func_IMPL

def fscore_at_k(theta):
    p_func = precision_at_k(theta)
    r_func = recall_at_k(theta)
    def score_func_IMPL(truth, conf):
        p = p_func(truth, conf)
        r = r_func(truth, conf)
        if p == 0 or r == 0:
            return 0
        elif math.isnan(p) or math.isnan(r):
            return float('NaN')
        else:
            log_x = numpy.log([p, r])
            return numpy.exp(log_x.sum() / len(log_x))
    return score_func_IMPL


def precision_at_conf(theta):
    def score_func_IMPL(truth, conf):
        ii = numpy.where(numpy.array(conf) > theta)[0]
        if len(ii) == 0: return float('NaN')
        count = sum(numpy.array(truth)[ii])
        return 1.0*count / len(ii)
    return score_func_IMPL

def recall_at_conf(theta):
    def score_func_IMPL(truth, conf):
        ii = numpy.where(numpy.array(conf) > theta)[0]
        count = sum(numpy.array(truth)[ii])
        total = sum(numpy.array(truth))
        if total == 0: return float('NaN')
        return 1.0*count / total
    return score_func_IMPL

def fscore_at_conf(theta):
    p_func = precision_at_conf(theta)
    r_func = recall_at_conf(theta)
    def score_func_IMPL(truth, conf):
        p = p_func(truth, conf)
        r = r_func(truth, conf)
        if p == 0 or r == 0:
            return 0
        elif math.isnan(p) or math.isnan(r):
            return float('NaN')
        else:
            log_x = numpy.log([p, r])
            return numpy.exp(log_x.sum() / len(log_x))
    return score_func_IMPL


def n_positive_labels(truth, conf):
    return sum(numpy.array(truth))

_eval_metric_func = {
    'AUC':   metrics.roc_auc_score,
    'AP':    metrics.average_precision_score,
    'n_pos': n_positive_labels
}

def resolve_metric_by_string(s):
    if "@" in s:
        s = s.split("@")
        try:
            theta = int(s[1])
            theta_type = 'k'
        except ValueError:
            try:
                theta = float(s[1])
                theta_type = 'conf'
            except ValueError:
                raise ValueError('Expected numerical value for threshold')
        metric = s[0]
        metric_abbreviation = {
            "R": "recall",
            "P": "precision",
            "F": "fscore"
        }
        if metric in metric_abbreviation:
            metric = metric_abbreviation[metric]
        metric = metric.lower()
        try:
            metric_funcname = '%s_at_%s' % (metric, theta_type)
            return globals()[metric_funcname](theta)
        except KeyError:
            raise ValueError("Unknown metric function '%s'" % (metric_funcname))
    else:
        try:
            return _eval_metric_func[s]
        except KeyError:
            raise ValueError("Unknown metric function '%s'" % (s))

if __name__ == "__main__":
    import sys, traceback
    import argparse
    
    parser = argparse.ArgumentParser(description = '')
    # ~~~~~~ Basic Arguments ~~~~~~                                                                  
    parser.add_argument('--data',             nargs  = '+', required = True,
                        help = 'Input files in DataStream format')
    parser.add_argument('--get',              nargs  = '+',
                        help = 'Restrict selection to rows for which the condition holds')
    parser.add_argument('--metrics',          nargs = '*',
                        default = ['AUC'], type = str)
    parser.add_argument('--fold',             type = str)

    args = parser.parse_args()
    
    records = []
    for filename in args.data:
        d = DataStream.parse(open(filename).read())
        d.add_column('filename', filename)
        records.append(d)
    data = DataStream(*records[0].header)
    for d in records:
        data.merge(d)

    row_conditions = args.get and args.get or []
    row_conditions = parse_conditions(row_conditions)
    data = data.select(row_conditions)
    
    for metric in args.metrics:
        score_func = resolve_metric_by_string(metric)
        if args.fold:
            print('%s per fold:' % metric)
            scores = []
            fold_values = numpy.unique(data[args.fold])
            fold_names_max_len = max(map(len, map(str, fold_values)))
            for value in fold_values:
                left_col = str(value).rjust(fold_names_max_len)
                d = data[numpy.where(numpy.array(data[args.fold]) == value)[0]]

                truth = d.label
                conf = list(map(float, d.confidence))
                if sum(numpy.array(truth) == 'Y') == 0:
                    print('  %s: --' % (left_col))
                    continue
                if len(numpy.unique(conf)) == 1:
                    print('  %s: --' % (left_col))
                    continue
                score = score_func(numpy.array(truth) == 'Y', conf)
                if math.isnan(score):
                    print('  %s: --' % (left_col))
                    continue
                
                scores.append(score)
                print('  %s: %.3f' % (left_col, score))
            if len(scores) > 0:
                mean_score = numpy.mean(scores)
                print('  %s: %.3f' % ('mean'.rjust(fold_names_max_len), mean_score))
        else:
            print('%s:' % metric)
            truth = data.label
            conf = list(map(float, data.confidence))
            if sum(numpy.array(truth) == 'Y') == 0:
                print('--')
                continue
            if len(numpy.unique(conf)) == 1:
                print('  %s: --' % (left_col))
                continue
            score = score_func(numpy.array(truth) == 'Y', conf)
            if math.isnan(score):
                print('--')
                continue
            print('%.3f' % (score))
