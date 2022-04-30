"""
eval_metrics.pu
 - evaluation metrics: MRR, MP
"""

import csv
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score


def evalMRR_online(src_words, src_years, tar_words, tar_years, src_nebs_list):
    #query_target_gold = readTestData(test_fn)
    sum_MRR = 0
    #sum_count = 0
    sum_count = len(src_words)
    missing_count = 0

    MRR_temporal = []
    MRR_temporal_sep = []

    for cur_t in range(1991, 2017):
        sum_MRR_sep = 0
        sum_count_sep = 0
        for i in range(0, len(src_words)):
            src_w = src_words[i]
            src_y = src_years[i]
            tar_w = tar_words[i]
            tar_y = tar_years[i]

            if not ((src_y == cur_t and tar_y < cur_t) or (src_y < cur_t and tar_y == cur_t)):
                continue
           
            sum_count_sep += 1

            nebs = src_nebs_list[src_y][src_w][tar_y]
            if len(nebs) == 0:
                missing_count += 1
                continue
            
            #sum_count += 1
            nebs = nebs[:10]
            nebs_w = [tup[0] for tup in nebs]
            if tar_w not in nebs_w:
                continue
            rank = nebs_w.index(tar_w) + 1
            sum_MRR += 1.0/rank
            sum_MRR_sep += 1.0/rank

            '''
            if i % 100 == 0:
                print(src_w, src_y, tar_w, tar_y)
                print(nebs)
                print()
            '''
        if sum_count == 0:
            MRR = 0.
        else:
            MRR = sum_MRR / float(sum_count)

        if sum_count == 0:
            MRR_sep = 0.
        else:
            MRR_sep = sum_MRR_sep / float(sum_count)

        MRR_temporal.append(MRR)
        MRR_temporal_sep.append(MRR_sep)
        # print("missing count: {}/{}".format(missing_count, sum_count))
        # print("MRR: %.5f" % (MRR))
        # print("MRR(sep): %.5f /(%d)" % (MRR_sep, sum_count_sep))

    print("missing count: {}/{}".format(missing_count, sum_count))
    print("MRR: %.5f" % (MRR))

    return MRR_temporal, MRR_temporal_sep


def evalMP_online(src_words, src_years, tar_words, tar_years, src_nebs_list, topk):
    # topk: 1, 3, 5, 10
    sum_prec = 0
    #sum_count = 0
    sum_count = len(src_words)
    missing_count = 0

    MP_temporal = []
    MP_temporal_sep = []

    for cur_t in range(1991, 2017):
        sum_prec_sep = 0
        sum_count_sep = 0
        for i in range(0, len(src_words)):
            src_w = src_words[i]
            src_y = src_years[i]
            tar_w = tar_words[i]
            tar_y = tar_years[i]

            if not ((src_y == cur_t and tar_y < cur_t) or (src_y < cur_t and tar_y == cur_t)):
                continue

            sum_count_sep += 1

            nebs = src_nebs_list[src_y][src_w][tar_y]
            if len(nebs) == 0:
                missing_count += 1
                continue
            #sum_count += 1
            nebs = nebs[:topk]
            nebs_w = [tup[0] for tup in nebs]
            if tar_w in nebs_w:
                sum_prec += 1.0
                sum_prec_sep += 1.0

        if sum_count == 0:
            mp = 0.
        else:
            mp = float(sum_prec) / sum_count

        if sum_count == 0:
            mp_sep = 0.
        else:
            mp_sep = float(sum_prec_sep) / sum_count

        MP_temporal.append(mp)
        MP_temporal_sep.append(mp_sep)
        # print("MP@%d: %.5f" % (topk, mp))
        # print("MP@%d(sep): %.5f /(%d)" % (topk, mp, sum_count_sep))

    print("MP@%d: %.5f" % (topk, mp))

    return MP_temporal, MP_temporal_sep


def evalMRR_cur(cur_t, src_words, src_years, tar_words, tar_years, src_nebs_list):
    #query_target_gold = readTestData(test_fn)
    sum_MRR = 0
    sum_MRR_sep = 0
    #sum_count = 0
    sum_count = len(src_words)
    sum_count_sep = 0
    missing_count = 0

    for i in range(0, len(src_words)):
        src_w = src_words[i]
        src_y = src_years[i]
        tar_w = tar_words[i]
        tar_y = tar_years[i]

        if not (src_y <= cur_t and tar_y <= cur_t):
            continue
       
        if src_y == cur_t or tar_y == cur_t:
            sum_count_sep += 1

        nebs = src_nebs_list[src_y][src_w][tar_y]
        if len(nebs) == 0:
            missing_count += 1
            continue
        #sum_count += 1
        nebs = nebs[:10]
        nebs_w = [tup[0] for tup in nebs]
        if tar_w not in nebs_w:
            continue
        rank = nebs_w.index(tar_w) + 1
        sum_MRR += 1.0/rank
        if src_y == cur_t or tar_y == cur_t:
            sum_MRR_sep += 1.0/rank
        '''
        if i % 100 == 0:
            print(src_w, src_y, tar_w, tar_y)
            print(nebs)
            print()
        '''
    if sum_count == 0:
        MRR = 0.
    else:
        MRR = sum_MRR / float(sum_count)

    if sum_count_sep == 0:
        MRR_sep = 0.
    else:
        MRR_sep = sum_MRR_sep / float(sum_count_sep)

    print("missing count: {}".format(missing_count))
    print("MRR: {}, sum_count: {}".format(MRR, sum_count))
    print("MRR(sep): %.5f /(%d)" % (MRR_sep, sum_count_sep))
    return MRR, MRR_sep


def evalMP_cur(cur_t, src_words, src_years, tar_words, tar_years, src_nebs_list, topk):
    # topk: 1, 3, 5, 10
    sum_prec = 0
    sum_prec_sep = 0
    #sum_count = 0
    sum_count = len(src_words)
    sum_count_sep = 0
    missing_count = 0

    for i in range(0, len(src_words)):
        src_w = src_words[i]
        src_y = src_years[i]
        tar_w = tar_words[i]
        tar_y = tar_years[i]

        if not (src_y <= cur_t and tar_y <= cur_t):
            continue

        if src_y == cur_t or tar_y == cur_t:
            sum_count_sep += 1

        nebs = src_nebs_list[src_y][src_w][tar_y]
        if len(nebs) == 0:
            missing_count += 1
            continue
        #sum_count += 1
        nebs = nebs[:topk]
        nebs_w = [tup[0] for tup in nebs]
        if tar_w in nebs_w:
            sum_prec += 1.0

            if src_y == cur_t or tar_y == cur_t:
                sum_prec_sep += 1

    if sum_count == 0:
        mp = 0.
    else:
        mp = float(sum_prec) / sum_count

    if sum_count_sep == 0:
        mp_sep = 0.
    else:
        mp_sep = float(sum_prec_sep) / sum_count_sep

    print("MP@%d: %.5f" % (topk, mp))
    print("MP@%d(sep): %.5f /(%d)" % (topk, mp, sum_count_sep))
    return mp, mp_sep


def evalMRR(src_words, src_years, tar_words, tar_years, src_nebs_list):
    #query_target_gold = readTestData(test_fn)
    sum_MRR = 0
    sum_count = 0
    missing_count = 0

    for i in range(0, len(src_words)):
        src_w = src_words[i]
        src_y = src_years[i]
        tar_w = tar_words[i]
        tar_y = tar_years[i]
       
        nebs = src_nebs_list[src_y][src_w][tar_y]
        if len(nebs) == 0:
            missing_count += 1
            continue
        sum_count += 1
        nebs = nebs[:10]
        nebs_w = [tup[0] for tup in nebs]
        if tar_w not in nebs_w:
            continue
        rank = nebs_w.index(tar_w) + 1
        sum_MRR += 1.0/rank
        '''
        if i % 100 == 0:
            print(src_w, src_y, tar_w, tar_y)
            print(nebs)
            print()
        '''
    MRR = sum_MRR / float(sum_count)
    print("missing count: {}".format(missing_count))
    print("MRR: {}, sum_count: {}".format(MRR, sum_count))
    return MRR


def evalMP(src_words, src_years, tar_words, tar_years, src_nebs_list, topk):
    # topk: 1, 3, 5, 10
    sum_prec = 0
    sum_count = 0
    missing_count = 0

    for i in range(0, len(src_words)):
        src_w = src_words[i]
        src_y = src_years[i]
        tar_w = tar_words[i]
        tar_y = tar_years[i]

        nebs = src_nebs_list[src_y][src_w][tar_y]
        if len(nebs) == 0:
            missing_count += 1
            continue
        sum_count += 1
        nebs = nebs[:topk]
        nebs_w = [tup[0] for tup in nebs]
        if tar_w in nebs_w:
            sum_prec += 1.0

    mp = float(sum_prec) / sum_count
    print("MP@{}: {}, sum_count: {}".format(topk, mp, sum_count))
    return mp


def evalNMI(U, V):
    NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')
    return NMI(U, V)



def evalFbeta(labels, clusters, beta=5):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    total_num = labels.size
    total_pair_num = 0.0
    for i in range(total_num - 1):
        for j in range(i+1, total_num):
            total_pair_num += 1.0

            l_i = labels[i]
            c_i = clusters[i]
            l_j = labels[j]
            c_j = clusters[j]

            if l_i == l_j:
                if c_i == c_j:
                    TP += 1.0
                else:
                    #FN += 1.0
                    FP += 1.0
            elif c_i == c_j:
                    #FP += 1.0
                    FN += 1.0

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F_beta_score = (beta*beta + 1) * P * R / (beta*beta*P + R)

    return F_beta_score
