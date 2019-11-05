import sys
import copy
import random
import time
import heapq
from HashTable import *
import numpy as np
from sampler import *
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict


def evaluate(model, Train, Valid, Test, args, sess, K=[5, 10, 50, 100]):
    NDCG = defaultdict(float)
    HT = defaultdict(float)
    MRR = defaultdict(float)
    NDCG_v = defaultdict(float)
    HT_v = defaultdict(float)
    MRR_v = defaultdict(float)
    maxK = max(K)
    valid_user = 0.0
    usernum = args.usernum
    itemnum = args.itemnum

    if usernum > 10000:
        users = random.sample(xrange(usernum), 10000)
    else:
        users = xrange(usernum)

    for u in users:

        if len(Test[u]) < 1 or len(Valid[u]) < 1: continue
        test_item = list(Test[u])[0]
        valid_item = list(Valid[u])[0]
        rank = sess.run(model.rank, {model.test_u: u})

        valid_user += 1
        # calc test performance
        gt_rank = 0
        found = False
        for i in range(len(rank)):
            if rank[i] in Train[u]:
                gt_rank -= 1
            if rank[i] == test_item:
                gt_rank += i
                found = True
                break
            if gt_rank + i > maxK:
                break
        if found:
            for k in K:
                if gt_rank < k:
                    NDCG[k] += 1 / np.log2(gt_rank + 2)
                    HT[k] += 1
                    MRR[k] += 1 / float(gt_rank + 1)

        # calc valid performance
        gt_rank = 0
        found = False
        for i in range(len(rank)):
            if rank[i] in Train[u]:
                gt_rank -= 1
            if rank[i] == valid_item:
                gt_rank += i
                found = True
                break
            if gt_rank + i > maxK:
                break
        if found:
            for k in K:
                if gt_rank < k:
                    NDCG_v[k] += 1 / np.log2(gt_rank + 2)
                    HT_v[k] += 1
                    MRR_v[k] += 1 / float(gt_rank + 1)

        if valid_user % 1000 == 0:
            print '.',
            sys.stdout.flush()
    for k in K:
        NDCG[k] /= valid_user
        HT[k] /= valid_user
        MRR[k] /= valid_user
        NDCG_v[k] /= valid_user
        HT_v[k] /= valid_user
        MRR_v[k] /= valid_user

    return (HT_v, NDCG_v, MRR_v), (HT, NDCG, MRR)


def data_partition(fname):
    try:
        return np.load('data/%s.npy' % fname)
    except:
        pass
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        if u < 0 or i < 0:
            print u, i
            raise ValueError
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    # assume user/item index starting from 0
    usernum += 1
    itemnum += 1
    random.seed(9001)
    user_train = [[] for _ in range(usernum)]
    user_valid = [[] for _ in range(usernum)]
    user_test = [[] for _ in range(usernum)]

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 5:
            raise ValueError
        else:
            random.shuffle(User[user])
            user_train[user] = np.asarray(User[user][:-2], np.int)
            user_valid[user].append(User[user][-2])
            user_test[user].append(User[user][-1])
    np.save('data/%s.npy' % fname, [user_train, user_valid, user_test, usernum, itemnum])
    return [user_train, user_valid, user_test, usernum, itemnum]



def evaluate_C_lookup(model, Train, Valid, Test, args, sess, K=[10, 100, 200]):
    NDCG = defaultdict(float)
    HT = defaultdict(float)
    MRR = defaultdict(float)
    NDCG_v = defaultdict(float)
    HT_v = defaultdict(float)
    MRR_v = defaultdict(float)
    maxK = max(K)
    valid_user = 0.0
    usernum = args.usernum
    itemnum = args.itemnum

    if usernum > 10000:
        users = random.sample(xrange(usernum), 10000)
    else:
        users = xrange(usernum)

    best_P, best_Q = sess.run([model.P, model.Q])
    # building candidates
    [P, Q] = best_P, best_Q
    P = (np.sign(P) + 1) / 2
    Q = (np.sign(Q) + 1) / 2
    H = HashTable(Q, m=args.C_m, r=args.hash_r)
    print 'Getting candidates...'
    candidates = {}
    t0 = time.time()
    for u in tqdm(users, total=len(users), ncols=70, leave=False, unit='users'):
    # for u in users:
        if len(Test[u]) < 1 or len(Valid[u]) < 1: continue
        test_item = list(Test[u])[0]
        valid_item = list(Valid[u])[0]
        candidates = H.query(P[u, :], Train[u], 200)
        # print l, ',',

        for i in range(len(candidates)):
            current_item = candidates[i]
            if current_item == test_item:
                for k in K:
                    if i < k:
                        NDCG[k] += 1 / np.log2(i + 2)
                        HT[k] += 1
                        MRR[k] += 1 / float(i + 1)
            if current_item == valid_item:
                for k in K:
                    if i < k:
                        NDCG_v[k] += 1 / np.log2(i + 2)
                        HT_v[k] += 1
                        MRR_v[k] += 1 / float(i + 1)

        valid_user += 1

    for k in K:
        NDCG[k] /= valid_user
        HT[k] /= valid_user
        MRR[k] /= valid_user
        NDCG_v[k] /= valid_user
        HT_v[k] /= valid_user
        MRR_v[k] /= valid_user

    return (HT_v, NDCG_v, MRR_v), (HT, NDCG, MRR)


def evaluate_C(model, Train, Valid, Test, args, sess, K=[5, 10, 50, 100]):
    NDCG = defaultdict(float)
    HT = defaultdict(float)
    MRR = defaultdict(float)
    NDCG_v = defaultdict(float)
    HT_v = defaultdict(float)
    MRR_v = defaultdict(float)
    maxK = max(K)
    valid_user = 0.0
    usernum = args.usernum
    itemnum = args.itemnum

    if usernum > 10000:
        users = random.sample(xrange(usernum), 10000)
    else:
        users = xrange(usernum)

    for u in users:
        if len(Test[u]) < 1 or len(Valid[u]) < 1: continue
        test_item = list(Test[u])[0]
        valid_item = list(Valid[u])[0]

        rank = sess.run(model.rank, {model.test_u: u})

        valid_user += 1
        # calc test performance
        gt_rank = 0
        found = False
        for i in range(len(rank)):
            if rank[i] in Train[u]:
                gt_rank -= 1
            if rank[i] == test_item:
                gt_rank += i
                found = True
                break
            if gt_rank + i > maxK:
                break
        if found:
            for k in K:
                if gt_rank < k:
                    NDCG[k] += 1 / np.log2(gt_rank + 2)
                    HT[k] += 1
                    MRR[k] += 1 / float(gt_rank + 1)

        # calc valid performance
        gt_rank = 0
        found = False
        for i in range(len(rank)):
            if rank[i] in Train[u]:
                gt_rank -= 1
            if rank[i] == valid_item:
                gt_rank += i
                found = True
                break
            if gt_rank + i > maxK:
                break
        if found:
            for k in K:
                if gt_rank < k:
                    NDCG_v[k] += 1 / np.log2(gt_rank + 2)
                    HT_v[k] += 1
                    MRR_v[k] += 1 / float(gt_rank + 1)

        if valid_user % 1000 == 0:
            print '.',
            sys.stdout.flush()
    for k in K:
        NDCG[k] /= valid_user
        HT[k] /= valid_user
        MRR[k] /= valid_user
        NDCG_v[k] /= valid_user
        HT_v[k] /= valid_user
        MRR_v[k] /= valid_user

    return (HT_v, NDCG_v, MRR_v), (HT, NDCG, MRR)


def evaluate_F(model, Candidates, sampled_users, Train, Valid, Test, args, sess):
    NDCG = 0.0
    HT = 0.0
    MRR = 0.0
    NDCG_v = 0.0
    HT_v = 0.0
    MRR_v = 0.0
    usernum = args.usernum
    itemnum = args.itemnum

    valid_user = 0.0

    if args.F_model == 'CML':
        sess.run([tf.assign(model.u_embeddings, tf.clip_by_norm(model.u_embeddings, 1.0, 1)),
                  tf.assign(model.i_embeddings, tf.clip_by_norm(model.i_embeddings, 1.0, 1))])

    if usernum > 10000:
        users = random.sample(xrange(usernum), 10000)
    else:
        users = xrange(usernum)
    # users = sampled_users

    rank = np.zeros([itemnum])
    for u in users:
        if len(Test[u]) < 1 or len(Valid[u]) < 1: continue
        valid_user += 1
        if len(Candidates[u]) == 0: continue
        test_item = list(Test[u])[0]
        valid_item = list(Valid[u])[0]
        scores = np.squeeze(sess.run(model.test_scores, {model.test_u: u, model.test_items: Candidates[u]}))
        rank = np.argsort(-scores)

        # calc test performance
        gt_rank = 0
        found = False
        for i in range(min(len(rank), 10)):
            current_item = Candidates[u][rank[i]]
            # if current_item in Train[u]:
            #     raise ValueError
            if current_item == test_item:
                NDCG += 1 / np.log2(i + 2)
                HT += 1
                MRR += 1 / float(i + 1)
            if current_item == valid_item:
                NDCG_v += 1 / np.log2(i + 2)
                HT_v += 1
                MRR_v += 1 / float(i + 1)

    NDCG /= valid_user
    HT /= valid_user
    MRR /= valid_user
    NDCG_v /= valid_user
    HT_v /= valid_user
    MRR_v /= valid_user

    return (HT_v, NDCG_v, MRR_v), (HT, NDCG, MRR)
