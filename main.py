from __future__ import division
import os
import pickle
import time
import argparse
import math
import tensorflow as tf
from sampler import *
from model import *
from tqdm import tqdm
from util import *
from HashTable import *
import traceback


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--hash_r', default=1, type=int)
parser.add_argument('--batch_size', default=10000, type=int)

parser.add_argument('--C_loss', default='BPR')
parser.add_argument('--C_lr', default=0.001, type=float)
parser.add_argument('--C_dim', default=64, type=int)
parser.add_argument('--C_num_epochs', default=100, type=int)
parser.add_argument('--C_optimizer', default='adam')
parser.add_argument('--C_lam', default=0.001, type=float)
parser.add_argument('--C_m', default=16, type=int)
parser.add_argument('--C_train', default=True, type=str2bool)

parser.add_argument('--F_model', default='BPR')
parser.add_argument('--F_lr', default=0.001, type=float)
parser.add_argument('--F_dim', default=50, type=int)
parser.add_argument('--F_num_epochs', default=100, type=int)
parser.add_argument('--F_optimizer', default='adam')
parser.add_argument('--F_lam', default=0.0001, type=float)
parser.add_argument('--F_m', default=16, type=int)
parser.add_argument('--F_hr', default=0.5, type=float)
parser.add_argument('--F_nc', default=200, type=int)
parser.add_argument('--F_margin', default=1.0, type=float)

args = parser.parse_args()
args.C_alpha = 10.0 / args.C_dim
args.C_m = args.C_dim // args.C_m
args.F_m = args.C_dim // args.F_m
print args
C_logdir = '%s_hash_r=%d_c_loss=%s_c_dim=%d_c_lam=%g_c_m=%d' % (
    args.dataset, args.hash_r, args.C_loss, args.C_dim, args.C_lam, args.C_m)

F_logdir = '%s_hash_r=%d_c_loss=%s_c_dim=%d_c_lam=%g_c_m=%d_f_model=%s_f_dim=%d_f_lam=%g_f_m=%d_F_hr=%g_F_nc=%d' % (
    args.dataset, args.hash_r, args.C_loss, args.C_dim, args.C_lam, args.C_m, args.F_model, args.F_dim, args.F_lam,
    args.F_m,
    args.F_hr, args.F_nc)

if not os.path.isdir(C_logdir):
    os.makedirs(C_logdir)
if not os.path.isdir(F_logdir):
    os.makedirs(F_logdir)

with open(C_logdir + '/args.txt', 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
with open(F_logdir + '/args.txt', 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
args.usernum = usernum
args.itemnum = itemnum
cc = 0
for u in user_train:
    cc += len(u)
print '#user: %d, #item: %d, #actions: %d' % (usernum, itemnum, cc + usernum * 2)
cc = min(cc, 10000000)
num_batch = cc // args.batch_size

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
model = CoarseGrainedModel(args)
saver = tf.train.Saver(max_to_keep=1)
if args.C_train:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(C_logdir, sess.graph)

    sess.run(tf.global_variables_initializer())
    f = open(C_logdir + '/log.txt', 'w')
    import copy

    sampler = Sampler(user_train, usernum, itemnum, batch_size=args.batch_size, n_workers=6)

    # Train C
    best_valid_hr_200 = 0.0
    best_epoch = 0
    best_test = None

    beta = 1.0
    num_iter = 0

    try:
        for epoch in range(1, args.C_num_epochs + 1):
            beta = 1.0 * (math.pow((1. + 10 * (epoch - 1)), 0.5))
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                num_iter += 1
                batch = sampler.next_batch()
                u, pos, neg = batch[:, 0], batch[:, 1], batch[:, 2]
                if num_iter % 100 == 0:
                    auc, c_auc, loss, _, summary = sess.run(
                        [model.auc, model.c_auc, model.loss, model.train, merged],
                        {model.u: u, model.pos: pos, model.neg: neg,
                         model.beta: beta})
                    writer.add_summary(summary, num_iter)
                else:
                    auc, c_auc, loss, _ = sess.run(
                        [model.auc, model.c_auc, model.loss, model.train],
                        {model.u: u, model.pos: pos, model.neg: neg, model.beta: beta})
            print loss, auc, c_auc
            f.write('epoch%d' % epoch)
            f.flush()

            if epoch % (args.C_num_epochs // 10) == 0:
                print 'Evaluating',
                t_valid, t_test = evaluate_C_lookup(model, user_train, user_valid, user_test, args, sess)

                print ''
                print t_valid, t_test
                for k in [10, 100, 200]:
                    f.write('%d: %f, %f, %f, %f, %f, %f\n' % (k, t_valid[0][k], t_valid[1][k], t_valid[2][k],
                                                              t_test[0][k], t_test[1][k], t_test[2][k]))
                f.flush()

                if t_valid[0][200] > best_valid_hr_200:
                    best_valid_hr_200 = t_valid[0][200]
                    best_valid = t_valid
                    best_test = t_test
                    best_epoch = epoch
                    save_path = saver.save(sess, C_logdir + "/%d.ckpt" % epoch)
                    best_P, best_Q = sess.run([model.P, model.Q])
                elif epoch >= best_epoch + args.C_num_epochs // 10 * 3:
                    break

    except:
        print(traceback.format_exc())
        sampler.close()
        f.close()
        exit(1)

    f.write('finished.\n')
    if best_test is not None:
        for k in [10, 100, 200]:
            f.write('%d: %f, %f, %f\n' % (k, best_test[0][k], best_test[1][k], best_test[2][k]))
    print("Tran C done.")
    sampler.close()
else:
    print tf.train.latest_checkpoint(C_logdir)
    saver.restore(sess, tf.train.latest_checkpoint(C_logdir))
    best_P, best_Q = sess.run([model.P, model.Q])
sess.close()
tf.reset_default_graph()

# building candidates
[P, Q] = best_P, best_Q
P = (np.sign(P) + 1) / 2
Q = (np.sign(Q) + 1) / 2
H = HashTable(Q, m=args.F_m, r=args.hash_r)
print 'Getting candidates...'
t0 = time.time()
candidates = defaultdict(list)
sampled_users = random.sample(xrange(usernum), 10000)
if args.F_hr > 0.0:
    try:
        [candidates] = np.load(C_logdir+'/cache.npy')
    except:
        candidates = [None for _ in range(usernum)]
        for u in tqdm(xrange(usernum), total=usernum, ncols=70, leave=False, unit='users'):
            # for u in range(args.usernum):
            candidates[u] = H.query(P[u, :], user_train[u], args.F_nc)
        np.save(C_logdir+'/cache.npy', [candidates])

else:
    try:
        [candidates] = np.load(C_logdir+'/cache.npy')

    except:
        candidates = [None for _ in range(usernum)]
        for u in tqdm(xrange(usernum), total=usernum, ncols=70, leave=False, unit='users'):
            candidates[u] = H.query(P[u, :], user_train[u], args.F_nc)
        np.save(C_logdir+'/cache.npy', [candidates])

print 'done. (time: %g)' % (time.time() - t0)

model = FineGrainedModel(args)
saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
f = open(F_logdir + '/log.txt', 'w')
best_valid_hr_10 = 0.0
best_epoch = 0
best_test = None
if args.F_hr > 0.0:
    sampler = Sampler(user_train, usernum, itemnum, batch_size=args.batch_size, n_workers=6, negatives=candidates,
                      hard_rate=args.F_hr)
else:
    sampler = Sampler(user_train, usernum, itemnum, batch_size=args.batch_size, n_workers=6)
try:
    for epoch in range(1, args.F_num_epochs + 1):
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            batch = sampler.next_batch()
            u, pos, neg = batch[:, 0], batch[:, 1], batch[:, 2]
            auc, loss, _ = sess.run([model.auc, model.loss, model.train], {model.u: u, model.pos: pos, model.neg: neg})
        print loss, auc

        if epoch % (args.F_num_epochs // 10) == 0:
            print 'Evaluating',
            t_valid, t_test = evaluate_F(model, candidates, sampled_users, user_train, user_valid, user_test, args,
                                         sess)

            print ''
            print t_valid, t_test

            f.flush()

            if t_valid[0] > best_valid_hr_10:
                best_valid_hr_10 = t_valid[0]
                best_valid = t_valid
                best_test = t_test
                best_epoch = epoch
                save_path = saver.save(sess, F_logdir + "/%d.ckpt" % epoch)
            elif epoch >= best_epoch + args.F_num_epochs // 10 * 2:
                # pass
                break

except:
    print(traceback.format_exc())
    sampler.close()
    f.close()
    exit(1)

f.write('finished.\n')
if best_test is not None:
    f.write('%f, %f, %f\n' % (best_test[0], best_test[1], best_test[2]))
f.close()
sampler.close()
sess.close()
