import tensorflow as tf
import numpy as np


class CoarseGrainedModel():
    def __init__(self, args, reuse=None):
        with tf.variable_scope('CGM'):
            self.u = tf.placeholder(tf.int32, shape=(None))
            self.test_u = tf.placeholder(tf.int32, shape=(None))
            self.pos = tf.placeholder(tf.int32, shape=(None))
            self.neg = tf.placeholder(tf.int32, shape=(None))
            self.test_u = tf.placeholder(tf.int32, shape=(None))
            self.test_items = tf.placeholder(tf.int32, shape=(None))
            self.beta = tf.placeholder(tf.float32, shape=())
            self.alpha = args.C_alpha
            self.args = args

            usernum = args.usernum
            itemnum = args.itemnum
            pos = self.pos
            neg = self.neg

            self.P = tf.get_variable('u_embs', [usernum, args.C_dim])
            self.Q = tf.get_variable('i_embs', [itemnum, args.C_dim])

            p_u_emb = tf.gather(self.P, self.u)
            q_i_emb = tf.gather(self.Q, self.pos)
            q_j_emb = tf.gather(self.Q, self.neg)
            p_u = tf.tanh(self.beta * p_u_emb)
            q_i = tf.tanh(self.beta * q_i_emb)
            q_j = tf.tanh(self.beta * q_j_emb)

            test_p = tf.sign(tf.expand_dims(tf.gather(self.P, self.test_u), 0))
            test_q = tf.sign(self.Q)

            self.pos_scores = tf.reduce_sum(tf.multiply(p_u, q_i), axis=-1, keepdims=True)
            self.neg_scores = tf.reduce_sum(tf.multiply(p_u, q_j), axis=-1, keepdims=True)

            self.quant_loss = \
                (tf.reduce_mean(tf.square(p_u - tf.sign(p_u_emb))) +
                 tf.reduce_mean(tf.square(q_i - tf.sign(q_i_emb))) +
                 tf.reduce_mean(tf.square(q_j - tf.sign(q_j_emb)))) / 3
            self.c_pos_scores = tf.reduce_sum(tf.multiply(tf.sign(p_u), tf.sign(q_i)), axis=-1, keepdims=True)
            self.c_neg_scores = tf.reduce_sum(tf.multiply(tf.sign(p_u), tf.sign(q_j)), axis=-1, keepdims=True)

            tf.summary.histogram('p_u', p_u)
            tf.summary.histogram('q_i', q_i)
            tf.summary.histogram('c_pos_score', self.c_pos_scores)
            tf.summary.histogram('c_neg_score', self.c_neg_scores)
            tf.summary.histogram('pos_score', self.pos_scores)
            tf.summary.histogram('neg_score', self.neg_scores)

            self.test_all_scores = tf.reduce_sum(tf.multiply(test_p, test_q), axis=-1)
            self.rank = tf.contrib.framework.argsort(tf.random.uniform([itemnum], maxval=0.1) + self.test_all_scores,
                                                     direction='DESCENDING')

            if args.C_loss == 'BPR':
                self.loss = - tf.reduce_sum(tf.log_sigmoid(self.alpha * (self.pos_scores - self.neg_scores)))
                self._loss = self.loss
                self.c_loss = - tf.reduce_sum(tf.log_sigmoid(self.alpha * (self.c_pos_scores - self.c_neg_scores)))
            elif args.C_loss == 'CE':
                self.loss = tf.reduce_sum(-tf.log(tf.sigmoid(self.alpha * self.pos_scores) + 1e-24)) - tf.reduce_sum(
                    tf.log(1 - tf.sigmoid(self.alpha * self.neg_scores) + 1e-24))
            tf.summary.scalar('loss', self.loss)

            self.loss += args.C_lam * sum(map(tf.nn.l2_loss, [p_u_emb, q_i_emb, q_j_emb]))
            self.auc = tf.reduce_mean((tf.sign(self.pos_scores - self.neg_scores) + 1) / 2)
            self.c_auc = tf.reduce_mean((tf.sign(self.c_pos_scores - self.c_neg_scores) + 1) / 2)

            if args.C_optimizer == 'adam':
                self.train = tf.contrib.opt.LazyAdamOptimizer(args.C_lr).minimize(self.loss)
            if args.C_optimizer == 'sgd':
                self.train = tf.train.GradientDescentOptimizer(args.C_lr).minimize(self.loss)
            if args.C_optimizer == 'sgdm':
                self.train = tf.train.MomentumOptimizer(args.C_lr, momentum=0.9).minimize(self.loss)


class FineGrainedModel():
    def __init__(self, args, reuse=None):
        with tf.variable_scope('FGM'):
            self.u = tf.placeholder(tf.int32, shape=(None))
            self.test_u = tf.placeholder(tf.int32, shape=(None))
            self.pos = tf.placeholder(tf.int32, shape=(None))
            self.neg = tf.placeholder(tf.int32, shape=(None))
            self.test_u = tf.placeholder(tf.int32, shape=(None))
            self.test_items = tf.placeholder(tf.int32, shape=(None))
            self.args = args

            usernum = args.usernum
            itemnum = args.itemnum
            pos = self.pos
            neg = self.neg

            if args.F_model == 'BPR':
                self.P = tf.get_variable('u_embs', [usernum, args.F_dim])
                self.Q = tf.get_variable('i_embs', [itemnum, args.F_dim])

                p_u_emb = tf.gather(self.P, self.u)
                q_i_emb = tf.gather(self.Q, self.pos)
                q_j_emb = tf.gather(self.Q, self.neg)
                test_u_emb = tf.tile(tf.expand_dims(tf.gather(self.P, self.test_u), 0),
                                     [tf.shape(self.test_items)[0], 1])
                test_emb = tf.gather(self.Q, self.test_items)

                self.pos_scores = tf.reduce_sum(tf.multiply(p_u_emb, q_i_emb), axis=-1, keepdims=True)
                self.neg_scores = tf.reduce_sum(tf.multiply(p_u_emb, q_j_emb), axis=-1, keepdims=True)

                self.test_scores = tf.reduce_sum(tf.multiply(test_u_emb, test_emb), axis=-1, keepdims=True)
                self.rank = tf.contrib.framework.argsort(
                    tf.reduce_sum(tf.multiply(self.Q, tf.expand_dims(tf.gather(self.P, self.test_u), 0)), axis=-1),
                    direction='DESCENDING')

                self.loss = - tf.reduce_sum(tf.log_sigmoid(self.pos_scores - self.neg_scores))

                self.loss += args.F_lam * sum(map(tf.nn.l2_loss, [p_u_emb, q_i_emb, q_j_emb]))
                self.auc = tf.reduce_mean((tf.sign(self.pos_scores - self.neg_scores) + 1) / 2)

                self.train = tf.contrib.opt.LazyAdamOptimizer(args.F_lr).minimize(self.loss)
            elif args.F_model == 'CML':
                d_emb = args.F_dim
                self.u_embeddings = tf.Variable(tf.random_normal([args.usernum, d_emb], stddev=1 / (d_emb ** 0.5)))
                self.i_embeddings = tf.Variable(tf.random_normal([args.itemnum, d_emb], stddev=1 / (d_emb ** 0.5)))

                # i j r , i jp r

                self.batch_u_emb = tf.clip_by_norm(tf.gather(self.u_embeddings, self.u), 1.0, -1)
                self.batch_i_emb = tf.clip_by_norm(tf.gather(self.i_embeddings, self.pos), 1.0, -1)
                self.batch_j_emb = tf.clip_by_norm(tf.gather(self.i_embeddings, self.neg), 1.0, -1)

                pos_distances = tf.reduce_sum(tf.square(self.batch_u_emb - self.batch_i_emb), 1)
                neg_distances = tf.reduce_sum(tf.square(self.batch_u_emb - self.batch_j_emb), 1)

                self.test_u = tf.placeholder(tf.int32, shape=(None))
                test_u_emb = tf.expand_dims(tf.gather(self.u_embeddings, self.test_u), 0)
                test_i_emb = tf.gather(self.i_embeddings, self.test_items)
                self.test_scores = - tf.reduce_sum(tf.square(test_u_emb - test_i_emb), axis=1)

                self.loss = tf.reduce_sum(tf.maximum(pos_distances - neg_distances + args.F_margin, 0))
                self.auc = tf.reduce_mean((tf.sign(neg_distances - pos_distances) + 1) / 2)

                self.train = tf.contrib.opt.LazyAdamOptimizer(args.F_lr).minimize(self.loss)
            elif args.F_model == 'NeuMF':

                pos_scores = self.NeuMF(self.u, self.pos)
                neg_scores = self.NeuMF(self.u, self.neg, reuse=True)

                self.test_scores = self.NeuMF(tf.tile(tf.reshape(self.test_u, [-1]), [tf.shape(self.test_items)[0]]),
                                              self.test_items, reuse=True)
                print pos_scores, self.test_scores

                self.loss = tf.reduce_sum(-tf.log(pos_scores + 1e-24)) - tf.reduce_sum(tf.log(1 - neg_scores + 1e-24))
                self.auc = tf.reduce_mean((tf.sign(pos_scores - neg_scores) + 1) / 2)

                self.train = tf.contrib.opt.LazyAdamOptimizer(args.F_lr).minimize(self.loss)

    def NeuMF(self, u, i, reuse=False):
        with tf.variable_scope("NeuMF", reuse=reuse):
            self.args.layers = [self.args.F_dim * 4, self.args.F_dim * 2, self.args.F_dim, self.args.F_dim / 2]
            MF_User = tf.get_variable('MF_User', [self.args.usernum, self.args.F_dim//2])
            MF_Item = tf.get_variable('MF_Item', [self.args.itemnum, self.args.F_dim//2])
            MLP_User = tf.get_variable('MLP_User', [self.args.usernum, self.args.layers[0] // 2])
            MLP_Item = tf.get_variable('MLP_Item', [self.args.itemnum, self.args.layers[0] // 2])

            mf_uemb = tf.gather(MF_User, u)
            mf_iemb = tf.gather(MF_Item, i)
            mf_emb = tf.multiply(mf_uemb, mf_iemb)
            mf_emb = tf.reshape(mf_emb, [-1, self.args.F_dim//2])

            mlp_uemb = tf.gather(MLP_User, u)
            mlp_iemb = tf.gather(MLP_Item, i)
            mlp_emb = tf.concat([mlp_uemb, mlp_iemb], axis=1)
            mlp_emb = tf.reshape(mlp_emb, [-1, self.args.layers[0]])

            for idx in xrange(1, len(self.args.layers)):
                mlp_emb = tf.layers.dense(mlp_emb, self.args.layers[idx], activation=tf.nn.relu,
                                          name="layer%d" % idx)

            final_emb = tf.concat([mf_emb, mlp_emb], axis=1)
            scores = tf.layers.dense(final_emb, 1, activation=tf.nn.sigmoid, name="prediction")

            return scores
