import numpy as np
import random
from multiprocessing import Process, Queue


class Sampler(object):
    def sample_function(self, user_train, usernum, itemnum, batch_size, result_queue, SEED, negatives=None,
                        hard_rate=1.0):

        def _sample_with_negatives():
            user = np.random.randint(usernum)
            while len(user_train[user]) == 0:
                user = np.random.randint(usernum)

            pos = random.sample(user_train[user], 1)[0]

            if len(negatives[user]) == 0 or random.random() > hard_rate:
                while True:
                    neg = np.random.randint(itemnum)
                    if neg not in user_train[user]:
                        break
            else:
                neg = random.sample(negatives[user], 1)[0]

            return np.asarray([user, pos, neg], np.int32)

        def _sample():
            user = np.random.randint(usernum)
            while len(user_train[user]) == 0:
                user = np.random.randint(usernum)

            pos = random.sample(user_train[user], 1)[0]
            while True:
                neg = np.random.randint(itemnum)
                if neg not in user_train[user]:
                    break
            return np.asarray([user, pos, neg], np.int32)

        np.random.seed(SEED)
        random.seed(SEED)
        if negatives is None: sample = _sample
        else: sample = _sample_with_negatives
        while True:
            one_batch = np.zeros([batch_size, 3], dtype=np.int32)
            for i in range(batch_size):
                one_batch[i, :] = sample()

            result_queue.put(one_batch)

    def __init__(self, User, usernum, itemnum, batch_size=10000, n_workers=1, negatives=None, hard_rate=1.0):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=self.sample_function, args=(User,
                                                           usernum,
                                                           itemnum,
                                                           batch_size,
                                                           self.result_queue,
                                                           np.random.randint(2e9),
                                                           negatives,
                                                           hard_rate
                                                           )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
