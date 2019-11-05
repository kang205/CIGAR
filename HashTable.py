import numpy as np
import time
import random


class HashTable:
    def __init__(self, B, m=10, r=0):
        self.m = m
        self.n = B.shape[0]
        self.l = B.shape[1]
        self.B = B
        self.nB = np.asarray(B * 2 - 1, np.int8)
        self.r = r
        print 'Build Multi Index Hashing Table (n=%d, m=%d, l=%d)' % (self.n, m, self.l)
        t0 = time.time()
        self.p = np.array([2 ** i for i in range(m)], dtype=np.int32)

        self.ht = [[set() for _ in range(2 ** m)] for _ in range(self.l / m)]

        for i in range(self.n):
            for k in range(self.l / m):
                s = k * m
                t = (k + 1) * m
                self.ht[k][np.sum(B[i, s:t] * self.p, dtype=np.int32)].add(i)
        print 'Finished, elapsed time %g' % (time.time() - t0)

    def query(self, p, train_items, num_items=10):
        ans = set()
        for i in range(self.l / self.m):
            s = i * self.m
            t = (i + 1) * self.m
            ans = ans.union(self.ht[i][np.sum(self.p * p[s:t], dtype=np.int32)])
        ans = ans - set(train_items)

        if len(ans) > num_items:
            ans = np.asarray(list(ans))
            rank = np.argsort(-np.dot(self.nB[ans], np.asarray(p.T * 2 - 1, np.int8)))
            L = ans[rank[:num_items]]
        else:
            if self.r == 0:
                L = list(ans)
            else:
                for i in range(self.l / self.m):
                    s = i * self.m
                    t = (i + 1) * self.m
                    digit = 1
                    for _ in range(self.m):
                        ans = ans.union(self.ht[i][np.sum(self.p * p[s:t], dtype=np.int32) ^ digit])
                        digit *= 2
                ans = ans - set(train_items)

                if len(ans) > num_items:
                    ans = np.asarray(list(ans))
                    rank = np.argsort(-np.dot(self.nB[ans], np.asarray(p.T * 2 - 1, np.int8)))
                    L = ans[rank[:num_items]]
                else:
                    L = list(ans)
        return np.asarray(L)
