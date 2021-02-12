#!/usr/bin/env python3
import pickle
from matplotlib import pyplot as plt
import numpy as np

test, train = np.zeros((6, 21)), np.zeros((6, 21))
indices = set()

for i in range(6):
    test_dict = pickle.load(open("aurocs_depth_"+str(i)+"_cands.p", "rb"))
    train_dict = pickle.load(open("aurocs_depth_"+str(i)+"_train_cands.p", "rb"))

    for k, j in enumerate(train_dict):
        indices.add(j)
        train[i,k] = train_dict[j]
        test[i,k] = test_dict[j]

indices = sorted(indices, reverse=True)
test_avg = np.average(test, axis=0)
train_avg = np.average(train, axis=0)

print(train)
print(test)

for i in range(6):
   plt.plot(indices, test[i], label="test"+str(i), color="blue")
   plt.plot(indices, train[i], label="train"+str(i), color="green")
# plt.plot(indices, test_avg, label="test"+str(i), color="blue")
# plt.plot(indices, train_avg, label="train"+str(i), color="green")

plt.legend()
plt.show()
