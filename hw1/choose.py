import numpy as np

# choose 1: B
a = np.arange(60.).reshape(3, 4, 5)
print(np.arange(60.).reshape(3, 4, 5))
b = np.sum(a, axis=0, keepdims=True)
print(b)
print(b.shape)
c = np.sum(a, axis=0)
print(c)
print(c.shape)

# choose 2: A

# choose 3: A

# choose 4: A

# choose 5: C
