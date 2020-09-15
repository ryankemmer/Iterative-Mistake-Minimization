# Iterative-Mistake-Minimization
My python implementation of the "Iterative Mistake Minimization" algorithm, an explainable approximation of the k-means clustering algorithm proposed in https://arxiv.org/abs/2002.12538. Hoping to add some more functionality to this and get it on pip within the next year!

##Sample Usage

~~~
from imm import imm
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

#initialize with 2 clusters
imm = imm(2).fit(X)
imm.predict([2,3])
~~~

Still need to add a function that returns a decision tree for how clusters are determined! 


