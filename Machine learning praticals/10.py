import numpy as np
from scipy.stats import norm

# Data
X = np.array([1, 2, 3, 7, 8, 9])

# Number of clusters
K = 2

# Initialize parameters
np.random.seed(0)
means = np.random.choice(X, K)
variances = np.ones(K)
weights = np.ones(K) / K

# EM Algorithm
for iteration in range(10):

    # ---------- E-Step ----------
    responsibilities = np.zeros((len(X), K))

    for k in range(K):
        responsibilities[:, k] = weights[k] * norm.pdf(
            X, means[k], np.sqrt(variances[k])
        )

    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

    # ---------- M-Step ----------
    Nk = responsibilities.sum(axis=0)

    for k in range(K):
        means[k] = np.sum(responsibilities[:, k] * X) / Nk[k]
        variances[k] = np.sum(
            responsibilities[:, k] * (X - means[k]) ** 2
        ) / Nk[k]
        weights[k] = Nk[k] / len(X)

# Final output
print("Final Means:", means)
print("Final Variances:", variances)
print("Final Weights:", weights)
