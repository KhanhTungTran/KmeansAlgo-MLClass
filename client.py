import numpy as np
from kmeans import Kmeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')

# Import data
df = pd.read_csv('./data/old_faithful.csv')

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of raw data');

plt.show()

# Chuẩn hóa dữ liệu
X_std = MinMaxScaler().fit_transform(df)
# Plot
plt.figure(figsize=(6, 6))
plt.scatter(X_std[:, 0], X_std[:, 1])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of preprocessed data');

plt.show()

# Chạy Kmeans
km = Kmeans(num_clusters=2, max_iter=100)
km.fit(X_std)

centers = km.centers
labels = km.labels
print(centers)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(X_std[labels == 0, 0], X_std[labels == 0, 1],
            c='green', label='cluster 1')
plt.scatter(X_std[labels == 1, 0], X_std[labels == 1, 1],
            c='blue', label='cluster 2')
plt.scatter(centers[:, 0], centers[:, 1], marker='o', s=250,
            c='red', label='centroid')
plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal')
plt.show()