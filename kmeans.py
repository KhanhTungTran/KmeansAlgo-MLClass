import numpy as np
from numpy.core.fromnumeric import var

class Kmeans():
    def __init__(self, num_clusters, max_iter):
        self.k = num_clusters
        self.max_iter = max_iter
    
    # Khởi tạo theo kiểu ngẫu nhiên:
    def init_centers_random(self, X):
        np.random.RandomState(123456789)
        random_idx = np.random.permutation(X.shape[0])
        centers = X[random_idx[:self.k]]
        return centers

    # Khởi tạo theo kiểu aldaoud:
    def init_centers_aldaoud(self, X):
        centers = []
        vars = np.var(X, axis = 0)
        max_variance_idx = np.argmax(vars)
        X = X[X[:, max_variance_idx].argsort()]
        batch_size = int(self.n/self.k)

        for start in range(0, self.n, batch_size):
            end = min(start + batch_size, self.n)
            centers.append(np.median(X[start:end, :], axis = 0))
        return centers

    def assign_center(self, X):
        labels = np.zeros((X.shape[0]), dtype=int)
        i = 0
        for point in X:
            # Sử dụng khoảng cách Euclide:
            labels[i] = np.argmin(np.linalg.norm(self.centers - point, axis = 1))
            i += 1

        return labels

    def compute_centers(self, X):
        center_lengths = np.zeros((self.k,1), dtype=int)
        new_centers = np.zeros((self.k, X.shape[1]))
        i = 0
        for label in self.labels:
            center_lengths[label] += 1
            new_centers[label] += X[i]
            i += 1
        
        return new_centers/center_lengths
            
    def compute_error(self, X):
        error = 0
        for i in range(len(X)):
            error += np.linalg.norm(X[i] - self.centers[self.labels[i]])

        return error

    def fit(self, X):
        self.n = X.shape[0]
        self.centers = self.init_centers_random(X) # centers là list của k center/cluster

        # Lặp, mỗi lần lặp thực hiện gán nhãn (gán center) cho các điểm dữ liệu 
        # Và tính toán centers mới dựa trên bộ nhãn đó
        for _ in range(self.max_iter):
            self.labels = self.assign_center(X)
            new_centers = self.compute_centers(X)
            self.sum_squared_deviations = self.compute_error(X)
            print(self.sum_squared_deviations)
            # Điều kiện dừng:
            if np.all(new_centers == self.centers):
                break
            
            self.centers = new_centers

        self.sum_squared_deviations = self.compute_error(X)
        print(self.sum_squared_deviations)

    def predict(self, X):
        return self.assign_center(X)