import numpy as np

class Kmeans():
    def __init__(self, num_clusters, max_iters):
        self.k = num_clusters
        self.max_iters = max_iters
    
    # TODO: hiện thực nhiều kĩ thuật khởi tạo khác nhau 
    def init_centers(self, X):
        np.random.RandomState(123456789)
        random_idx = np.random.permutation(X.shape[0])
        centers = X[random_idx[:self.n_clusters]]
        return centers

    def assign_center(self, X):
        labels = np.zeros((self.k, 1))
        i = 0
        for point in X:
            # Sử dụng khoảng cách Euclide:
            labels[i] = np.argmin(np.linalg.norm(self.centers - point, axis = 1))
            i += 1

        return labels

    def compute_centers(self, X):
        center_lengths = np.zeros((self.k, 1))
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
        self.centers = self.init_centers(X) # centers là list của k center/cluster

        # Lặp, mỗi lần lặp thực hiện gán nhãn (gán center) cho các điểm dữ liệu 
        # Và tính toán centers mới dựa trên bộ nhãn đó
        for _ in range(self.max_iters):
            self.labels = self.assign_center(X)
            new_centers = self.compute_centers(X)
            
            # Điều kiện dừng:
            if new_centers == self.centers:
                break
            
            self.centers = new_centers

        self.sum_squared_deviations = self.compute_error(X)

    def predict(self, X):
        return np.argmin(np.linalg.norm(self.centers - X, axis = 1))