import numpy as np

class Kmeans():
    def __init__(self, num_clusters, max_iters):
        self.k = num_clusters
        self.max_iters = max_iters
    
    # TODO: hiện thực nhiều kĩ thuật khởi tạo khác nhau 
    def init_centers(self):
        pass

    def assign_center(self, X):
        labels = np.zeros((self.k, 1))
        i = 0
        for point in X:
            # Sử dụng khoảng cách Euclide:
            labels[i] = np.argmin(np.linalg.norm(self.centers - point, axis = 1))
            i += 1

        return labels


    def compute_centers(self, X):
        pass

    def compute_errors(self, X):
        pass
    
    def fit(self, X):
        self.centers = self.init_centers(X) # centers là list của k center/cluster

        # Lặp, mỗi lần lặp thực hiện gán nhãn (gán center) cho các điểm dữ liệu 
        # Và tính toán centers mới dựa trên bộ nhãn đó
        for i in range(self.max_iters):
            self.labels = self.assign_center(X)
            new_centers = self.compute_centers(X)
            
            # Điều kiện dừng:
            if new_centers == self.centers:
                break
            
            self.centers = new_centers

        self.sum_squared_deviations = self.compute_errors(X)

    def predict(self):
        pass