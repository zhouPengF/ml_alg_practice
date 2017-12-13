

class KNN(object):

	def __init__(self,k=1):
		self.k = k
		self.x_list = []
		self.y_list = []

	def add_train_data(self,x,y):
		self.x_list.append(x)
		self.y_list.append(y)

	def calculate_distance(self,a,b):
		result = 0.0
		for a_i,b_i in zip(a,b):
			result +=( a_i - b_i) *(a_i - b_i)
		import math
		result = math.sqrt(result)
		return result


if __name__ == '__main__':
	knn = KNN(10)
	knn.add_train_data([1,1,1,1],1)	
	rst = knn.calculate_distance([1,2],[1,2])
	print(rst)