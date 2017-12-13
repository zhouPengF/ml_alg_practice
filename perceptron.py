

# perceptron

class Model(object):
	classes = None
	w_list = None
	b = 0.0
	def __init__(self,feature_num):
		self.classes = [-1,1]
		self.feature_num = feature_num
		self.w_list = [0.0 for i in range(feature_num)]
		self.b = 0.0
	def calculate_wxb(self,x):
		result = self.b
		for i in range(self.feature_num):
			result += self.w_list[i] * x[i]
		return result
	def adjust(self,x,y,rate):
		for i in range(self.feature_num):
			self.w_list[i] += x[i] * y * rate
		self.b +=  y * rate
	def predict(self,x):
		result = self.b
		for i in range(self.feature_num):
			result += self.w_list[i] * x[i]
		if result > 0:
			return 1
		elif result < 0:
			return -1
class NewTownModel(object):
	def __init__(self,feature_num):
		self.feature_num = feature_num
		self.alpha = [0 for i in range(feature_num)]
		self.b = 0
	def dot(self,x,y):
		result = 0.0
		for a,b in zip(x,y):
			result  += a*b
		return result
	def calculat_Gram(self,x_list):
		matrix = [[0 for i in range(len(x_list))] for j in range(len(x_list))]
		for i in range(len(x_list)):
			for j in range(len(x_list)):
				if i <= j:
					num = self.dot(x_list[i] , x_list[j])
					matrix [i][j] = num
					matrix [j][i] = num
		self.matrix = matrix
	def calculate_x(self,x,x_index,y_list):
		result =self.b
		for i in range(len(y_list)):
			result += y_list[i] * self.alpha[i] * self.matrix [x_index][i]
		return result
	def adjust(self,y,x_index,rate):
		self.alpha[x_index] += rate
		self.b += rate * y
class Perceptron(object):

	def __init__(self):
		pass

	def fit(self,x_list,y_list,rate=0.01):
		model = Model(len(x_list[0]))
		finished = False
		while not finished:
			print("train")
			finished = True
			for x,y in zip(x_list,y_list):
				if model.calculate_wxb(x)*y <=0:
					model.adjust(x,y,rate)
					finished = False
		return model
	def fit_newtown(self,x_list,y_list,rate=1):
		model = NewTownModel(len(x_list))
		model.calculat_Gram(x_list)
		finished = False
		while not finished:
			print("train")
			finished = True
			for x_index in range(len(x_list)):
				tmp_x = model.calculate_x(x_list[x_index],x_index,y_list)
				if y_list[x_index] * tmp_x <=0:
					finished = False
					model.adjust(y_list[x_index],x_index,rate)
		return model


if __name__ == '__main__':
	a = Perceptron()
	x_list = []
	y_list = []
	x_list.append([3,3])
	y_list.append(1)
	x_list.append([4,3])
	y_list.append(1)
	x_list.append([1,1])
	y_list.append(-1)

	model = a.fit(x_list,y_list)
	print(model.w_list,model.b)

	# model = a.fit_newtown(x_list,y_list)
	# print(model.alpha)
	# print(model.b)