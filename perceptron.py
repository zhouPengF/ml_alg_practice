

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

if __name__ == '__main__':
	a = Perceptron()
	x_list = []
	y_list = []
	x_list.append([1,1,1])
	y_list.append(1)
	x_list.append([2,2,2])
	y_list.append(1)
	x_list.append([1,0,0])
	y_list.append(-1)
	model = a.fit(x_list,y_list)
	print(model.w_list,model.b)