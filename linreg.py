import numpy as np
import numpy.linalg as la
import sys

class linreg:
	eps = 0.001								#learning rate (will change depending on the computed loss)
	delta = 1 								#max error allowable, will be edited in generateIO()
	
	A = np.arange(-10,10+1,0.01)	#input
	B = np.zeros(A.shape)		#output
	def __init__ (self):
		print ("START")

		print(len(sys.argv)-1)
		self.inputs = []
		for x in range((len(sys.argv)-1)):
			self.inputs = np.append(self.inputs,float(sys.argv[x+1]))
		
		print("inputs: ",self.inputs)

		self.generateIO()

	def generateIO(self):
		print("A: ",self.A)
		
		#compute for the output values given the fxn and the input values
		indx = len(self.inputs)-1
		temp2 = []
		for x in range(len(self.inputs)):
			self.B += (self.A**indx) * self.inputs[x]
			indx -= 1
			print("x: ",x)

		#create the function matrix (assume 4 values since max 3rd degree polynomial)
		for x in range(4):
			print("1x: ",x,"(3-x): ",3-x)
			if x==0:
				temp2 = (self.A**(3-x)).reshape(1,-1)
			else:
				temp2 = np.append(temp2, (self.A**(3-x)).reshape(1,-1), axis=0)

		
		self.x = np.random.uniform(-1,1,len(temp2))		#set initial values (or initial guess) for function variable coefficients
		self.A = np.transpose(temp2)
		
		
		print("A: ",self.A)
		print("B: ",self.B)
		print(self.eps)

		self.B = self.B + np.random.uniform(-1,1,self.B.shape)

		print("A: ",self.A)
		print("B: ",self.B)
		print(self.eps)

		self.computeGradDescent()


	def computeGradDescent(self):
		iterations = 1;
		temp2 = np.matmul( np.matmul(np.transpose(self.A),self.A), self.x) - np.matmul(np.transpose(self.A), self.B)
		loss = la.norm(temp2,2)	
		if loss < 1000000.0:
			self.delta = float(loss)*0.0000001 			#if loss is too small in the beginning, reduce tolerance
		print("loss: ", loss)

		old_loss = loss
		while loss > self.delta:
			self.x = self.x - self.eps*temp2
			temp2 = np.matmul( np.matmul(np.transpose(self.A),self.A), self.x) - np.matmul(np.transpose(self.A), self.B)
			old_loss = loss
			loss = la.norm(temp2,2)	
			print("iteration: ", iterations, "x: ", self.x, "loss: ", loss, "eps: ", self.eps)
			iterations += 1
			if old_loss < loss:
				self.eps = self.eps*0.10
			else:
				self.eps = self.eps*1.08

		print("Function: ", self.x[0], "x^3 + ", self.x[1], "x^2 + ", self.x[2], "x + ", self.x[3])

implementation = linreg()
