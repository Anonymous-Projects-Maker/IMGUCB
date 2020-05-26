import numpy as np
import networkx as nx
# from BanditAlg.BanditAlgorithms import ArmBaseStruct
import random

class MFUserStruct:
	def __init__(self, featureDimension, lambda_, userID):
		self.userID = userID
		self.dim = featureDimension
		self.A = lambda_*np.identity(n = self.dim)
		self.C = lambda_*np.identity(n = self.dim)

		self.b = np.array([random.random() for i in range(self.dim)])
		self.d = np.array([random.random() for i in range(self.dim)])
		self.AInv = np.linalg.inv(self.A)
		self.CInv = np.linalg.inv(self.C)

		self.theta_out = np.dot(self.AInv, self.b) # 这里也并不是直接存的theta, 而是存A与b，相乘而得到theta
		self.theta_in = np.dot(self.CInv, self.d)

		self.pta_max = 1
		
	def updateOut(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv =  np.linalg.inv(self.A)
		self.theta_out = np.dot(self.AInv, self.b)

	def updateIn(self, articlePicked_FeatureVector, click):
		self.C += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.d += articlePicked_FeatureVector*click
		self.CInv =  np.linalg.inv(self.C)
		self.theta_in = np.dot(self.CInv, self.d)

class MFAlgorithm:
	'''Matrix factorization bandit algorithm
	'''
	def __init__(self, G, P, parameter, seed_size, oracle, dimension, feedback = 'edge'):
		self.G = G
		self.trueP = P
		self.parameter = parameter
		self.oracle = oracle
		self.seed_size = seed_size
		self.q = 0.25

		self.dimension = dimension
		self.feedback = feedback
		self.list_loss = []

		self.currentP = {}
		self.users = {}  # Nodes
		lambda_ = 0.4
		for u in self.G.nodes():
			self.users[u] = MFUserStruct(dimension, lambda_ , u)
			for v in self.G[u]:
				self.currentP[(u, v)] = random.random()

	def decide(self):
		S = self.oracle(self.G, self.seed_size, self.currentP) # 依据自己当前估计出来的P，使用oracle计算出一组seeds
		return S

	def computeLoss(self):
		"""
		全局loss
		"""
		loss_p = 0
		count = 0
		for (u,v) in self.G.edges():
			estimate_p = np.dot(self.users[u].theta_out, self.users[v].theta_in)
			# true_p = self.trueP[u][v]['weight']
			true_p = self.trueP[(u, v)]
			loss_p += np.abs(estimate_p - true_p) # 算出loss，并保存之
			count += 1
		
		loss_p = loss_p/count
		self.list_loss.append(loss_p)


	def updateParameters(self, S, live_nodes, live_edges, it):
		# count = 0
		# loss_p = 0 
		# loss_out = 0
		# loss_in = 0
		for u in live_nodes: # 确定了边是只针对observed edges的
			for (u, v) in self.G.edges(u):
				if (u, v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.users[u].updateOut(self.users[v].theta_in, reward) # 更新
				self.users[v].updateIn(self.users[u].theta_out, reward) # 更新
				self.currentP[(u, v)]  = self.getP(self.users[u], self.users[v], it) # 更新当前对概率的估计

				# compute error
				# estimateP = np.dot(self.users[u].theta_out, self.users[v].theta_in)
				# trueP = self.trueP[u][v]['weight']
				# loss_p += np.abs(estimateP-trueP) # 算出loss，并保存之
				# loss_out += np.linalg.norm(self.users[u].theta_out-self.parameter[u][1], ord =2) # parameter保存真实的vector信息？1代表influence factor, 0代表susceptibility factor？
				# loss_in += np.linalg.norm(self.users[v].theta_in-self.parameter[v][0], ord =2)
				# count += 1

		# self.list_loss.append([loss_p/count, loss_out/count, loss_in/count])
		# self.list_loss.append(loss_p/count)
		self.computeLoss()

	def getP(self, u, v, it):
		alpha_1 = 0.1
		alpha_2 = 0.1
		CB = alpha_1 * np.dot(np.dot(v.theta_in, u.AInv), v.theta_in) + alpha_2 * np.dot(np.dot(u.theta_out, v.CInv), u.theta_out)
		prob = np.dot(u.theta_out, v.theta_in) + CB + 4 * np.power(self.q, it)
		if prob > 1:
			prob = 1
		if prob < 0:
			prob = 0
		return prob		

	def getLoss(self):
		return self.list_loss[-1]
