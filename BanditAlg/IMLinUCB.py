from random import choice, random, sample
import numpy as np
import networkx as nx
from BanditAlg.CUCB import ArmBaseStruct

class LinUCBUserStruct:
	def __init__(self, featureDimension,lambda_, userID, RankoneInverse = False):
		self.userID = userID
		self.d = featureDimension
		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.zeros(self.d)

		self.RankoneInverse = RankoneInverse

		self.pta_max = 1
		
	def updateParameters(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		if self.RankoneInverse:
			temp = np.dot(self.AInv, articlePicked_FeatureVector)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(articlePicked_FeatureVector),temp))
		else:
			self.AInv =  np.linalg.inv(self.A)

		self.UserTheta = np.dot(self.AInv, self.b)
		
	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, article_FeatureVector):
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		pta = mean + alpha * var
		if pta > self.pta_max:
			pta = self.pta_max
		#print self.UserTheta
		#print article_FeatureVector
		#print pta, mean, alpha*var
		# if mean >0:
		# 	print 'largerthan0', mean
		return pta

class N_LinUCBAlgorithm:
	'''IMLinUCB
	'''
	def __init__(self, G, P, parameter, seed_size, oracle, dimension, alpha,  lambda_ , FeatureDic, FeatureScaling, feedback = 'edge'):
		self.G = G
		self.trueP = P
		self.parameter = parameter		
		self.oracle = oracle
		self.seed_size = seed_size
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.FeatureDic = FeatureDic
		self.FeatureScaling = FeatureScaling
		self.feedback = feedback

		self.currentP = {}
		self.users = {}  #Nodes
		self.arms = {}
		for u in self.G.nodes():
			self.users[u] = LinUCBUserStruct(dimension, lambda_ , u)
			for v in self.G[u]:
				self.arms[(u,v)] = ArmBaseStruct((u,v))
				self.currentP[(u, v)] = random()
		self.list_loss = []

	def decide(self):
		S = self.oracle(self.G, self.seed_size, self.currentP)
		return S

	def updateParameters(self, S, live_nodes, live_edges, _iter):
		count = 0
		loss_p = 0 
		loss_out = 0
		loss_in = 0
		for u in live_nodes: # 这里是只对lived_nodes发射的边进行学习
			for (u, v) in self.G.edges(u):
				featureVector = self.FeatureScaling*self.FeatureDic[(u,v)] # 边的feature
				if (u,v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.arms[(u, v)].updateParameters(reward=reward) # arms有何用？
				# reward = self.arms[(u, v)].averageReward    #####Average Reward
				self.users[u].updateParameters(featureVector, reward)
				self.currentP[(u, v)]  = self.users[v].getProb(self.alpha, featureVector) # user[v]的UserTheta，也就是每一个节点都有一个theta？theta不是公共的吗

				estimateP = self.currentP[(u, v)]
				trueP = self.trueP[(u, v)]
				loss_p += np.abs(estimateP-trueP)
				count += 1
		self.list_loss.append(loss_p/count)

	def getCoTheta(self, userID):
		return self.users[userID].UserTheta # get到每一个结点的theta

	def getP(self):
		return self.currentP

	def getLoss(self):
		return self.list_loss[-1]

class LinUCBAlgorithm:
	def __init__(self, G, seed_size, oracle, dimension, alpha,  lambda_ , FeatureDic, feedback = 'edge'):
		self.G = G
		self.oracle = oracle
		self.seed_size = seed_size

		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.FeatureDic = FeatureDic
		self.feedback = feedback

		self.currentP = {}
		self.USER = LinUCBUserStruct(dimension, lambda_ , 0) # 一个单独的user，相当于是所有edge共享这个theta。这个才符合IMLinUCB的算法。但是下面用到的edges，只是从S里面发射出来的edge，又不符合了。
		for u in self.G.nodes():
			for v in self.G[u]:
				self.currentP[(u,v)] = 0

	def decide(self):
		S = self.oracle(self.G, self.seed_size, self.currentP)
		return S

	def updateParameters(self, S, live_nodes, live_edges):
		for u in S:
			for (u, v) in self.G.edges(u): # 为什么用的是S集合发射出来的边，而不是所有live_nodes发射出来的边（即observed edges）。
				featureVector = self.FeatureDic[(u,v)]
				if (u,v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.USER.updateParameters(featureVector, reward)
				self.currentP[(u,v)]  = self.USER.getProb(self.alpha, featureVector)
	def getCoTheta(self, userID):
		return self.USER.UserTheta
	def getP(self):
		return self.currentP