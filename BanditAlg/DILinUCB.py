from random import choice, random, sample
import numpy as np
import networkx as nx
from BanditAlg.BanditAlgorithms import ArmBaseStruct
import datetime

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
		
	def updateParameters(self, updated_A, updated_b):
		self.A += updated_A
		self.b += updated_b
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
	def __init__(self, parameter, node_list, seed_size, oracle, dimension, alpha,  lambda_ , feedback = 'edge'):
		self.param = parameter
		self.node_list = node_list
		self.oracle = oracle
		self.seed_size = seed_size
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.feedback = feedback

		self.users = []  #Nodes
		self.Theta = np.zeros((len(node_list), dimension)) # 每一个node的θ参数
		for idx, u in enumerate(self.node_list):
			self.users.append(LinUCBUserStruct(dimension, lambda_ , u))
			self.Theta[idx, :] = self.users[-1].UserTheta

	def decide(self):
		'''
		实现的是greedy地选择S的过程,相当于是对Oracle的替代.
		'''
		n = len(self.node_list)
		MG = np.zeros((n, 2))
		MG[:, 0] = np.arange(n)			
		influence_UCB = np.matmul(self.Theta, self.param[:, :self.dimension].T) # 每一个node的θ组成矩阵，跟每一个node的x组成的矩阵X，相乘就是概率矩阵。n*n的大小，n是node个数。
		np.fill_diagonal(influence_UCB, 1)
		np.clip(influence_UCB, 0, 1)
		MG[:, 1] = np.sum(influence_UCB, axis=1) # 每一个node影响其他所有节点的概率之和(每一个node的期望spread)
		# print('initialize time', datetime.datetime.now() - startTime)
		
		S = []
		args = []
		temp =  np.zeros(n)
		prev_spread = 0

		for k in range(self.seed_size):
			MG = MG[MG[:,1].argsort()] # 按照第二维（概率和）进行从小到大排序M的行，最后一个影响力最大。
			
			for i in range(0, n-k-1):
				iStartTime = datetime.datetime.now()
				select_node = int(MG[-1, 0]) # 最后一个项的node序号

				# np.sum(np.maximum(influence_UCB[select_node, :], temp)):是加上select_node后,S能够达到的spread的期望,因为temp的和就相当于是之前的spread的期望
				# prev_spread:之前S能达到的总的spread
				# 两项相减,计算的就是select_node的边际收益
				MG[-1, 1] = np.sum(np.maximum(influence_UCB[select_node, :], temp)) - prev_spread  # np.maximum：两个list选出elemen-wise的较大值，给出一个新list。
				if MG[-1, 1] >= MG[-2, 1]: # 如果边际收益大于第二大的节点,则select_node直接是下一个被加入到S中的节点(根据子模性,第二大的节点重新计算后只会更小)
					prev_spread = prev_spread + MG[-1, 1]
					break
				else: # 如果边际收益不大于第二大的节点的话,则将其插入到序列里面,下一次计算的是第二大的节点. 这种对子模性的利用,好像和CELF里面的不同.
					val = MG[-1, 1] # 这一段是将MG[-1]这一项根据其值MG[-1,1]，插入到MG合适的位置中去
					idx = np.searchsorted(MG[:, 1], val) # 
					MG_new = np.zeros(MG.shape)
					MG_new[:idx, :] = MG[:idx, :]
					MG_new[idx, :] = MG[-1, :]
					MG_new[idx+1:	, :] = MG[idx:-1, :]
					MG = MG_new
			

			args.append(int(MG[-1, 0])) # 影响力最大的node在node_list里面的序号
			S.append(self.node_list[int(MG[-1, 0])]) # 影响力最大的node本身的编号
			temp = np.amax(influence_UCB[np.array(args), :], axis=0) # 每一个节点,能被S里面的节点影响到的最大的概率. surrogate function:f(S, v, p)
			MG[-1, 1] = -1 # 最后一个节点被选为seed,就把这个节点去除掉.

		return S

	def updateParameters(self, S, live_nodes, live_edges, _iter):
		A_item = np.array([self.node_list.index(x) for x in self.node_list if x not in S]) # 除了S的所有节点
		b_item = np.array([self.node_list.index(x) for x in live_nodes if x not in S]) # live_nodes中除了S的其他节点
		update_A = self.param[A_item, :self.dimension]
		add_A = np.sum(np.matmul(update_A[:, :, np.newaxis], update_A[:, np.newaxis,:]), axis=0) # 每个node的x做外积,然后相加,最后得出的是一个d*d的矩阵.(搞得复杂了,直接A^T * A)
		add_b = np.sum(self.param[b_item, :self.dimension], axis=0) # 直接这些x进行相加
		for u in S:
			u_idx = self.node_list.index(u)
			self.users[u_idx].updateParameters(add_A, add_b)
			self.Theta[u_idx, :] = self.users[u_idx].UserTheta

	def getCoTheta(self, userID):
		return self.users[userID].UserTheta

	def getP(self):
		return self.currentP		
# 这里用的node feature理论上应该是按照论文里面用laplacian矩阵构建的,但是这个代码也只是随机采出node feature。所有实际上还是有出入。



# class LinUCBAlgorithm:
# 	def __init__(self, G, seed_size, oracle, dimension, alpha,  lambda_ , FeatureDic, feedback = 'edge'):
# 		self.G = G
# 		self.oracle = oracle
# 		self.seed_size = seed_size

# 		self.dimension = dimension
# 		self.alpha = alpha
# 		self.lambda_ = lambda_
# 		self.FeatureDic = FeatureDic
# 		self.feedback = feedback

# 		self.currentP =nx.DiGraph()
# 		self.USER = LinUCBUserStruct(dimension, lambda_ , 0)
# 		for u in self.G.nodes():
# 			for v in self.G[u]:
# 				self.currentP.add_edge(u,v, weight=0)

# 	def decide(self):
# 		S = self.oracle(self.G, self.seed_size, self.currentP)
# 		return S

# 	def updateParameters(self, S, live_nodes, live_edges):
# 		for u in S:
# 			for (u, v) in self.G.edges(u):
# 				featureVector = self.FeatureDic[(u,v)]
# 				if (u,v) in live_edges:
# 					reward = live_edges[(u,v)]
# 				else:
# 					reward = 0
# 				self.USER.updateParameters(featureVector, reward)
# 				self.currentP[u][v]['weight']  = self.USER.getProb(self.alpha, featureVector)
# 	def getCoTheta(self, userID):
# 		return self.USER.UserTheta
# 	def getP(self):
# 		return self.currentP