from random import choice, random, sample
import numpy as np
import networkx as nx

class ArmBaseStruct(object):
    def __init__(self, armID):
        self.armID = armID
        self.totalReward = 0.0
        self.numPlayed = 0
        self.averageReward  = 0.0
        self.p_max = 1
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1
        self.averageReward = self.totalReward/float(self.numPlayed)

        
class eGreedyArmStruct(ArmBaseStruct):
    def getProb(self, epsilon):
        if random() < epsilon: # random exploration
            pta = random()
        else:
            if self.numPlayed == 0:
                pta = 0
            else:
                #print 'GreedyProb', self.totalReward/float(self.numPlayed)
                pta = self.totalReward/float(self.numPlayed)
                if pta > self.p_max:
                    pta = self.p_max
        return pta
        

class eGreedyAlgorithm:
    def __init__(self, G, trueP, seed_size, oracle, epsilon, feedback = 'edge'):
        self.G = G
        self.trueP = trueP
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        # self.loss = []
        self.list_loss = []
        #Initialize P
        self.currentP = {}
        for (u,v) in self.G.edges():
            self.arms[(u,v)] = eGreedyArmStruct((u,v))
            self.currentP[(u,v)] = 0

        self.TotalPlayCounter = 0
        self.epsilon = epsilon

    def decide(self):
        S = self.oracle(self.G, self.seed_size, self.currentP)# self.oracle(self.G, self.seed_size, self.arms)
        return S

    def updateParameters(self, S, live_nodes, live_edges, iter_): 
        # loss = 0
        # count = 0
        for u in live_nodes:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    self.arms[(u, v)].updateParameters(reward=live_edges[(u,v)])
                else:
                    self.arms[(u, v)].updateParameters(reward=0)
                #update current P
                self.currentP[(u,v)] = self.arms[(u,v)].getProb(self.epsilon) 

                # loss += np.abs(self.currentP[u][v]['weight'] - self.trueP[u][v]['weight'] ) 
                # count += 1
        
        # 计算loss，所有的edge，不仅仅是观测到的
        count = 0
        loss_p = 0 
        for u in self.G.nodes():
            for v in self.G[u]:
                estimateP = self.currentP[(u,v)]
                trueP = self.trueP[(u,v)]
                loss_p += np.abs(estimateP-trueP)
                count += 1
        loss_p = loss_p/count
        self.list_loss.append(loss_p)
        
        # loss = loss/count
        # self.loss.append(loss)


    def getP(self):
        return self.currentP


    def getLoss(self):
        # return self.loss
        return self.list_loss[-1]