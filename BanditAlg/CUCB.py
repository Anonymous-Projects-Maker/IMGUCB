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

class UCB1Struct(ArmBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return 0
        else:
            p = self.totalReward / float(self.numPlayed) + 0.1*np.sqrt(3*np.log(allNumPlayed) / (2.0 * self.numPlayed))
            if p > self.p_max:
                p = self.p_max
                # print 'p_max'
            return p

             
class UCB1Algorithm:
    def __init__(self, G, P, parameter, seed_size, oracle, feedback = 'edge'):
        self.G = G
        self.trueP = P
        self.parameter = parameter  
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        #Initialize P
        self.currentP = {}
        for (u,v) in self.G.edges():
            self.arms[(u,v)] = UCB1Struct((u,v))
            self.currentP[(u,v)] = 0
        self.list_loss = []
        self.TotalPlayCounter = 0
        
    def decide(self):
        self.TotalPlayCounter +=1
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S       
         
    def updateParameters(self, S, live_nodes, live_edges, iter_): 
        
        # loss_out = 0
        # loss_in = 0
        for u in live_nodes:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    self.arms[(u, v)].updateParameters(reward=live_edges[(u,v)])
                else:
                    self.arms[(u, v)].updateParameters(reward=0)
                #update current P
                #print self.TotalPlayCounter
                self.currentP[(u,v)] = self.arms[(u,v)].getProb(self.TotalPlayCounter) 
                # estimateP = self.currentP[u][v]['weight']
                # trueP = self.trueP[u][v]['weight']
                # loss_p += np.abs(estimateP-trueP)
                

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

    def getLoss(self):
        return self.list_loss[-1]

    def getP(self):
        return self.currentP

