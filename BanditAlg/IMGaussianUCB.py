import numpy as np 
import numexpr as ne
import heapq
import networkx as nx
import copy
import tqdm
import random
from multiprocessing import Pool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import logging
logging.basicConfig(level=logging.INFO)

class UserStruct(object):
    def __init__(self, userID, vector):
        self.userID = userID
        self.x = vector
        pass

class IMGaussianUCB(object):
    ''' Gaussian process UCB algorithm for semi-bandit IM '''

    def __init__(self, G, trueP, node_vector, seed_size, oracle, edge_dim, 
                kernel_size=1000, 
                sigma=0.1, 
                z=0.2):
        """
        kernel_size: number of samples to compute the Σ matrix, 用来做GP的样本数
        sigma: kernel matrix里面乘到单位阵上的系数
        gamma, var: RBF函数的参数。K(x1, x2) = var * exp(-gamma * ||x1 - x2||^2)
        z: 计算upper bound时的variance的倍数，μ + z*var
        """
        self.G = G
        self.trueP = trueP # dict类型
        self.node_vector = node_vector # dict
        self.seed_size = seed_size
        self.oracle = oracle
        self.edge_dim = edge_dim
        self.sigma = sigma 
        self.z = z
        self.kernel_size = kernel_size 
        self.decayFactor = 1
        self.decay = 0.92

        self.loss = [] # 每次迭代之后的loss。没有loss_in, loss_out，只有loss_p，在每条边上。
        self.edgeCounter = {} # 每条边被记录到的次数
        self.edgeP = {} # 根据每条边的历史观测记录的历史均值概率
        
        self.trueP_mean = np.mean( list(self.trueP.values()) )
        for u in self.G.nodes(): # node的原始编号
            for v in self.G[u]:
                self.edgeCounter[(u,v)] = 0
                self.edgeP[(u,v)] = 0

        self.n_edges = self.G.size()
        self.topKedges = [] # top K edges. 被观测到频次最多的topK条边，构成tuple list: [(u1,v1), (u2,v2), ...]。用来构成Σ矩阵。这里也许有替代方案。
        self.KernelMat = None # Kernel matrix, 这个矩阵是需要保存的

        self.miuP = {} # 均值/期望(不加upper bound)
        self.ucbP = {} # UCB估计(加upper bound)
        for edge in self.G.edges():
            self.ucbP[edge] = 1.0
            self.miuP[edge] = random.random()
 
    def decide(self):
        S = self.oracle(self.G, self.seed_size, self.ucbP) # 暂且这么写
        return S
        
    def updateParameters(self, S, live_nodes, live_edges, iter_, mod=20, us_sklearn=True):
        '''怎么更新呢？如果是使用频次最高的前k个edge的话，那所有observed edges都应该更新频次，observed edges出现在live_edges中，则reward=1，
        observed_edges没出现在live_edges里面，则reward=0，按照这个来更新self.edgeP，也就是所记录的历史均值。'''

        for u in live_nodes:
            # for v in self.G.nodes[u]: # 致命bug！！！
            for v in self.G[u]:
                if (u,v) in live_edges:
                    reward = 1 # 还是reward=live_edges[(u,v)]?要看lived_edges给到的是什么。
                else: reward = 0
                self.edgeCounter[(u,v)] += 1
                self.edgeP[(u,v)] = (self.edgeP[(u,v)]*(self.edgeCounter[(u,v)]-1) + reward) / self.edgeCounter[(u,v)]

        if (iter_ + 1) % mod == 0 and iter_ != 0: # 更新kernel matrix，更新估计
            # self.topKedges = self.TopKkeys(self.kernel_size, self.edgeCounter)
            self.topKedges = self.TopKkeys_proportional(self.kernel_size)
            self.X, self.Y = self.extract_edge_vectors(self.topKedges)
            if us_sklearn:
                print('learning')
                self.updateParameters_sklearn(live_nodes, live_edges, iter_)
            else:
                # print('using self implementation')
                # self.KernelMat = self.computeKerMat()
                # self.updateEdgeEstimate(live_nodes, live_edges)
                raise NotImplementedError
            
            self.decayFactor *= self.decay
        
        # 计算loss
        self.computeLoss(live_nodes, live_edges)

        logging.info('miu: {:4f}'.format( np.mean(list(self.miuP.values())) ))
        logging.info('ucb: {:4f}'.format( np.mean(list(self.ucbP.values())) ))
        logging.info('hist p: {:4f}'.format( np.mean(list(self.edgeP.values())) ))
        logging.info('true p: {:4f}'.format(self.trueP_mean) )
        logging.info('z: {:4f}'.format(self.decayFactor * self.z))

    def extract_edge_vectors(self, edge_keys):
        X = np.zeros((len(edge_keys), self.edge_dim) )
        Y = np.zeros((len(edge_keys), 1))
        for i, (u,v) in enumerate(edge_keys):
            X[i] = self.genEdgeVector(self.node_vector[u], self.node_vector[v]) # (u,v)是边的键，值是边的vector
            Y[i] =  self.edgeP[(u,v)]
        return X, Y
    
    def updateParameters_sklearn(self, live_nodes, live_edges, iter_):
        """使用sklearn中的GP的实现，这个实现自带了学习核函数的超参数。"""
        # kernel = 1.0**2 * RBF(length_scale=1.0)
        # kernel = RBF()
        self.GP = GaussianProcessRegressor(alpha=self.sigma, normalize_y=False, optimizer='fmin_l_bfgs_b') # 默认带optimizer
        self.GP.fit(self.X, self.Y) # 拟合

        # if iter_ <= 10:
        #     observed_edges = list(self.G.edges())
        # else:
        #     indice = (iter_//10)%5
        #     batch_len = self.G.size()//5
        #     if indice == 4:
        #         observed_edges = list(self.G.edges())[indice*batch_len:]
        #     else:
        #         observed_edges = list(self.G.edges())[indice*batch_len:(indice+1)*batch_len]
        
        # observed_edges = []
        # for i, u in enumerate(live_nodes):
        #     for v in self.G[u]:
        #         observed_edges.append((u, v))

        observed_edges = list(self.G.edges()) # 更新所有边还是观察到的边

        # 计算边的估计与upper bound
        edge_matrix = np.zeros((len(observed_edges), self.edge_dim))
        for i, edge in enumerate(observed_edges):
            u, v = edge[0], edge[1]
            edge_matrix[i] = self.genEdgeVector(self.node_vector[u], self.node_vector[v])
        
        print('{} edges p need to be computed'.format(len(observed_edges)))
        
        num_process = 4
        indices = np.arange(1, num_process) * (len(observed_edges)//num_process) # 起始位置
        edge_matrix_split = np.split(edge_matrix, indices, axis=0)
        # GPS = [copy.copy(self.GP) for _ in edge_matrix_split]
        GPS = [copy.deepcopy(self.GP) for _ in edge_matrix_split]
        # GPS = [self.GP for _ in edge_matrix_split]
        
        p = Pool(num_process+2)
        # results = [p.apply_async(predict_miu_sig_skl, [data, GP]) for data, GP in zip(edge_matrix_split, GPS)] # 这里吧GP也分开，不然GP会被占用，其他进程无法访问，无法并行。
        results = p.starmap(predict_miu_sig_skl, zip(edge_matrix_split, GPS))
        p.close()
        p.join()
        # results_get = [res.get() for res in results]
        results_get = results
        estimate_mat = np.vstack(results_get)

        # 更新边的估计与upper bound        
        upp = np.clip(estimate_mat[:, 0] + estimate_mat[:, 1]*self.decayFactor*self.z, 0.0, 1.0)
        miu = np.clip(estimate_mat[:, 0], 0.0, 1.0)
        
        for i, edge in enumerate(observed_edges):
            self.miuP[edge] = miu[i]
            self.ucbP[edge] = upp[i]

    def TopKkeys_proportional(self, N):
        """对[0, 1]概率分成[0, 0.1], [0, 0.2], ... [0.9, 1.0]十份，按照每个区间的数量，按比例确定抽取个数。再在每一个区间里面使用topk。
        不按比例了，每一个区间等数量采样。
        """
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 最后一个bin是包含1.0的
        hist, _ = np.histogram(list(self.edgeP.values()), bins=bins )
        nums = np.ceil( hist/self.G.size() * N )
        nums = np.clip(nums, 100, 2000).astype(np.int32)
        # nums = [N // 10 for _ in range(10)]
        partion_dics = [{} for i in range(10)] # 10类每一类的edge与出现次数

        for key, val in self.edgeP.items():
            index = int( np.clip(val, 0, 0.99)//0.1 )
            partion_dics[index][key] = self.edgeCounter[key]
        
        result_keys = []
        for dics,n  in zip(partion_dics, nums):
            result_keys.extend(choose_keys(dics, n))


        # pool = Pool(10)
        # result = [pool.apply_async(choose_keys, [dict, n]) for dict, n in zip(partion_dics, nums)]
        # pool.close()
        # pool.join()
        # result_keys = []
        # for res in result:
        #     result_keys.extend(res.get())

        return result_keys
    
    def genEdgeVector(self, x1, x2):
        '''由node vector得到edge vector。取x1的influence和x2的susceptibility
        x1: [susceptibility, influence]
        x2: []'''
        return np.concatenate((x1[1], x2[0])).reshape(-1)
    
    def computeLoss(self, live_nodes, live_edges):
        '''计算当前在所有被观测到的边上的平均loss'''
        count = 0
        loss = 0
        for u in self.G.nodes():
            for v in self.G[u]:
                loss += np.abs( self.miuP[(u, v)] - self.trueP[(u, v)] )  # 这个地方应该用miuP计算loss，用ucbP就不合适了。
                count += 1
        loss = loss/count
        self.loss.append(loss)
    
    def getLoss(self):
        return self.loss[-1]

    # def TopKkeys(self, n, dic):
    #     '''值前K大的K个键，这里也有替换方案'''    
    #     return heapq.nlargest(n, dic, key=dic.get) # [key1, key2, key3, ...]

    # def TopKkeysDis(self, n, dic):
    #     '''依据概率p的个数的分布，来进行不同区间采样不同的x，来组成X
    #     '''
    #     pass

    # def updateEdgeEstimate(self, live_nodes, live_edges):
    #     '''
    #     update probability for each edge。对每条边，更新概率估计。
    #     '''
    #     edge_matrix = np.zeros((self.G.size(), self.edge_dim))
        
    #     for i, edge in enumerate(self.G.edges()):
    #         u, v = edge[0], edge[1]
    #         edge_matrix[i] = self.genEdgeVector(self.node_vector[u], self.node_vector[v])

    #     # print('{} edges p need to be computed'.format(self.G.size())

    #     # 计算边的μ
    #     miu_mat = self.predict_miu(edge_matrix)
        
    #     # 多进程来计算边的sigma
    #     num_process = 20
    #     p = Pool(num_process)

    #     indices = np.arange(num_process) * (self.G.size()//num_process) # 起始位置
    #     edge_matrix_split = np.split(edge_matrix, indices, axis=0)

    #     result_list = p.map(self.predict_sigma, edge_matrix_split)
    #     p.close()  # 关闭进程池，不再接受请求. Prevents any more tasks from being submitted to the pool.
    #     p.join() # Wait for the worker processes to exit. 不加join。子进程结束后会变成僵尸进程，资源不会被回收，会造成内存泄露。

    #     sig_mat = np.vstack(result_list)
        
    #     estimate_mat = miu_mat + self.z * np.diag(sig_mat).reshape(-1,1)
    #     estimate_mat = np.clip(estimate_mat, 0.0, 1.0) # 切到[0,1]之间
    #     miu_mat = np.clip(miu_mat, 0.0, 1.0)
    #     # 执行更新
    #     index = 0
    #     for u in live_nodes:
    #     # for u in self.G.nodes():
    #         for v in self.G[u]:
    #             self.ucbP[u][v]['weight'] = estimate_mat[index][0]
    #             self.miuP[(u,v)] = miu_mat[index][0]
    #             index += 1

    # def computeKerMat(self):
    #     '''计算kernel matrix: (K(X, X)+σI)^-1。这个是需要保存的'''
    #     # X, Y, and kernel matrix。构建X，Y和kernel矩阵
    #     # self.X, self.Y已经计算过
    #     KernelMat = np.linalg.inv( self.kernelF(self.X, self.X, self.gamma, self.var) + self.sigma * np.eye(self.kernel_size) )
    #     return KernelMat

    # def predict_miu(self, x):
    #     '''计算对x的预测分布的μ
    #     x当成矩阵，一行一个样本，按矩阵算。'''
    #     # x = np.array(x).reshape((1, -1))
    #     miu = np.linalg.multi_dot( [self.kernelF(x, self.X, self.gamma, self.var), self.KernelMat, self.Y] )
    #     # miu = miu.flatten()[0]
    #     return miu

    # def predict_sigma(self, X):
    #     '''计算对x的预测分布的σ
    #     x当成矩阵，一行一个样本，一个一个算，因为不需要计算协方差。'''
    #     sig_mat = np.zeros((X.shape[0], 1))
    #     for i, x in enumerate(X):
    #         x = np.array(x).reshape((1, -1))
    #         sig_sq = self.kernelF(x, x, self.gamma, self.var) + self.sigma * np.eye(x.shape[0]) \
    #             - np.linalg.multi_dot( [self.kernelF(x, self.X, self.gamma, self.var), self.KernelMat, self.kernelF(self.X, x, self.gamma, self.var)] )
 
    #         sig = np.sqrt(sig_sq.flatten()[0])

    #         sig_mat[i] = sig
        
    #     return sig_mat

    # def rbfKernel(self, X1, X2, gamma, var):
    #     '''
    #     X1, X2是矩阵，每一行代表一个sample，所以返回的结果是X1.shape[0]*X2.shape[0]的一个矩阵
    #     实现，依据如下等式：
    #     K[i,j] = var * exp(-gamma * ||X1[i] - X2[j]||^2)
    #     i.e. ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    #     '''
    #     X1_norm = np.sum(X1**2, axis=-1)
    #     X2_norm = np.sum(X2**2, axis=-1)
    #     K = ne.evaluate('v * exp(-g * (A + B - 2 * C) )', {
    #         'A': X1_norm[:, None],
    #         'B': X2_norm[None, :],
    #         'C': np.dot(X1, X2.T),
    #         'g': gamma,
    #         'v': var
    #     })
    #     return K

def predict_miu_sig_skl(X, GP):
    """用sklearn的GP计算miu与sigma
    """
    miu_var_mat = np.zeros((X.shape[0], 2))
    y_mean, y_std = GP.predict(X, return_std=True, return_cov=False)
    miu_var_mat[:, 0] = y_mean.flatten()
    miu_var_mat[:, 1] = y_std.flatten()
    return miu_var_mat

def choose_keys(dict, nums):
    list_ = list(dict.keys())
    if len(list_) > nums:
        list_ = heapq.nlargest(nums, dict, key=dict.get)
    return list_


if __name__ == "__main__":
    pass