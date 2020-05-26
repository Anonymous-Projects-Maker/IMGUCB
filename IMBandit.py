import time
import os
import pickle 
import pandas
import datetime
import random
import numpy as np
import networkx as nx
import argparse
import sys
from pathlib import Path
import logging

import conf
from BanditAlg.CUCB import UCB1Algorithm
from BanditAlg.greedy import eGreedyAlgorithm 
from BanditAlg.IMFB import MFAlgorithm
from BanditAlg.IMLinUCB import N_LinUCBAlgorithm # DILinUcb和IMLinUCB里面都有这个类，暂且先用IMLinUCB里面的
from BanditAlg.IMFB_MLE import IM_mle
from BanditAlg import IMGaussianUCB
from IC.IC import runICmodel_n, runICmodel_node_feedback
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp
from Oracle.random import random_seeds

logging.basicConfig(level=logging.INFO)


class SimulateAlgos():
    def __init__(self, G, P, oracle, seed_size, iterations, dataset, algorithms, resdir=None):
        self.G = G
        self.TrueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.actual_iters = 0 # 实际运行了多少iteration
        self.dataset = dataset
        self.algorithms = algorithms
        self.result_dir = resdir if resdir is not None else 'SimulationResults/'

        self.start_time = datetime.datetime.now()
        self.start_time_str = self.start_time.strftime('%y_%m_%d_%H_%M_%S') 
        self.cumul_reward = {}
        self.round_reward = {}
        self.loss = {}
        self.round_time = {}

    def runAlgorithms(self):
        for alg_name, alg in list(self.algorithms.items()) + [('Oracle', None), ('Random', None)] :
            self.round_reward[alg_name] = [] # 每一次运行（每一轮）的reward
            self.cumul_reward[alg_name] = [] #所有轮的累计reward
            self.round_time[alg_name] = []
            self.loss[alg_name] = []
            
        start_time = time.time()
        for i in range(self.iterations):
            logging.info('-'*20)
            logging.info("Iter {}/{}".format(i+1, self.iterations))

            for alg_name, alg in list(self.algorithms.items()) + [('Oracle', None), ('Random', None)]: 
                if alg_name == 'Oracle':
                    t1 = time.time()
                    S = self.oracle(self.G, self.seed_size, self.TrueP) # oracle算法根据真实edge probability算出来的最优seeds(只是近似最优)
                    t = time.time() - t1
                    reward, live_nodes, live_edges = runICmodel_n(self.G, S, self.TrueP)
                    
                elif alg_name == 'Random':
                    t1 = time.time()
                    S = random_seeds(self.G, self.seed_size, self.TrueP)
                    t = time.time() - t1
                    reward, live_nodes, live_edges = runICmodel_n(self.G, S, self.TrueP)
                else:
                    t1 = time.time()
                    S = alg.decide()
                    t2 = time.time()
                    reward, live_nodes, live_edges = runICmodel_n(self.G, S, self.TrueP)
                    t3 = time.time()
                    alg.updateParameters(S, live_nodes, live_edges, i)
                    t = time.time() - t3 + t2 - t1

                self.round_reward[alg_name].append(reward) # 每一次iteration的reward
                self.cumul_reward[alg_name].append( sum(self.round_reward[alg_name]) )
                self.round_time[alg_name].append(t)
                self.loss[alg_name].append(alg.getLoss() if alg_name not in ['Oracle', 'Random'] else 0 )
                
                logging.info("{}: reward:{}, loss:{:.4f}".format(alg_name, reward, self.loss[alg_name][-1]))

            self.store_results(i, 5)
            logging.info('total time: {:.2f}'.format( (time.time() - start_time)) )

    def store_results(self, iter_=None, mod=5):
        '''store results: round_reward, cumul_reward, loss
        '''
        if not ((iter_+1) % mod == 0 or (iter_+1) == self.iterations ): # 每隔一定轮次
            return
        if not Path(self.result_dir).exists(): os.mkdir(Path(self.result_dir))
        
        results = {'round_reward': self.round_reward,
                    'cumul_reward': self.cumul_reward,
                    'round_time': self.round_time,
                    'loss': self.loss}

        self.result_file = file_name = '{}_{}_{}_z{}_L{}_{}_{}'.format(self.start_time_str, \
                                                                        self.dataset, \
                                                                        self.seed_size, \
                                                                        conf.z, \
                                                                        conf.kernel_size, \
                                                                        str(self.oracle.__name__), \
                                                                        '_'.join(self.algorithms.keys()))

        file_path = Path(self.result_dir) / Path(file_name + '.pk')
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-imgucb', action='store_true')
    parser.add_argument('-imfbmle', action='store_true')
    parser.add_argument('-imfb', action='store_true')
    parser.add_argument('-imlinucb', action='store_true')
    parser.add_argument('-egreedy', action='store_true')
    parser.add_argument('-cucb', action='store_true')
    parser.add_argument('-repeat', type=int, default=1)
    parser.add_argument('-resdir', type=str)
    parser.add_argument('-dataset', type=str, choices=['Flickr', 'NetHEPT', 'HEPPH', 'DBLP'], required=True)
    parser.add_argument('-nlin',  action='store_true') # linear or nonlinear prob
    args = parser.parse_args()

    start = time.time()
    if args.dataset == 'Flickr':
        dataset = Path(conf.Flickr_dataset)
    elif args.dataset == 'NetHEPT':
        dataset = Path(conf.NetHEPT_dataset)
    elif args.dataset == 'HEPPH':
        dataset = Path(conf.HEPPH_dataset)
    elif args.dataset == 'DBLP':
        dataset = Path(conf.DBLP_dataset)

    G = nx.read_gpickle(dataset)
    if args.nlin:
        trueP  = dict([ ((u,v), p) for u, v, p in G.edges(data='probability_nlin') ])
    else:
        trueP  = dict([ ((u,v), p) for u, v, p in G.edges(data='probability') ])
    
    node_vector = dict(G.nodes(data='feature'))
    edge_vector = dict([ ((u,v), p) for u, v, p in G.edges(data='feature') ])
    
    for i in range(args.repeat):
        print("REPEAT {}".format(i+1))
        algorithms = {}
        if args.imgucb:
            algorithms['IMGUCB'] = IMGaussianUCB.IMGaussianUCB(G, trueP, node_vector, conf.seed_size, conf.oracle, \
                                                                edge_dim=conf.edge_dimension, kernel_size=conf.kernel_size, sigma=conf.sigma, z=conf.z)
        if args.imfb:
            algorithms['IMFB'] = MFAlgorithm(G, trueP, node_vector, conf.seed_size, conf.oracle, conf.node_dimension)
        if args.imlinucb:
            algorithms['IMLinUCB'] = N_LinUCBAlgorithm(G, trueP, node_vector, conf.seed_size, conf.oracle, conf.node_dimension*conf.node_dimension, \
                                                        conf.alpha_1, conf.lambda_, edge_vector, 1) # 这个需要edge feature
        if args.egreedy:
            algorithms['EGreedy'] = eGreedyAlgorithm(G, trueP, conf.seed_size, conf.oracle, 0.1)
        if args.cucb:
            algorithms['CUCB'] = UCB1Algorithm(G, trueP, node_vector, conf.seed_size, conf.oracle)

        simExperiment = SimulateAlgos(G, trueP, conf.oracle, conf.seed_size, conf.iterations, args.dataset, algorithms, args.resdir)
        logging.info(simExperiment.start_time)
        simExperiment.runAlgorithms()