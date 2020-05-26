import os
from Oracle.generalGreedy import generalGreedy
from Oracle.degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar, degreeDiscountIAC3

# save_address = "./SimulationResults"


Flickr_dataset = './datasets/data/Flickr/Flickr_NXPK'
NetHEPT_dataset = './datasets/data/NetHEPT/NetHEPT_NXPK'
HEPPH_dataset = None
DBLP_dataset = None
oracle = degreeDiscountIAC3
seed_size = 300
iterations = 300
node_dimension = 4 # 4
edge_dimension = node_dimension*2

##################### IMGUCB
sigma = 1e-4 # 加到kernel矩阵上的I的系数。不小心删除了conf之后，把这个设置为了1，巨大差别！因为这个代表了noise的scale，先验知识。
kernel_size = 2000 
z = 3 # 加几倍std作为上界
##################### IMFB

##################### LinUCB
alpha_1 = 0.1
lambda_ = 0.4
##################### Egreedy

##################### CUCB



# Flickr_data_dir = '/home/xiawenwen/datasets/Flickr'
# NetHEPT_data_dir = '/home/xiawenwen/datasets/NetHEPT'
# HEPPH_data_dir = '/home/xiawenwen/datasets/HEPPH'
# DBLP_dir = '/home/xiawenwen/datasets/DBLP'
# graph_address = 'Small_Final_SubG.G'
# node_feature_address = 'Small_nodeFeatures.dic'
# edge_feature_address = 'Small_edgeFeatures.dic'
# linear_prob_address = 'Probability.dic'
# nonlinear_prob_address = 'Probability_nonlinear.dic'
# oracle = degreeDiscountIAC3
# oracle = degreeDiscountIAC2
# oracle = degreeDiscountIAC

# alpha_2 = 0.1
# gamma = 0.1
# c = 1
# c = 3
# c = 0.2

