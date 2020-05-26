''' Implementation of degree discount heuristic [1] for Independent Cascade model
of influence propagation in graph G

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
'''
__author__ = 'ivanovsergey'
from Tool.priorityQueue import PriorityQueue as PQ # priority queue

def degreeDiscountIC(G, k, P=0):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                # print(t[v])
                p = P[u][v]['weight']
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                dd.add_task(v, -priority)
    return S

def degreeDiscountIAC(G, k, P):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    P -- propagation probability graph
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        # d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        d[u] = G.out_degree(u)
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                # t[v] += G[u][v]['weight'] # increase number of selected neighbors 
                t[v] += 1
                p = P[(u,v)]
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                dd.add_task(v, -priority)
    return S

def degreeDiscountIAC2(G, k, P):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    P -- propagation probability graph
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight']*P[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors 
                p = P[u][v]['weight']
                priority = d[v] - 2*t[v]*p - (d[v] - t[v]*p)*t[v]*p # discount of degree
                dd.add_task(v, -priority)
    return S

def degreeDiscountIC2(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (without priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    Note: the routine runs twice slower than using PQ. Implemented to verify results
    '''
    d = dict()
    dd = dict() # degree discount
    t = dict() # number of selected neighbors
    S = [] # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(iter(dd.items()), key=lambda k_v: k_v[1])
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
    return S

def degreeDiscountIAC3(G, k, P): # config中默认配置这个为oracle
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    P -- propagation probability graph, 传播概率图
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        # d[u] = sum([G[u][v]['weight']*P[u][v]['weight'] for v in G[u]]) + 1 # each edge adds degree 1
        # d[u] = sum([1*P[u][v]['weight'] for v in G[u]]) + 1
        d[u] = sum([1*P[(u, v)] for v in G[u]]) + 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node. degree最大，则-d[u]最小，则排在heap的前面(默认小根堆min-heap)
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                multi = 1
                add = 0
                for n in G.predecessors(v): # 前缀结点(有边指向v的结点)
                    if n in S: # 这些结点在S中，但是并没有激活v，的概率
                        # multi *= (1 - P[n][v]['weight'])
                        multi *= (1 - P[(n, v)])
                for n in G[v]:
                    if n in S:
                        continue
                    else:
                        # add += P[v][n]['weight']
                        add += P[(v, n)]
                add += 1
                priority = add * multi
                dd.add_task(v, -priority)
    return S
    
def degreeDiscountStar(G,k,P):
    
    S = []
    scores = PQ()
    d = dict()
    t = dict()
    for u in G:
        d[u] = sum([G[u][v]['weight']*P[u][v]['weight'] for v in G[u]])
        t[u] = 0
        score = -((1-P[u][v]['weight'])**t[u])*(1+(d[u]-t[u])*P[u][v]['weight'])
        scores.add_task(u, score)
    for iteration in range(k):
        u, priority = scores.pop_item()
        print(iteration, -priority)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']
                score = -((1-p)**t[u])*(1+(d[u]-t[u])*P[u][v]['weight'])
                scores.add_task(v, score)
    return S
            

if __name__ == '__main__':
    console = []