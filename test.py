from __future__ import print_function

import numpy as np
import os
from Discrepancy import Solver, Supergraph

if not os.path.exists('MyMCFSolve'):
    print("run 'make' in MCFSimplex and then copy the executable to this directory as 'MyMCFSolve'")
    quit()

n_users, n_items = 5000, 1000
k_start = 100
k_finish = 10

scores = np.random.random((n_users, n_items))
candidate_recs = np.argsort(-scores, axis=1)[:,:k_start]
print(candidate_recs.shape)

# NOTE NOTE NOTE: you NEED to index from 1 or the code does not work!!!
with open('rec_table.txt', 'w') as f:
    lines = []
    for u in range(n_users):
        for i in range(k_start):
            item_idx = candidate_recs[u,i]
            lines.append('{} {} {}'.format(u+1, item_idx+1, scores[u, item_idx]))
    f.write('\n'.join(lines))

m_supergraph = Supergraph.SuperGraph.readRecommendationTable('rec_table.txt')
target = m_supergraph.convexCombinationA(10, alpha=.5)
constraints = m_supergraph.uniformC(10)

m_solver = Solver.Solver(m_supergraph)
m_solver.solveGreedy(constraints,
                     target,
                     'rec_table.txt')
GREEDY_edges = m_solver.edges
v1_max, v2_max = 0, 0
v1_min, v2_min = 99999999, 99999999
for e in GREEDY_edges:
    v1_max = max(v1_max, e[0])
    v2_max = max(v2_max, e[1])
    v1_min = min(v1_min, e[0])
    v2_min = min(v2_min, e[1])
    
print(v1_max, v2_max)
print(v1_min, v2_min)

def convert_edges_to_normal(edges):
    return [(e[0]-1, e[1]-n_users-1) for e in edges]

GREEDY_edges = convert_edges_to_normal(m_solver.edges)

#okay -- edges are encoded in a weird way.
#the first value is the 1-indexed user
#the second value is the 1-indexed item + N_users.
print(len(GREEDY_edges))
m_solver.solveWithGoal(constraints,
                       target,
                       'dmx.problem',
                       'rec_table.txt')
GOL_edges = convert_edges_to_normal(m_solver.edges)
print(len(GOL_edges))


