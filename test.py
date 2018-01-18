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

with open('rec_table.txt', 'w') as f:
    lines = []
    for u in range(n_users):
        for i in range(k_start):
            item_idx = candidate_recs[u,i]
            lines.append('{} {} {}'.format(u, item_idx, scores[u, item_idx]))
    f.write('\n'.join(lines))

m_supergraph = Supergraph.SuperGraph.readRecommendationTable('rec_table.txt')
target = m_supergraph.uniformA(10)

m_solver = Solver.Solver(m_supergraph)
const = [10 for _ in range(n_users)]
const[0] = 0
res = m_solver.solve(const,
                     target,
                     'dmx.problem',
                     'rec_table.txt')
print(len(m_solver.edges))
quit()
