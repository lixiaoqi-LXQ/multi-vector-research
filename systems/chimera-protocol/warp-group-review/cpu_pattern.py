import json
from pathlib import Path
docs=['A','B','B','C','C','C'];scores=[1.,2.,8.,3.,9.,4.]
groups=[[j for j,d in enumerate(docs) if d==doc] for doc in docs]
iterations=[len(g)-1 for g in groups]
rounds=[{'iteration':i,'participating_lanes':[j for j,n in enumerate(iterations) if n>i], 'declared_active_mask':63} for i in range(max(iterations))]
assert [r['participating_lanes'] for r in rounds]==[[1,2,3,4,5],[3,4,5]]
values=scores.copy()
for i in range(max(iterations)):
 old=values.copy()
 for lane,group in enumerate(groups):
  peers=[j for j in group if j!=lane]
  if i<len(peers):values[lane]=max(old[lane],old[peers[i]])
expected=[max(scores[j] for j in g) for g in groups]
assert values==expected==[1.,8.,8.,9.,9.,9.]
Path(__file__).with_name('cpu-pattern.json').write_text(json.dumps({'docs':docs,'scores':scores,'loop_calls':iterations,'round_participation':rounds,'expected_group_max':expected,'group_mask_simulated_result':values},indent=2)+'\n')
print('CPU pattern PASS; unequal group sizes violate active-mask participation; per-group simultaneous max matches reference.')
