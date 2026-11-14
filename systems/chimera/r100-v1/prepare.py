from pathlib import Path
import json
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');P=R/'chimera/r100-v1'
s=json.loads((R/'chimera/fine-completion-v1/r10-95-winner-chunks45-full3-v3/r10-0.95-winner-chunk4-full3.json').read_text())
for key in ['split_manifest','heldout_rows_source','calibration_queries','calibration_query','calibration_gt','calibration_gt_ids','calibration_rows','selection_frozen_from','runner_freeze_sha256','finalist_id','finalist_repeat','screening_source','finalizer_requirements','user_requested_chunks','ladder_note','staging_gate']:
 s.pop(key,None)
s.update(k=100,rounds=1,phase='screening',stage='R100 full-query screening',selection_policy='All2931 queries, no heldout. Screen bounded neighbors, freeze qualified fastest point, confirm full3; means determine formal quality.',quality_targets=[{'metric':'recall100','threshold':.9},{'metric':'recall100','threshold':.95}],quality_target={'metric':'recall100','threshold':.95},quality_evaluation_scope='full2931 Recall@100; no heldout',goals=[.9,.95],points=[dict(nprobe=128,k_refine=3000,k_full_bit=300,cagra_itopk_size=160,num_chunks=c) for c in (1,5)])
(P/'screen-default-neighbors.json').write_text(json.dumps(s,indent=2)+'\n')
state=R/'state.json';d=json.loads(state.read_text());d.update(status='R100_active_screening_LoTTE_CPU20',objective='继续LoTTE Recall@100 端到端比较，目标0.90/0.95，CPU20，无留出集',pending=['Chimera R100 screening then full3','PLAID R100 screening then full3','R100 Markdown comparison and bottleneck update']);state.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(P/'screen-default-neighbors.json')
