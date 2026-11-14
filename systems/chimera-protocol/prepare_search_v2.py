from pathlib import Path
import shutil
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');p=R/'chimera';c=R/'chimera-ms-sorted-gather/q1-plan/controls'
s=(c/'supervision.py').read_text().replace('write_receipt, on_started, terminate_grace_s=5.0):','write_receipt, on_started, terminate_grace_s=5.0, on_poll=None):').replace('while child.poll() is None and requested is None:\n                try:','while child.poll() is None and requested is None:\n                if on_poll is not None:\n                    on_poll(child)\n                try:')
(p/'search_supervision_v2.py').write_text(s);shutil.copy2(c/'exec_guard',p/'search_exec_guard_v2')
s=(p/'run_search_grid.py').read_text();s=s.replace('from routing_audit import','from search_supervision_v2 import supervise_child\nfrom search_modes_v2 import modes,selection_family\nfrom routing_audit import',1).replace('out.mkdir(parents=True,exist_ok=True)','out.mkdir(parents=True,exist_ok=False)')
s=s.replace("spec=json.loads(Path(args.spec).read_text());reject_quarantined", "spec=json.loads(Path(args.spec).read_text());mode=modes(spec);reject_quarantined")
s=s.replace("base+=['--thread-info']", "base+=['--thread-info']\n if 'kernel_mode' in spec:base+=['--kernel-mode',mode['kernel_mode']]\n if 'algorithm_variant' in spec:base+=['--algorithm-variant',mode['algorithm_variant']]\n if mode['expected_coarse_clusters'] is not None:\n  need(mode['expected_coarse_clusters']==index_meta['n_clusters'],'Declared coarse clusters differ from index')\n  base+=['--ablation-coarse-clusters',str(mode['expected_coarse_clusters'])]")
s=s.replace("(out/'prepared.json').write_text", "prepared['search_modes']=mode;prepared['selection_family']=selection_family(prepared)\n (out/'prepared.json').write_text",1)
a=s.index("   with open(out/(name+'.log'),'w') as f:");b=s.index("   need(p.returncode==0",a)
s=s[:a]+'''   with open(out/(name+'.log'),'x') as f,open(out/(name+'-monitor.jsonl'),'x') as mon:
    last_poll=[float('-inf')]
    def monitor(child):
     if time.monotonic()-start>spec.get('timeout_per_point_s',7200):raise TimeoutError(name)
     if time.monotonic()-last_poll[0]<1:return
     last_poll[0]=time.monotonic();util,apps=query_gpu();foreign=[]
     for row in apps.stdout.strip().splitlines():
      try:other=int(row.split(',',1)[0])
      except ValueError:continue
      if other!=child.pid:foreign.append(row)
     mon.write(json.dumps(dict(time=time.time(),pid=child.pid,gpu=util.stdout.strip(),gpu_query_returncode=util.returncode,apps=apps.stdout.strip(),foreign_gpu_processes=foreign,foreign_note='Residency alone is not evidence of active compute; review actual interference',**memory_sample(child.pid)))+'\\n');mon.flush()
    def started(child):
     nonlocal p
     p=child
    def write_receipt(path,value):
     with open(path,'x') as receipt:json.dump(value,receipt,indent=2);receipt.flush();os.fsync(receipt.fileno())
    p=None
    supervision=supervise_child(cmd,env=env,log=f,output_lock=lock,exec_guard=ROOT/'search_exec_guard_v2',receipt_path=out/(name+'-supervision.json'),write_receipt=write_receipt,on_started=started,on_poll=monitor)
    need(supervision['requested_signal'] is None,'Search runner interrupted')
''' +s[b:]
s=s.replace("  exited=('p' not in locals()) or p.poll() is not None;", "  exited=('p' not in locals()) or p is None or p.poll() is not None;need(exited,'Cannot release global lease with owned child alive');")
(p/'run_search_grid_v2.py').write_text(s)
s=(p/'select_points.py').read_text().replace('from run_search_grid import need,sha,omp_budget','from run_search_grid_v2 import need,sha,omp_budget\nfrom search_modes_v2 import selection_family')
a=s.index('  key=(');b=s.index('\n  if family',a);s=s[:a]+"  key=selection_family(meta)"+s[b:];s=s.replace('k=family[3]','k=family[\'k\']');s=s.replace('Cannot mix bitwidth/K/corpus/query/GT families in selection','Cannot mix mode/binary/index/bitwidth/K/query/GT/calibration/resource families in selection');(p/'select_points_v2.py').write_text(s)
