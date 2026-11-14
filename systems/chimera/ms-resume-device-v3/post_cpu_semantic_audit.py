"""Authorized saved-centroid audit inside inherited build lease, after full CPU gate."""
from pathlib import Path
import os,sys,json,hashlib,signal,time
P=Path(__file__).resolve().parent;R=P.parent;B=R.parent;sys.path.insert(0,str(R));from search_supervision_v2 import supervise_child

def run():
 s=json.loads((P/'spec.json').read_text());cpu=json.loads((P/'full-CPU-acceptance.json').read_text());assert cpu['status']=='PASS_FULL_MS_LABELS_PAYLOAD_IDENTITIES_AND_IVF';assert cpu['tokens']==597909919 and cpu['clusters']==4194304
 cmd=s['command'];index=Path(cmd[cmd.index('--index')+1]);cache=Path(cmd[cmd.index('--cache')+1]);base=Path(s['base']);out=base/'full-saved-centroid-audit';control=base/'centroid-audit-control';control.mkdir(exist_ok=False)
 def sha(p):
  with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 def st(p):
  q=p.stat();return {'device':q.st_dev,'inode':q.st_ino,'size':q.st_size,'mtime_ns':q.st_mtime_ns}
 def write(p,x):Path(p).write_text(json.dumps(x,indent=2))
 helper=B/'chimera-checkpoint/full-dataset-audit/build/chimera_saved_dataset_audit';freeze=B/'chimera-checkpoint/full-dataset-audit/freeze.json';assert sha(helper)=='e600ed6595372afd92c9a0695f3a4e5ada1d2142835d3147b6056b35f4445e77';assert sha(freeze)=='95db590589bcfe0facb73530c59eaa5f8cd44845e7d80bf0f8b3138c237c422c'
 matrix=cache/'centroids.f32';graph=index/'centroids.carga';before=[st(matrix),st(graph)];assert before[0]['size']==2147483648;assert before[1]=={k:cpu['payload_identities']['centroids.carga'][k] for k in before[1]}
 # Reuse the descriptor actually inherited through exec_guard. No new flock/open.
 fds=[]
 for p in Path('/proc/self/fd').iterdir():
  try:
   if os.readlink(p)=='/home/lxq/device.lock':fds.append(int(p.name))
  except FileNotFoundError:pass
 assert fds,'No inherited persistent device.lock FD';lease=os.fdopen(os.dup(fds[0]),'a+');env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS='4',OMP_PLACES='{15},{16},{17},{18}',OMP_PROC_BIND='true',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1');env.pop('GOMP_CPU_AFFINITY',None)
 command=['taskset','-c','15-18',str(helper),str(matrix),str(graph),str(out)];write(control/'prepared.json',{'command':command,'inherited_device_lock_FD':fds[0],'helper_sha256':sha(helper),'helper_freeze_sha256':sha(freeze),'input_stat':before,'full_CPU_acceptance_sha256':sha(P/'full-CPU-acceptance.json'),'searches':0,'timeout_s':175,'grace_s':15})
 def alarm(sig,frame):raise TimeoutError('Saved centroid audit175s')
 old=signal.signal(signal.SIGALRM,alarm);signal.alarm(175)
 try:
  with (control/'stdout.log').open('x') as log:
   def started(c):
    raw=Path(f'/proc/{c.pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();write(control/'process.json',{'pid':c.pid,'start_ticks':int(v[19]),'command':command})
   core=Path(cmd[cmd.index('--binary')+1]).parent;q=supervise_child(command,env=env,log=log,output_lock=lease,exec_guard=core/'exec_guard',receipt_path=control/'supervision.json',write_receipt=write,on_started=started,terminate_grace_s=15)
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);lease.close() # No LOCK_UN; outer owner retains lease.
 assert q['returncode']==0 and q['owned_child_terminal_and_reaped'];a=json.loads((out/'result.json').read_text());assert a['status']=='PASS_POST_ENCODE_SAVE_DATASET' and a['rows']==4194304 and a['dimension']==128 and a['float32_bit_mismatches']==0 and a['expected_rotated_dataset_sha256']==a['actual_saved_dataset_sha256'] and a['dataset_memory_type']==2;assert a['final_cagra_file_sha256']==cpu['payload_identities']['centroids.carga']['sha256'];assert before==[st(matrix),st(graph)];write(control/'complete.json',{'passed':True,'native_terminal_reaped':True,'no_search_or_fullindex_load':True,'root_metadata_published':False,'result_sha256':sha(out/'result.json')});print('PASS MS saved4Mcentroid semantic audit in inherited lease',flush=True)
