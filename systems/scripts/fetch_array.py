"""Resume a pinned large HF array with bounded parallel ranges and cooperative pause."""
import argparse,concurrent.futures,urllib.request,pathlib,time,os,json,fcntl
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--pause-file',required=True);p.add_argument('--lock-file',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--proxy',default='http://127.0.0.1:7890');a=p.parse_args()
root=pathlib.Path(a.data);meta=json.loads((root/'download-provenance.json').read_text());repo=meta['repository'];revision=meta['revision'];url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/documents.npy'
info=meta['files']['documents.npy'];total=info['bytes'];assert info['url']==url and len(revision)==40
lock=open(a.lock_file,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
final=root/'documents.npy';partial=root/'documents.npy.partial';pause=pathlib.Path(a.pause_file);chunk=32*1024*1024
if final.exists():
 assert final.stat().st_size==total
 print('ALREADY_COMPLETE');raise SystemExit

def wait_pause():
 while pause.exists():time.sleep(1)

def fetch(start):
 end=min(total,start+chunk)-1
 for attempt in range(10):
  wait_pause()
  try:
   op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':a.proxy}))
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with op.open(req,timeout=90) as r:
    expected=f'bytes {start}-{end}/{total}'
    if r.status!=206 or r.headers.get('Content-Range')!=expected:raise RuntimeError((r.status,r.headers.get('Content-Range'),expected))
    data=r.read()
    if len(data)!=end-start+1:raise RuntimeError('Short transfer')
    return data
  except Exception as exc:
   print(json.dumps({'retry':attempt,'start':start,'error':str(exc)}),flush=True)
   if attempt==9:raise
   time.sleep(min(30,2**attempt))
start=partial.stat().st_size if partial.exists() else 0;assert start<=total
print(json.dumps({'resume':start,'total':total,'workers':a.workers,'chunk_bytes':chunk,'url':url}),flush=True);started=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool,open(partial,'ab',buffering=0) as out:
 pending={};nextstart=start
 for _ in range(a.workers):
  if nextstart<total:pending[nextstart]=pool.submit(fetch,nextstart);nextstart+=chunk
 while pending:
  lo=min(pending);data=pending.pop(lo).result();wait_pause();assert out.tell()==lo
  assert out.write(data)==len(data)
  if nextstart<total:pending[nextstart]=pool.submit(fetch,nextstart);nextstart+=chunk
  done=out.tell();print(json.dumps({'downloaded':done,'total':total,'elapsed_s':time.time()-started,'MiB_s':(done-start)/1048576/(time.time()-started)}),flush=True)
assert partial.stat().st_size==total
os.replace(partial,final);print('DOWNLOAD_COMPLETE',flush=True)
