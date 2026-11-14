"""CPU-only actual fullbit evaluation counts from existing NVTX INT64 payloads."""
import argparse,json,sqlite3
from pathlib import Path
Q='Chimera/query_local_row';W='Chimera/wait_chunk';F='Chimera/cpu_fullbit';C='Chimera/collaborative_scoring'
def need(x,msg):
 if not x:raise ValueError(msg)
def load_sqlite(path):
 db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 cols={r['name'] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')};need({'start','end','globalTid','int64Value'}<=cols,'NVTX export must retain start/end/globalTid/int64Value')
 strings={r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')} if 'textId' in cols else {};events=[]
 for row in db.execute('SELECT * FROM NVTX_EVENTS'):
  x=dict(row);name=x.get('text') or strings.get(x.get('textId'))
  if name in (Q,W,F,C):events.append(dict(name=name,start=x['start'],end=x['end'],tid=x['globalTid'],payload=x['int64Value']))
 db.close();return events
def parse(events,kfull=None):
 rows=[]
 for original in events:
  if original.get('name') not in (Q,W,F,C):continue
  x=dict(original);need(all(k in x for k in ('start','end','tid','payload')),'Incomplete NVTX event')
  need(isinstance(x['start'],int) and isinstance(x['end'],int) and x['end']>=x['start'],'Incomplete/invalid NVTX time range')
  need(isinstance(x['payload'],int) and not isinstance(x['payload'],bool) and x['payload']>=0,'Missing/noninteger INT64 payload')
  need(isinstance(x['tid'],int) and not isinstance(x['tid'],bool) and x['tid']>=0,'Missing/invalid global thread id')
  rows.append(x)
 queries=sorted((e for e in rows if e['name']==Q),key=lambda x:(str(x['tid']),x['start']));need(queries,'No query scopes captured');seen=set();result=[];owned=set()
 for i,q in enumerate(queries):
  key=(q['tid'],q['payload']);need(key not in seen,'Duplicate local query id on one thread');seen.add(key)
  same=[e for e in queries if e is not q and e['tid']==q['tid']];need(all(e['end']<=q['start'] or e['start']>=q['end'] for e in same),'Overlapping query scopes on one thread')
  inside=[(j,e) for j,e in enumerate(rows) if e['name']!=Q and e['tid']==q['tid'] and q['start']<=e['start']<=e['end']<=q['end']]
  owned.update(j for j,e in inside);waits=sorted((e for j,e in inside if e['name']==W),key=lambda x:x['start']);need([e['payload'] for e in waits]==list(range(len(waits))),'Chunk ids must be unique contiguous chronological0..C-1')
  counts=[0]*len(waits);elapsed=[0]*len(waits);assigned=set()
  for j,e in sorted(((j,e) for j,e in inside if e['name']==F),key=lambda z:z[1]['start']):
   need(e['payload']>0,'CPU fullbit scope unexpectedly has zero candidates');prior=[w for w in waits if w['end']<=e['start']];need(prior,'CPU fullbit event has no preceding chunk wait');w=prior[-1];c=w['payload'];need(c not in assigned,'Multiple CPU fullbit ranges for one chunk');assigned.add(c)
   nxt=waits[c+1]['start'] if c+1<len(waits) else q['end'];need(e['end']<=nxt,'CPU fullbit range crosses next chunk wait');counts[c]=e['payload'];elapsed[c]=e['end']-e['start']
  refs=[e['payload'] for j,e in inside if e['name']==C];need(len(refs)<=1,'Duplicate collaborative score scope');khat=sum(counts)
  if refs:need(khat<=refs[0],'Fullbit evaluated more unique candidates than refined pool')
  result.append(dict(thread=q['tid'],local_query_row=q['payload'],actual_chunks=len(waits),actual_fullbit_per_chunk=counts,Khat=khat,k_full_bit_config=kfull,refined_candidates=refs[0] if refs else None,cpu_fullbit_host_elapsed_ns_per_chunk=elapsed,query_host_elapsed_ns=q['end']-q['start']))
 need(all(j in owned for j,e in enumerate(rows) if e['name']!=Q),'Orphan chunk/fullbit/refinement event outside captured queries')
 return dict(queries=result,total_Khat=sum(q['Khat'] for q in result),note='Counts are actual cpu_fullbit INT64 payloads; zero-work chunks remain0. Host NVTX durations are not GPU-exclusive stage times.')
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--events-json');g.add_argument('--sqlite');p.add_argument('--kfull',type=int);p.add_argument('--out',required=True);a=p.parse_args();events=load_sqlite(a.sqlite) if a.sqlite else json.loads(Path(a.events_json).read_text());result=parse(events,a.kfull);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({'queries':len(result['queries']),'total_Khat':result['total_Khat']}))
if __name__=='__main__':main()
