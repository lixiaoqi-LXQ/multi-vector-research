import copy,json,sqlite3,tempfile
from pathlib import Path
from parse_khat import parse,load_sqlite,Q,W,F,C
r=Path(__file__).resolve().parent
rows=[dict(name=Q,start=0,end=1000,tid=7,payload=0),dict(name=C,start=1,end=999,tid=7,payload=100),dict(name=W,start=10,end=20,tid=7,payload=0),dict(name=F,start=25,end=30,tid=7,payload=4),dict(name=W,start=40,end=50,tid=7,payload=1),dict(name=W,start=60,end=70,tid=7,payload=2),dict(name=F,start=75,end=80,tid=7,payload=8)]
x=parse(rows,8);assert x['queries'][0]['actual_fullbit_per_chunk']==[4,0,8] and x['total_Khat']==12 # nominal8 does not cap actual12
mutations=[]
a=copy.deepcopy(rows);a[3]['payload']=None;mutations.append(a)
a=copy.deepcopy(rows);a[2]['payload']=2;mutations.append(a)
a=copy.deepcopy(rows);a[3]['start']=2;a[3]['end']=4;mutations.append(a)
a=copy.deepcopy(rows);a[3]['end']=45;mutations.append(a)
a=copy.deepcopy(rows);a.append(dict(a[3]));mutations.append(a)
a=copy.deepcopy(rows);a[3]['tid']=8;mutations.append(a)
a=copy.deepcopy(rows);a.append(dict(a[0]));mutations.append(a)
a=copy.deepcopy(rows);a[1]['payload']=10;mutations.append(a)
a=copy.deepcopy(rows);a[3]['tid']=None;mutations.append(a)
for a in mutations:
 try:parse(a,8)
 except ValueError:pass
 else:raise AssertionError('Invalid trace accepted')
with tempfile.TemporaryDirectory(dir=r) as tmp:
 p=Path(tmp)/'trace.sqlite';db=sqlite3.connect(p);db.execute('CREATE TABLE StringIds(id INTEGER,value TEXT)');db.execute('CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,int64Value INTEGER,text TEXT,textId INTEGER)')
 for e in rows:db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,NULL)',(e['start'],e['end'],e['tid'],e['payload'],e['name']))
 db.commit();db.close();assert parse(load_sqlite(p),8)['total_Khat']==12
(r/'khat-cpu-validation.json').write_text(json.dumps({'positive_json':True,'zero_work_chunk_preserved':True,'Khat_exceeds_nominal_supported':True,'positive_SQLite_readonly_adapter':True,'negative_cases_rejected':len(mutations),'real_NSYS_capture_verified':False},indent=2));print('PASS actual[4,0,8], Khat12 vsnominal8; SQLite adapter;9 negative traces')
