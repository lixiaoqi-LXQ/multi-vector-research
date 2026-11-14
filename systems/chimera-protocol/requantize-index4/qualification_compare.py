import hashlib,json,os,pathlib,shutil,struct,subprocess,time
R=pathlib.Path(__file__).resolve().parent;D=pathlib.Path('/home/lxq/graduate/data/multivector-colbertv2');I=D/'indexes/chimera'
src=I/'lotte-prefix-10000-c8192-b4-requant-qualification';fresh=I/'lotte-prefix-10000-c8192-b8-requant-qualification';derived=I/'lotte-prefix-10000-c8192-b8-requantized-qualification';raw=D/'chimera-inputs/lotte-prefix-10000/documents-prefix-qualification.npy'
binding=json.loads((R/'qualification-binding.json').read_text());files=list(binding['source_files_sha256'])
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def equal(a,b):
 assert a.stat().st_size==b.stat().st_size
 with open(a,'rb') as f,open(b,'rb') as g:
  offset=0
  while True:
   x=f.read(8<<20);y=g.read(8<<20);assert x==y,(str(a),str(b),offset)
   if not x:break
   offset+=len(x)
 return {'bytes':offset,'sha256':sha(a)}
result={'status':'PENDING','gpu_used':False,'fresh4':str(src),'fresh8':str(fresh),'derived8':str(derived),'raw_fp16':str(raw),'payload_comparison':{n:equal(fresh/n,derived/n) for n in ['doc_full.bin','doc_1bit.bin']}}
for name in files:assert sha(src/name)==binding['source_files_sha256'][name]
result['source4_all_six_payloads_preserved']=True
fixed=['doclens.bin','ivf.bin','cluster_1bit.bin','centroids.carga'];result['hardlinks']={}
for name in fixed:
 a=(src/name).stat();b=(derived/name).stat();assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino);result['hardlinks'][name]={'same_device_inode':True,'bytes':b.st_size,'sha256':sha(derived/name)}
# Verify source4 and derived8 non-fullfactor doc1bit sections, apart from ex_bits header field.
a=(src/'doc_1bit.bin').read_bytes();b=(derived/'doc_1bit.bin').read_bytes();h4=struct.unpack('<5Q',a[:40]);h8=struct.unpack('<5Q',b[:40]);assert h4[:3]==h8[:3] and h4[3]==3 and h8[3]==7 and h4[4]==h8[4]
assert a[40:len(a)-4*h4[0]]==b[40:len(b)-4*h8[0]];result['rotator_onebit_codes_onebit_factors_preserved']=True
T=R/'mutation-qualification';T.mkdir(exist_ok=True);badsrc=T/'mutated-source4';badsrc.mkdir(exist_ok=True)
for n in files:
 if n=='doc_1bit.bin':shutil.copy2(src/n,badsrc/n)
 else:os.link(src/n,badsrc/n)
# Serialized FHT-Kac rotator has4*128/8=64 sign bytes; byte104 is first onebit code.
with open(badsrc/'doc_1bit.bin','r+b') as f:f.seek(104);v=f.read(1);f.seek(104);f.write(bytes([v[0]^1]))
mutout=T/'source-corruption-output';mutout.mkdir();env=dict(os.environ,OMP_NUM_THREADS='20',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='-1')
p=subprocess.run([str(R/'requantize'),str(badsrc),str(raw),str(mutout)],env=env,capture_output=True,text=True)
assert p.returncode!=0 and 'invariant mismatch at row 0' in p.stderr,p.stderr
assert not any(mutout.iterdir());result['source_row_mutation_rejected']={'returncode':p.returncode,'stderr':p.stderr.strip(),'published_files':0}
# Missing/wrong raw SHA is rejected before destination creation or encoding.
badbinding=dict(binding,raw_npy_sha256='0'*64);(T/'bad-raw-binding.json').write_text(json.dumps(badbinding));badout=T/'wrong-raw-output'
p=subprocess.run(['/home/lxq/graduate/.venv/bin/python',str(R/'run_requantize.py'),'--source',str(src),'--raw-fp16-npy',str(raw),'--binding',str(T/'bad-raw-binding.json'),'--tool',str(R/'requantize'),'--output',str(badout)],env=env,capture_output=True,text=True)
assert p.returncode!=0 and 'Input SHA256 mismatch: raw_fp16_npy' in p.stderr and not badout.exists();result['raw_sha_mutation_rejected']=True
# An existing output must preserve every accepted payload.
p=subprocess.run(['/home/lxq/graduate/.venv/bin/python',str(R/'run_requantize.py'),'--source',str(src),'--raw-fp16-npy',str(raw),'--binding',str(R/'qualification-binding.json'),'--tool',str(R/'requantize'),'--output',str(derived)],env=env,capture_output=True,text=True)
assert p.returncode!=0 and 'FileExistsError' in p.stderr;result['existing_output_rejected']=True
for name in files:assert sha(src/name)==binding['source_files_sha256'][name]
for name in ['doc_full.bin','doc_1bit.bin']:assert sha(derived/name)==result['payload_comparison'][name]['sha256']
result['source_and_accepted_output_preserved_after_negative_tests']=True;result['binary_sha256']=sha(R/'requantize');result['fresh8_receipt_sha256']=sha(fresh/'build-provenance.json');result['fresh4_receipt_sha256']=sha(src/'build-provenance.json');result['derived_receipt_sha256']=sha(derived/'requantize-receipt.json');result['status']='PASS_CPU_BYTE_QUALIFICATION';result['gpu_engine_query_validation']='Delegated to Chimera owner; not run by this CPU child'
(R/'qualification-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
