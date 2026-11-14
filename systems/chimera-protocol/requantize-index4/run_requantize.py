#!/usr/bin/env python3
"""Verified CPU requantization. No engine imports and no CUDA use in the child tool."""
import argparse,hashlib,json,os,pathlib,subprocess,time
FILES=['doclens.bin','ivf.bin','cluster_1bit.bin','centroids.carga','doc_1bit.bin','doc_full.bin']
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def snapshot(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw-fp16-npy',required=True);p.add_argument('--binding',required=True);p.add_argument('--tool',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 src=pathlib.Path(a.source).resolve(strict=True);raw=pathlib.Path(a.raw_fp16_npy).resolve(strict=True);bindingpath=pathlib.Path(a.binding).resolve(strict=True);tool=pathlib.Path(a.tool).resolve(strict=True);dest=pathlib.Path(a.output).absolute();binding=json.loads(bindingpath.read_text())
 if binding.get('schema')!=1 or binding.get('source_build_receipt_sha256') is None:raise ValueError('Binding schema1 and source-build receipt identity required')
 receipt=pathlib.Path(binding['source_build_receipt']).resolve(strict=True)
 if sha(receipt)!=binding['source_build_receipt_sha256']:raise ValueError('Source receipt identity mismatch')
 if not binding.get('raw_identity_confirmed_against_source_build'):raise ValueError('Explicit raw identity confirmation against source build required')
 expected=binding['source_files_sha256'];checks={};before={}
 for name,path,want in [(name,src/name,expected[name]) for name in FILES]+[('raw_fp16_npy',raw,binding['raw_npy_sha256'])]:
  before[name]=snapshot(path);actual=sha(path)
  if actual!=want:raise ValueError('Input SHA256 mismatch: '+name)
  checks[name]={'path':str(path),'sha256':actual,'stat':before[name]}
 if dest.exists():raise FileExistsError(dest)
 if not dest.parent.is_dir():raise ValueError('Output parent must exist')
 if src.stat().st_dev!=dest.parent.stat().st_dev:raise ValueError('Output must share filesystem with source index for hardlinks')
 stage=dest.parent/(dest.name+'.requantize-staging-'+str(os.getpid()));stage.mkdir(mode=0o700)
 start=time.perf_counter();completed=subprocess.run([str(tool),str(src),str(raw),str(stage)],text=True,capture_output=True)
 (stage/'tool.stdout').write_text(completed.stdout);(stage/'tool.stderr').write_text(completed.stderr)
 if completed.returncode:raise RuntimeError('Requantization failed; retained staging diagnostics at '+str(stage))
 result=json.loads(completed.stdout.strip().splitlines()[-1])
 for name,check in checks.items():
  if snapshot(pathlib.Path(check['path']))!=check['stat']:raise RuntimeError('Source identity changed during requantization: '+name)
 outputs={name:{'sha256':sha(stage/name),'bytes':(stage/name).stat().st_size} for name in FILES}
 receipt={'schema':1,'operation':'CPU requantize from fixed 4-bit index to 8-bit','fresh_8bit_build':False,'seconds_requantize_and_validate':time.perf_counter()-start,'binding_sha256':sha(bindingpath),'source_build_receipt_sha256':binding['source_build_receipt_sha256'],'inputs':checks,'tool_sha256':sha(tool),'wrapper_sha256':sha(__file__),'outputs':outputs,'validation':result}
 (stage/'requantize-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 # Atomic directory publication, with an exclusive reservation to prevent overwrites.
 # Linux renameat2(RENAME_NOREPLACE) is deliberately used instead of replace().
 import ctypes
 libc=ctypes.CDLL(None,use_errno=True);fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
 if fn(-100,os.fsencode(stage),-100,os.fsencode(dest),1)!=0:raise OSError(ctypes.get_errno(),'Exclusive directory publication failed; staging retained',str(dest))
 fd=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 print(json.dumps({'output':str(dest),'receipt':str(dest/'requantize-receipt.json')}))
if __name__=='__main__':main()
