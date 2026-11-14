from pathlib import Path
import ctypes,os,json,time
D=Path('/home/lxq/graduate/data/multivector-colbertv2');P=Path(__file__).resolve().parent;libc=ctypes.CDLL(None,use_errno=True);libc.mmap.restype=ctypes.c_void_p;libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long];libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t];page=os.sysconf('SC_PAGESIZE');chunk=256*1024*1024
files=list((D/'indexes').rglob('*'))+list((D/'chimera-build-cache').rglob('*'))+[D/'msmarco-v1-passage/documents.npy',D/'lotte-pooled-dev/documents.npy'];records=[];seen=set();started=time.time()
for p in files:
 if not p.is_file():continue
 s=p.stat()
 if s.st_size<4*1024*1024 or (s.st_dev,s.st_ino) in seen:continue
 seen.add((s.st_dev,s.st_ino));fd=os.open(p,os.O_RDONLY);addr=libc.mmap(None,s.st_size,1,1,fd,0);os.close(fd)
 if addr==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_errno(),str(p))
 resident=0
 try:
  for offset in range(0,s.st_size,chunk):
   count=min(chunk,s.st_size-offset);pages=(count+page-1)//page;v=(ctypes.c_ubyte*pages)();status=libc.mincore(addr+offset,count,v)
   if status:raise OSError(ctypes.get_errno(),str(p))
   resident+=sum(x&1 for x in v)
 finally:libc.munmap(addr,s.st_size)
 t=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns);records.append({'path':str(p),'size':s.st_size,'resident_cache_bytes':min(s.st_size,resident*page),'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
records.sort(key=lambda x:x['resident_cache_bytes'],reverse=True);x={'operation':'mincore only; no file content reads; no cache advice','elapsed_s':time.time()-started,'files':len(records),'total_resident_bytes_unique_inodes':sum(r['resident_cache_bytes'] for r in records),'records':records};(P/'mincore-before-recovery-v3.json').write_text(json.dumps(x,indent=2));print(json.dumps({k:v for k,v in x.items() if k!='records'}));print(json.dumps(records[:20],indent=2))
