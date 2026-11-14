#!/usr/bin/env python3
import ctypes, fcntl, json, os, signal, stat, subprocess, sys, time
from pathlib import Path
from supervision import supervise_child
from run_checkpoint_build import exclusive_json, declared_derivation_identity, construction_input_provenance, identity, sha
ROOT=Path(__file__).resolve().parent

def wait_until(predicate, seconds=5):
    deadline=time.monotonic()+seconds
    while not predicate():
        if time.monotonic()>deadline:raise AssertionError('condition timed out')
        time.sleep(.01)

def locked(path):
    with open(path,'a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        return False

def child(mode,directory,fd):
    d=Path(directory)
    if mode in ('ignore','exception'):signal.signal(signal.SIGTERM,signal.SIG_IGN)
    inherited=os.fstat(int(fd)).st_ino==(d/'output.lock').stat().st_ino
    exclusive_json(d/'child-ready.json',dict(pid=os.getpid(),parent=os.getppid(),inherited_output_lock_fd=inherited))
    if mode=='exit3':return 3
    while True:time.sleep(1)

def parent(mode,directory):
    d=Path(directory)
    with open(d/'output.lock','a') as lock,open(d/'child.log','x') as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def started(p):
            exclusive_json(d/'process.json',dict(pid=p.pid))
            if mode=='exception':
                wait_until(lambda:(d/'child-ready.json').exists())
                exclusive_json(d/'exception-started.json',dict(pid=p.pid))
                raise RuntimeError('intentional post-spawn failure')
        try:
            result=supervise_child([sys.executable,str(__file__),'_child',mode,str(d),str(lock.fileno())],env=os.environ.copy(),log=log,output_lock=lock,exec_guard=ROOT/'exec_guard',receipt_path=d/'supervision.json',write_receipt=exclusive_json,on_started=started,terminate_grace_s=.35)
        except RuntimeError:return 17
        return 128+result['requested_signal'] if result['requested_signal'] else result['returncode']

def main():
    root=Path(sys.argv[1]);root.mkdir(exist_ok=False);cases=[]
    for name,sig,mode in [('sigterm',signal.SIGTERM,'normal'),('sigint',signal.SIGINT,'normal'),('escalation',signal.SIGTERM,'ignore')]:
        d=root/name;d.mkdir();p=subprocess.Popen([sys.executable,str(__file__),'_parent',mode,str(d)])
        wait_until(lambda:(d/'child-ready.json').exists());ready=json.loads((d/'child-ready.json').read_text());assert ready['inherited_output_lock_fd'];assert locked(d/'output.lock')
        p.send_signal(sig)
        if mode=='ignore':time.sleep(.05);assert locked(d/'output.lock'),'lock released while child ignores termination'
        assert p.wait(timeout=5)==128+sig
        record=json.loads((d/'supervision.json').read_text());assert record['status']=='interrupted' and record['requested_signal']==sig and record['owned_child_terminal_and_reaped'];assert record['termination_escalated']==(mode=='ignore')
        assert not Path(f"/proc/{ready['pid']}").exists();assert not locked(d/'output.lock');cases.append(name)
    d=root/'exception';d.mkdir();p=subprocess.Popen([sys.executable,str(__file__),'_parent','exception',str(d)]);wait_until(lambda:(d/'exception-started.json').exists());assert locked(d/'output.lock');assert p.wait(timeout=5)==17
    record=json.loads((d/'supervision.json').read_text());assert record['status']=='supervision_failed' and record['error']['type']=='RuntimeError' and record['owned_child_terminal_and_reaped'];assert not Path(f"/proc/{record['child_pid']}").exists();assert not locked(d/'output.lock');cases.append('post-spawn exception')
    d=root/'exit3';d.mkdir();p=subprocess.Popen([sys.executable,str(__file__),'_parent','exit3',str(d)]);assert p.wait(timeout=5)==3;record=json.loads((d/'supervision.json').read_text());assert record['returncode']==3 and record['owned_child_terminal_and_reaped'];assert not locked(d/'output.lock');cases.append('normal exit3 preserved')
    # Become a subreaper only in this isolated test process, so hard-parent-death child can be reaped.
    libc=ctypes.CDLL(None,use_errno=True);old=ctypes.c_int();assert libc.prctl(37,ctypes.byref(old),0,0,0)==0;assert libc.prctl(36,1,0,0,0)==0
    try:
        d=root/'hard-parent-death';d.mkdir();p=subprocess.Popen([sys.executable,str(__file__),'_parent','ignore',str(d)]);wait_until(lambda:(d/'child-ready.json').exists());ready=json.loads((d/'child-ready.json').read_text());assert ready['inherited_output_lock_fd'];p.kill();assert p.wait(timeout=5)==-signal.SIGKILL
        reaped=[]
        def reap():
            result=os.waitpid(ready['pid'],os.WNOHANG)
            if result[0]:reaped.append(result);return True
            return False
        wait_until(reap);assert os.waitstatus_to_exitcode(reaped[0][1])==-signal.SIGKILL;assert not locked(d/'output.lock');cases.append('PDEATHSIG hard-parent-death')
    finally:assert libc.prctl(36,old.value,0,0,0)==0
    # Observe actual fsync fd types/targets, not a mirrored publication implementation.
    events=[];original_fsync=os.fsync
    def traced(fd):
        events.append((stat.S_ISDIR(os.fstat(fd).st_mode),os.readlink(f'/proc/self/fd/{fd}')));return original_fsync(fd)
    try:
        os.fsync=traced;exclusive_json(root/'binding.json',{'value':1})
    finally:os.fsync=original_fsync
    assert events==[(False,str(root/'binding.json')),(True,str(root))],events;cases.append('file then correct parent-directory fsync')
    assert subprocess.run([str(ROOT/'exec_guard'),'2','/bin/true']).returncode==125;cases.append('exec guard parent-PID mismatch rejected')
    receipt=root/'derivation.json';exclusive_json(receipt,{'source':'fixture'});prov={'input_derivation_receipt':str(receipt),'input_derivation_receipt_sha256':sha(receipt)};assert declared_derivation_identity(prov)['sha256']==prov['input_derivation_receipt_sha256'];cases.append('declared derivation receipt verified')
    saved=receipt.stat();receipt.write_text('{"source":"mutation"}');os.utime(receipt,ns=(saved.st_atime_ns,saved.st_mtime_ns))
    try:declared_derivation_identity(prov);raise AssertionError('mutated receipt accepted')
    except ValueError:pass
    cases.append('declared derivation receipt content mutation rejected')
    try:declared_derivation_identity({'input_derivation_receipt':str(receipt)});raise AssertionError('incomplete pair accepted')
    except ValueError:pass
    assert declared_derivation_identity({}) is None;cases.append('missing derivation pair rejected / optional absent accepted')
    raw=root/'raw-fp16-fixture.npy';raw.write_bytes(b'CPU fixture header/data placeholder')
    doclens=root/'doclens-fixture.bin';doclens.write_bytes(b'CPU doclens placeholder')
    raw_id,doc_id=identity(raw),identity(doclens);source4=construction_input_provenance('fp16-npy',raw_id,doc_id)
    # Exact source4 metadata access used by the qualified derived-index adapter.
    assert Path(source4['vector_input']).resolve()==raw.resolve() and source4['input_mode']=='fp16_npy'
    original_raw=source4['input_files']['vector_input'];actual_raw=raw.stat()
    assert [actual_raw.st_dev,actual_raw.st_ino,actual_raw.st_size,actual_raw.st_mtime_ns]==[original_raw['device'],original_raw['inode'],original_raw['size'],original_raw['mtime_ns']]
    assert source4['input_files']['doclens.bin']==doc_id and original_raw['sha256']==raw_id['sha256']
    cases.append('qualified derived8 adapter source4 vector_input/raw-stat metadata compatibility')
    result={'status':'PASS_CPU_SUPERVISION_DURABILITY_DERIVATION_TESTS','GPU_used':False,'cases':cases,'count':len(cases)};exclusive_json(root/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    if sys.argv[1]=='_child':raise SystemExit(child(*sys.argv[2:]))
    if sys.argv[1]=='_parent':raise SystemExit(parent(*sys.argv[2:]))
    main()
