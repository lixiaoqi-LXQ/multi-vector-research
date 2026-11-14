"""Concurrent-query quality screening only; official B1 rank, no fused-query algorithm.
Legacy CUDA decompressor requires explicit stream handoff. Formal/finalist B1
must use the unchanged endpoint without this bridge or ThreadPoolExecutor.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os,threading,time
import torch

@contextmanager
def legacy_decompress_stream_bridge():
    from colbert.indexing.codecs.residual import ResidualCodec
    original=ResidualCodec.decompress_residuals
    enqueue_lock=threading.RLock()
    def bridged(*args):
        worker=torch.cuda.current_stream(0)
        default=torch.cuda.default_stream(0)
        if worker==default:return original(*args)
        with enqueue_lock:
            before=torch.cuda.Event();before.record(worker);default.wait_event(before)
            # The extension launches <<<blocks,threads>>> on the legacy default
            # stream, so its zero-output allocation must also be on that stream.
            with torch.cuda.stream(default):
                for value in args:
                    if torch.is_tensor(value) and value.is_cuda:value.record_stream(default)
                result=original(*args)
                after=torch.cuda.Event();after.record(default)
            worker.wait_event(after)
            result.record_stream(worker)
        return result
    # Rank calls this via the class, not via a codec instance.
    ResidualCodec.decompress_residuals=staticmethod(bridged)
    try:yield
    finally:
        try:torch.cuda.synchronize(0)
        finally:ResidualCodec.decompress_residuals=original

def payload_snapshot(scorer):
    pairs={'codes':scorer.embeddings.codes,'residuals':scorer.embeddings.residuals,'ivf':scorer.ivf.tensor,'doclens':scorer.doclens,'centroids':scorer.codec.centroids,'codes_offsets':scorer.embeddings_strided.codes_strided.offsets,'residual_offsets':scorer.embeddings_strided.residuals_strided.offsets}
    result={}
    for name,tensor in pairs.items():
        try:version=tensor._version
        except RuntimeError:version=None
        result[name]={'data_ptr':tensor.data_ptr(),'shape':list(tensor.shape),'device':str(tensor.device),'version':version}
    return result

def screen(scorer,queries_cpu,config_factory,concurrency=4,topk=10,cpu_budget=tuple(range(20))):
    assert concurrency in [2,4] and queries_cpu.dtype==torch.float32 and not queries_cpu.is_cuda
    before=payload_snapshot(scorer);tls=threading.local();ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(0));initialized=[];init_lock=threading.Lock();barrier=threading.Barrier(concurrency)
    def initialize():
        torch.cuda.set_device(0)
        if cpu_budget:os.sched_setaffinity(threading.get_native_id(),set(cpu_budget))
        tls.stream=torch.cuda.Stream(device=0);tls.stream.wait_event(ready);tls.config=config_factory()
        with init_lock:initialized.append({'tid':threading.get_native_id(),'stream':int(tls.stream.cuda_stream),'CPU_affinity':sorted(os.sched_getaffinity(0))})
    def one(item):
        row,q=item
        with torch.inference_mode(),torch.cuda.stream(tls.stream):
            pids,scores=scorer.rank(tls.config,q.unsqueeze(0).cuda().half())
            pids,scores=pids[:topk],scores[:topk]
            tls.stream.synchronize()
        return {'query_index':row,'ids':pids,'scores':scores}
    def warm(slot):
        barrier.wait(timeout=30)
        return one((slot,queries_cpu[slot%len(queries_cpu)]))
    torch.cuda.synchronize(0)
    with legacy_decompress_stream_bridge():
        with ThreadPoolExecutor(max_workers=concurrency,initializer=initialize) as pool:
            list(pool.map(warm,range(concurrency)))
            assert len(initialized)==concurrency
            started=time.perf_counter();rows=list(pool.map(one,enumerate(queries_cpu)))
            torch.cuda.synchronize(0);wall=time.perf_counter()-started
    after=payload_snapshot(scorer);assert before==after,'Persistent index payload metadata/version changed'
    return {'mode':'concurrent-query calibration, not fused batch','concurrency':concurrency,'ordered_results':rows,'wall_seconds':wall,'B1_latency_not_measured':True,'default_stream_event_bridge':True,'payload_immutable_snapshot':after,'workers':initialized}
