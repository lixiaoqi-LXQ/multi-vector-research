import ast,copy,json,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');sys.path.insert(0,str(R/'chimera'))
import run_search_grid_v2 as runner
from search_modes_v2 import modes,selection_family
class Tests(unittest.TestCase):
 def test_original_scientific_functions_unchanged(self):
  def defs(p):return {f.name:ast.dump(f,include_attributes=False) for f in ast.parse(p.read_text()).body if isinstance(f,ast.FunctionDef)}
  old=defs(R/'chimera/run_search_grid.py');new=defs(R/'chimera/run_search_grid_v2.py')
  for name in old:
   if name not in ('main','query_gpu'):self.assertEqual(old[name],new[name],name)
 def test_modes(self):
  for k in ('full','no-lut','no-aggregation','neither'):
   for a in ('full','no-cr','no-ivf-pg','neither'):
    s=dict(kernel_mode=k,algorithm_variant=a)
    if a in ('no-ivf-pg','neither'):s['expected_coarse_clusters']=8192
    self.assertEqual(modes(s)['algorithm_variant'],a)
  for s in ({'kernel_mode':'bad'},{'algorithm_variant':'bad'},{'algorithm_variant':'neither'},{'algorithm_variant':'no-ivf-pg','expected_coarse_clusters':True},{'expected_coarse_clusters':8192}):
   with self.assertRaises(ValueError):modes(s)
 def test_family_rejects_cross_modes_binary_index_gt_cal_resource(self):
  m=dict(spec={'phase':'calibration','query_rows':'cal.npy'},binary_sha256='a',index_identity={'N':20,'total_bits':4,'files':{'codes':{'sha256':'c'}}},k=100,query_payload_sha256='q',gt_ids_sha256='gt',GT_manifest={'corpus_source_id':'corpus'},GT_manifest_sha256='g',query_rows_manifest_sha256='split',CPU_affinity='0-19',OpenMP_environment={'OMP_NUM_THREADS':'20','OMP_PLACES':'all20'},load_mode='mmap')
  family=selection_family(m)
  mutations=[('binary_sha256','b'),('index_identity',{'N':20,'total_bits':8}),('k',10),('query_payload_sha256','other'),('gt_ids_sha256','other'),('GT_manifest',{'corpus_source_id':'other'}),('query_rows_manifest_sha256','other'),('CPU_affinity','7'),('OpenMP_environment',{'OMP_NUM_THREADS':'1'}),('load_mode','original')]
  for field,value in mutations:
   changed=copy.deepcopy(m);changed[field]=value;self.assertNotEqual(family,selection_family(changed),field)
  for field,value in [('kernel_mode','no-lut'),('algorithm_variant','no-cr'),('query_rows','other')]:
   changed=copy.deepcopy(m);changed['spec'][field]=value;self.assertNotEqual(family,selection_family(changed),field)
  changed=copy.deepcopy(m);changed['search_modes']={'kernel_mode':'no-lut'}
  with self.assertRaises(ValueError):selection_family(changed)
 def test_no_cr_refinement_parameter_semantics(self):
  from search_modes_v2 import validate_points
  for alg in ('no-cr','neither'):
   spec=dict(algorithm_variant=alg,points=[dict(nprobe=1,k_refine=1,k_full_bit=11,num_chunks=3)])
   if alg=='neither':spec['expected_coarse_clusters']=8192
   actual=validate_points(spec,512,11)[0];self.assertFalse(actual['k_refine_effective']);self.assertEqual(actual['actual_CG_documents_bound'],[0,11]);self.assertEqual(actual['num_refined_candidates'],'actual_CG_documents')
   for field,bad in [('k_refine',0),('k_full_bit',12),('nprobe',513),('num_chunks',0),('cagra_itopk_size',0)]:
    changed=copy.deepcopy(spec);changed['points'][0][field]=bad
    with self.assertRaises(ValueError):validate_points(changed,512,11)
  for alg in ('full','no-ivf-pg'):
   spec=dict(algorithm_variant=alg,points=[dict(nprobe=1,k_refine=1,k_full_bit=11,num_chunks=1)])
   if alg=='no-ivf-pg':spec['expected_coarse_clusters']=8192
   with self.assertRaisesRegex(ValueError,'CR-enabled'):validate_points(spec,512,11)
   spec['points'][0]['k_refine']=11;self.assertTrue(validate_points(spec,512,11)[0]['k_refine_effective'])
 def test_actual_mode_receipt(self):
  from search_modes_v2 import verify_mode_log
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
   f=Path(t)/'engine.log';spec=dict(kernel_mode='no-lut',algorithm_variant='full');f.write_text('[VARIANT] algorithm_variant=full kernel_mode=no-lut use_lut=0 use_aggregation=1\n');self.assertTrue(verify_mode_log(f,spec)['verified'])
   for text in ('',f.read_text()*2,'[VARIANT] algorithm_variant=full kernel_mode=full use_lut=1 use_aggregation=1\n','[VARIANT] algorithm_variant=full kernel_mode=no-lut use_lut=1 use_aggregation=1\n'):
    f.write_text(text)
    with self.assertRaises(ValueError):verify_mode_log(f,spec)
   f.write_text('');self.assertTrue(verify_mode_log(f,{})['verified_legacy_full'])
 def test_selector_actually_rejects_mixed_binary(self):
  import select_points_v2 as selector
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
   p=Path(t);folders=[]
   for i in range(2):
    f=p/str(i);f.mkdir();folders.append(str(f));meta=dict(spec={'phase':'calibration','rounds':1,'k':100},queries=100,expected_searches=1,binary_sha256=str(i),index_identity={'N':20,'total_bits':4},k=100,query_payload_sha256='q',gt_ids_sha256='g',GT_manifest={'corpus_source_id':'same'},CPU_affinity='0-19',OpenMP_environment={'OMP_NUM_THREADS':'20'},load_mode='mmap')
    (f/'prepared.json').write_text(json.dumps(meta));(f/'release.json').write_text(json.dumps(dict(complete=True,completed_searches=1)));(f/'summary.json').write_text(json.dumps([dict(queries=100,k=100,recall100=.5,qps=1,point={})]))
   with patch.object(sys,'argv',['select','--runs',*folders,'--out',str(p/'selected')]):
    with self.assertRaisesRegex(ValueError,'Cannot mix'):selector.main()
   self.assertFalse((p/'selected/selection.json').exists())
 def test_plan_commands_bind_explicit_modes(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
   p=Path(t);index=p/'index';index.mkdir();(index/'doclens.bin').write_bytes(struct.pack('<ii',1,20));query=p/'queries.bin';query.write_bytes(struct.pack('<iii',1,32,128));out=p/'plan';spec=p/'spec.json'
   spec.write_text(json.dumps(dict(index=str(index),query=str(query),gt='gt.bin',gt_ids='ids.npy',binary='/bin/true',threads=1,cpu_affinity='7',kernel_mode='full',algorithm_variant='no-ivf-pg',expected_coarse_clusters=8192,points=[dict(nprobe=1,k_refine=1,k_full_bit=1,num_chunks=1)])))
   idx=dict(N=20,D=128,n_clusters=8192,total_bits=4,files={});identity=dict(k=100,query_payload_sha256='q',gt_ids_sha256='g',GT_manifest={'corpus_source_id':'fixture'})
   with patch.object(sys,'argv',['runner','--spec',str(spec),'--output',str(out)]),patch.object(runner,'index_identity',return_value=idx),patch.object(runner,'identity_checks',return_value=identity):runner.main()
   meta=json.loads((out/'prepared.json').read_text());cmd=meta['commands'][0]
   for flag,value in [('--kernel-mode','full'),('--algorithm-variant','no-ivf-pg'),('--ablation-coarse-clusters','8192')]:self.assertEqual(cmd[cmd.index(flag)+1],value)
   self.assertEqual(meta['selection_family']['search_modes'],modes(meta['spec']));self.assertEqual(meta['exec_guard_sha256'],runner.sha(out/'frozen-exec-guard'))
 def test_existing_output_never_overwrites_prepared(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
   p=Path(t);index=p/'index';index.mkdir();(index/'doclens.bin').write_bytes(struct.pack('<ii',1,1));query=p/'queries.bin';query.write_bytes(struct.pack('<iii',1,32,128));out=p/'existing';out.mkdir();sentinel=out/'prepared.json';sentinel.write_text('preserve original')
   spec=p/'spec.json';spec.write_text(json.dumps(dict(index=str(index),query=str(query),points=[dict(nprobe=1,k_refine=1,k_full_bit=1,num_chunks=1)])))
   with patch.object(sys,'argv',['runner','--spec',str(spec),'--output',str(out)]):
    with self.assertRaises(FileExistsError):runner.main()
   self.assertEqual(sentinel.read_text(),'preserve original')
if __name__=='__main__':unittest.main()
