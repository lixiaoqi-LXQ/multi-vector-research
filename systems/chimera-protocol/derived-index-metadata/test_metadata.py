"""Small synthetic identity fixtures; no quantization/GPU or real-index mutation."""
import copy,importlib.util,json,os,shutil,struct,tempfile,unittest
from pathlib import Path
import publish_metadata as m
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);r=Path(self.tmp.name);self.s=r/'source';self.d=r/'derived';self.s.mkdir();self.d.mkdir();self.raw=r/'raw.npy';self.raw.write_bytes(b'synthetic identity fixture')
  for n in m.FIXED:(self.s/n).write_bytes(n.encode())
  (self.s/'doclens.bin').write_bytes(struct.pack('<3i',2,1,1));(self.s/'doc_1bit.bin').write_bytes(struct.pack('<5Q',2,128,1,3,128)+bytes(112));(self.s/'doc_full.bin').write_bytes(bytes(128))
  for n in m.FIXED:os.link(self.s/n,self.d/n)
  (self.d/'doc_1bit.bin').write_bytes(struct.pack('<5Q',2,128,1,7,128)+bytes(112));(self.d/'doc_full.bin').write_bytes(bytes(256))
  self.sp=self.s/'build-provenance.json';sid={'repository':'owner/repo','revision':'a'*40,'documents_url':'https://huggingface.co/datasets/owner/repo/resolve/'+'a'*40+'/documents.npy'}
  self.sp.write_text(json.dumps({'complete':True,'docs':2,'tokens':2,'n_clusters':1,'ex_bits':3,'total_bits':4,'vector_input':str(self.raw),'input_files':{'vector_input':dict(zip(('device','inode','size','mtime_ns'),m.stat4(self.raw)))},'source_identity':sid,'index_identity':m.identity(self.s)}))
  self.q=r/'qualification.json';self.q.write_text(json.dumps({'status':'PASS_CPU_BYTE_QUALIFICATION','binary_sha256':'b'*64,**{k:True for k in ('source4_all_six_payloads_preserved','rotator_onebit_codes_onebit_factors_preserved','raw_sha_mutation_rejected','existing_output_rejected','source_and_accepted_output_preserved_after_negative_tests')}}))
  self.rp=self.d/'requantize-receipt.json';self.r={'schema':1,'operation':m.OP,'fresh_8bit_build':False,'source_build_receipt_sha256':m.sha(self.sp),'tool_sha256':'b'*64,'inputs':{n:{'path':str(self.s/n),'sha256':m.sha(self.s/n),'stat':m.stat4(self.s/n)} for n in m.FILES},'outputs':{n:{'sha256':m.sha(self.d/n),'bytes':(self.d/n).stat().st_size} for n in m.FILES},'validation':{'all_rows_bitwise_source_verified':True,'gpu_used':False,'tokens':2,'dimension':128,'source_bits':4,'destination_bits':8},'seconds_requantize_and_validate':1.0};self.r['inputs']['raw_fp16_npy']={'path':str(self.raw),'sha256':m.sha(self.raw),'stat':m.stat4(self.raw)};self.rp.write_text(json.dumps(self.r))
 def tearDown(self):self.tmp.cleanup()
 def prepare(self):return m.prepare(self.d,self.sp,m.sha(self.rp),self.q,m.sha(self.q))
 def test_valid_and_publish_preserves_payload_receipts(self):
  before={str(p):m.sha(p) for p in [*[self.s/n for n in m.FILES],*[self.d/n for n in m.FILES],self.rp,self.sp,self.raw]};v=self.prepare();self.assertFalse((self.d/'build-provenance.json').exists());self.assertEqual((v['docs'],v['tokens'],v['dimension'],v['ex_bits'],v['total_bits']),(2,2,128,7,8));self.assertFalse(v['fresh_8bit_build']);m.publish(self.d,v)
  for p,h in before.items():self.assertEqual(m.sha(p),h)
  with self.assertRaisesRegex(ValueError,'existing'):m.publish(self.d,v)
 def test_existing_manifest(self):
  (self.d/'build-provenance.json').write_text('preserve')
  with self.assertRaisesRegex(ValueError,'existing'):self.prepare()
 def test_bad_requant_receipt_sha(self):
  with self.assertRaisesRegex(ValueError,'receipt SHA'):m.prepare(self.d,self.sp,'0'*64,self.q,m.sha(self.q))
 def test_source_payload_changed(self):
  (self.s/'doc_full.bin').write_bytes(b'x'*128)
  with self.assertRaisesRegex(ValueError,'Source4 payload SHA'):self.prepare()
 def test_raw_changed(self):
  self.raw.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'Raw'):self.prepare()
 def test_derived_payload_changed(self):
  (self.d/'doc_full.bin').write_bytes(b'x'*256)
  with self.assertRaisesRegex(ValueError,'Derived payload SHA'):self.prepare()
 def test_disallowed_tool(self):
  q=json.loads(self.q.read_text());q['binary_sha256']='c'*64;self.q.write_text(json.dumps(q))
  with self.assertRaisesRegex(ValueError,'qualified binary'):self.prepare()
 def test_reused_file_must_be_hardlink(self):
  p=self.d/'ivf.bin';p.unlink();shutil.copy2(self.s/'ivf.bin',p)
  with self.assertRaisesRegex(ValueError,'hardlink'):self.prepare()
 def test_publication_rechecks_payload_stat(self):
  v=self.prepare();(self.d/'doc_full.bin').write_bytes(b'x'*256)
  with self.assertRaisesRegex(ValueError,'changed before'):m.publish(self.d,v)
class RunnerTests(Tests):
 def runner(self):
  path=Path(__file__).with_name('run_search_grid.derived.py');sp=importlib.util.spec_from_file_location('derived_grid_fixture',path);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);return module
 def test_runner_explicit_approved_derived(self):
  v=self.prepare();spec={'allow_derived_index':True,'approved_requantize_tool_sha256':v['requantize_tool_sha256'],'approved_requantize_qualification_sha256':v['qualification_receipt_sha256']};self.runner().check_derived_build_provenance(spec,v,v['index_identity'])
 def test_runner_rejects_implicit_derived(self):
  v=self.prepare()
  with self.assertRaisesRegex(ValueError,'explicit allow'):self.runner().check_derived_build_provenance({},v,v['index_identity'])
 def test_runner_rejects_unapproved_tool(self):
  v=self.prepare();spec={'allow_derived_index':True,'approved_requantize_tool_sha256':'0'*64,'approved_requantize_qualification_sha256':v['qualification_receipt_sha256']}
  with self.assertRaisesRegex(ValueError,'Unapproved requantize'):self.runner().check_derived_build_provenance(spec,v,v['index_identity'])
 def test_runner_fresh_build_unchanged(self):self.runner().check_derived_build_provenance({}, {'complete':True}, {})
if __name__=='__main__':unittest.main()
