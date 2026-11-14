import sys,json,tempfile,unittest,shutil,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'chimera'))
import routing_audit as m
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.r=Path(self.t.name);self.index=self.r/'index';self.index.mkdir();self.graph=self.index/'centroids.carga';self.graph.write_bytes(b'accepted graph fixture');self.receipt=self.r/'audit.json';self.a={'status':'PASS_POST_ENCODE_SAVE_DATASET','rows':8192,'dimension':128,'rotation_seed':42,'rotator':'FhtKacRotator','expected_rotated_dataset_sha256':'a'*64,'actual_saved_dataset_sha256':'a'*64,'float32_bit_mismatches':0,'dataset_memory_type':2,'final_cagra_file_sha256':m.sha(self.graph)};self.ident={'n_clusters':8192,'paddedD':128,'files':{'centroids.carga':{'sha256':m.sha(self.graph)}}};self.refresh()
 def tearDown(self):self.t.cleanup()
 def refresh(self):self.receipt.write_text(json.dumps(self.a));self.spec={'index':str(self.index),'routing_dataset_audit_receipt':str(self.receipt),'routing_dataset_audit_sha256':m.sha(self.receipt)}
 def test_pass(self):self.assertTrue(m.verify_routing_dataset(self.spec,self.ident)['verified'])
 def test_marker_always_rejects(self):
  (self.index/m.MARKER).write_text('{}')
  with self.assertRaisesRegex(ValueError,'quarantined'):m.verify_routing_dataset(self.spec,self.ident)
 def test_missing_receipt(self):
  with self.assertRaisesRegex(ValueError,'Require pinned'):m.verify_routing_dataset({'index':str(self.index)},self.ident)
 def test_receipt_mutation(self):
  self.receipt.write_text('{}')
  with self.assertRaisesRegex(ValueError,'receipt SHA'):m.verify_routing_dataset(self.spec,self.ident)
 def test_wrong_graph(self):
  self.a['final_cagra_file_sha256']='b'*64;self.refresh()
  with self.assertRaisesRegex(ValueError,'different centroids'):m.verify_routing_dataset(self.spec,self.ident)
 def test_dataset_mismatch(self):
  self.a['actual_saved_dataset_sha256']='b'*64;self.refresh()
  with self.assertRaisesRegex(ValueError,'differs'):m.verify_routing_dataset(self.spec,self.ident)
 def test_nonzero_mismatches(self):
  self.a['float32_bit_mismatches']=1;self.refresh()
  with self.assertRaisesRegex(ValueError,'differs'):m.verify_routing_dataset(self.spec,self.ident)
 def test_wrong_shape(self):
  self.a['rows']=8193;self.refresh()
  with self.assertRaisesRegex(ValueError,'shape'):m.verify_routing_dataset(self.spec,self.ident)
 def test_wrong_seed(self):
  self.a['rotation_seed']=43;self.refresh()
  with self.assertRaisesRegex(ValueError,'rotation'):m.verify_routing_dataset(self.spec,self.ident)
 def test_not_device_owned(self):
  self.a['dataset_memory_type']=0;self.refresh()
  with self.assertRaisesRegex(ValueError,'device-owned'):m.verify_routing_dataset(self.spec,self.ident)
 def make_derived(self):
  source=self.r/'fixed4';source.mkdir();os.link(self.graph,source/'centroids.carga');(self.index/'build-provenance.json').write_text(json.dumps({'fresh_8bit_build':False,'source4_index':str(source)}));return source
 def test_derived_same_graph_inherits(self):
  source=self.make_derived();self.assertEqual(m.verify_routing_dataset(self.spec,self.ident)['source4_inherited_from'],str(source))
 def test_derived_quarantined_source_rejected(self):
  source=self.make_derived();(source/m.MARKER).write_text('{}')
  with self.assertRaisesRegex(ValueError,'quarantined'):m.verify_routing_dataset(self.spec,self.ident)
 def test_derived_different_source_graph_rejected(self):
  source=self.make_derived();(source/'centroids.carga').unlink();(source/'centroids.carga').write_bytes(b'other graph')
  with self.assertRaisesRegex(ValueError,'source4 routing graph'):m.verify_routing_dataset(self.spec,self.ident)
 def test_grid_and_accept_cli_reject_before_inputs_or_gpu(self):
  (self.index/m.MARKER).write_text('{}');root=Path(m.__file__).parent;spec=self.r/'spec.json';spec.write_text(json.dumps({'index':str(self.index)}));run=self.r/'run';run.mkdir();(run/'prepared.json').write_text(json.dumps({'spec':{'index':str(self.index)}}))
  for cmd in [[sys.executable,str(root/'run_search_grid_v2.py'),'--spec',str(spec),'--output',str(self.r/'out')],[sys.executable,str(root/'accept_run.py'),str(run)]]:
   p=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('quarantined/unverified',p.stderr)
  self.assertFalse((self.r/'out').exists());self.assertFalse((run/'accepted-summary.json').exists())
if __name__=='__main__':unittest.main()
