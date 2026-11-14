import unittest,json,copy
from pathlib import Path
import preflight
class Tests(unittest.TestCase):
 def setUp(self):self.spec=json.loads(Path(__file__).with_name('spec.json').read_text())
 def test_actual_CPU_preflight(self):self.assertEqual(preflight.verify(self.spec)['status'],'PASS_PREPARED_CPU_ONLY')
 def test_actual_full_shape(self):self.assertEqual((self.spec['docs'],self.spec['tokens'],self.spec['n_clusters']),(2428854,266205513,131072));self.assertNotIn('--stop-after-batches',self.spec['command'])
 def test_memory_guard(self):
  self.spec['minimum_memavailable_bytes']=1<<80
  with self.assertRaisesRegex(ValueError,'MemAvailable'):preflight.verify_resources(self.spec)
 def test_disk_guard(self):
  self.spec['minimum_disk_free_bytes']=1<<80
  with self.assertRaisesRegex(ValueError,'Disk free'):preflight.verify_resources(self.spec)
 def test_PID_reuse_guard(self):self.spec['authorized_offline_cohort'][0]['start_ticks']+=1;self.assertFalse(preflight.cohort_live(self.spec))
 def test_command_identity_guard(self):self.spec['authorized_offline_cohort'][1]['cmdline_sha256']='0'*64;self.assertFalse(preflight.cohort_live(self.spec))
 def test_no_early_stop_allowed(self):
  self.spec['command']+=['--stop-after-batches','1']
  with self.assertRaisesRegex(ValueError,'stopped early'):preflight.verify(self.spec)
 def test_coarse_C_guard(self):
  self.spec['command'][self.spec['command'].index('--clusters')+1]='131073'
  with self.assertRaisesRegex(ValueError,'parameters'):preflight.verify(self.spec)
 def test_no_initial_resume(self):
  self.spec['command']+=['--resume']
  with self.assertRaisesRegex(ValueError,'stopped early'):preflight.verify(self.spec)
if __name__=='__main__':unittest.main()
