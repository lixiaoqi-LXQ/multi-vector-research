import copy,json,unittest
from pathlib import Path
import preflight
class Tests(unittest.TestCase):
 def setUp(self):self.s=json.loads(Path(__file__).with_name('spec.json').read_text())
 def test_actual_cpu_preflight(self):self.assertEqual(preflight.verify(self.s)['status'],'PASS_PREPARED_CPU_ONLY')
 def test_mem_guard(self):
  self.s['minimum_memavailable_bytes']=1<<80
  with self.assertRaisesRegex(ValueError,'MemAvailable'):preflight.verify_resources(self.s)
 def test_disk_guard(self):
  self.s['minimum_disk_free_bytes']=1<<80
  with self.assertRaisesRegex(ValueError,'Disk free'):preflight.verify_resources(self.s)
 def test_pid_reuse_cohort_guard(self):
  self.s['authorized_offline_cohort'][0]['start_ticks']+=1;self.assertFalse(preflight.cohort_live(self.s))
 def test_changed_command_cohort_guard(self):
  self.s['authorized_offline_cohort'][1]['cmdline_sha256']='0'*64;self.assertFalse(preflight.cohort_live(self.s))
 def test_original_binding_environment_and_controls(self):
  from pathlib import Path
  cache=Path(self.s['command'][self.s['command'].index('--cache')+1]);b=json.loads(Path(str(cache)+'.identity.json').read_text())['bindings']
  for key,value in b['environment'].items():self.assertEqual(self.s['environment'].get(key),value)
  self.assertEqual(self.s['environment'],json.loads(Path('/home/lxq/graduate/ai-workspace/research/multivector/chimera-ms-sorted-gather/q1-plan/spec.json').read_text())['environment'])
  self.assertIn('--resume',self.s['command']);self.assertNotIn('--stop-after-batches',self.s['command'])
 def test_changed_checkpoint_state_rejected(self):
  self.s['expected_checkpoint_state_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'checkpoint prefix changed'):preflight.verify(self.s)
 def test_changed_cache_stat_rejected(self):
  self.s['verified_cache_artifacts']['centroids.f32']['inode']+=1
  with self.assertRaisesRegex(ValueError,'cache artifact stat changed'):preflight.verify(self.s)
 def test_no_stop_flag_in_full_resume(self):
  self.s['command']+=['--stop-after-batches','1']
  with self.assertRaisesRegex(ValueError,'Whole resume'):preflight.verify(self.s)
if __name__=='__main__':unittest.main()
