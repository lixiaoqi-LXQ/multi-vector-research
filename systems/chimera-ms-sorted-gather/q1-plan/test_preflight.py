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
 def test_gt_identity_exact(self):
  gt=self.s['existing_GT_identity'];self.assertEqual(gt,{k:self.s['authorized_offline_cohort'][2][k] for k in ('pid','start_ticks','cmdline_sha256')});self.assertEqual(gt['pid'],3255093)
if __name__=='__main__':unittest.main()
