import os,unittest
from pathlib import Path
from unittest.mock import patch
from proc_observer import proc,bind_cmdline
class Tests(unittest.TestCase):
 def test_permission_io_preserves_live_stat(self):
  original=Path.read_text
  def read(path,*a,**kw):
   if str(path)==f'/proc/{os.getpid()}/io':raise PermissionError(13,'fixture proc/io transition')
   return original(path,*a,**kw)
  with patch.object(Path,'read_text',read):r=proc(os.getpid())
  self.assertIn('state',r);self.assertNotIn('terminal_or_gone',r);self.assertEqual(r['unavailable'][0]['exception'],'PermissionError')
 def test_missing_pid_is_unavailable_not_terminal(self):
  r=proc(2147483647);self.assertEqual(r['observation_status'],'partial_unavailable');self.assertNotIn('terminal_or_gone',r);self.assertNotIn('state',r)
 def test_task_directory_generic_oserror(self):
  with patch.object(Path,'iterdir',side_effect=OSError(5,'fixture exit transition')):r=proc(os.getpid())
  self.assertIn('state',r);self.assertEqual(r['thread_allowed_cpus'],{});self.assertTrue(r['unavailable'])
 def test_cmdline_permission_is_unknown(self):
  with patch.object(Path,'read_bytes',side_effect=PermissionError(13,'fixture')):r=bind_cmdline({'pid':os.getpid()},'a'*64)
  self.assertIsNone(r['cmdline_sha_matches']);self.assertEqual(r['unavailable'][0]['exception'],'PermissionError')
if __name__=='__main__':unittest.main()
