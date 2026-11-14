import os,signal,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
if sys.argv[2]=='short':time.sleep(.1)
else:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 while True:time.sleep(.02)
