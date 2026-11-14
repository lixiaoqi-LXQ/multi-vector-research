import os,signal,time,sys
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
def done(*a):
 time.sleep(.5)
 sys.exit(0)
signal.signal(signal.SIGTERM,done)
while True:time.sleep(.05)
