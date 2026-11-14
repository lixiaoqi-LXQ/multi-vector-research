import subprocess,sys
p=subprocess.Popen(sys.argv[1:],close_fds=True,start_new_session=True)
sys.exit(p.wait())
