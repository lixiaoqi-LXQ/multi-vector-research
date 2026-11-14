"""Exact-owned-child supervision. No CUDA APIs, process-group killing or hidden work."""
import os
import signal
import subprocess
import time


def supervise_child(command, *, env, log, output_lock, device_lock, exec_guard, receipt_path,
                    write_receipt, on_started, terminate_grace_s=5.0, on_poll=None):
    """Hold caller's output lock until child terminal/reaped, including failures.

    The lock fd is also inherited across exec, protecting hard-parent-death teardown.
    exec_guard installs Linux PDEATHSIG(SIGKILL) and verifies the expected parent PID.
    Graceful SIGTERM/SIGINT are recorded, forwarded as exact-child SIGTERM, then
    escalated after grace. Handlers never raise during Popen's spawn window.
    """
    requested = None
    child = None
    error = None
    escalated = False
    previous = {}

    def handler(signum, frame):
        nonlocal requested
        if requested is None:
            requested = signum

    def terminate_and_reap():
        nonlocal escalated
        if child is None:
            return
        if child.poll() is None:
            child.terminate()  # Popen owns this exact unreaped PID.
            try:
                child.wait(timeout=terminate_grace_s)
            except subprocess.TimeoutExpired:
                escalated = True
                child.kill()
        # Never release the caller's lock while the owned child is alive.
        child.wait()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, handler)
        if requested is None:
            child = subprocess.Popen(
                [str(exec_guard), str(os.getpid()), *command], env=env,
                stdout=log, stderr=subprocess.STDOUT,
                pass_fds=tuple(dict.fromkeys((output_lock.fileno(),device_lock.fileno()))))
            on_started(child)
            while child.poll() is None and requested is None:
                if on_poll is not None:
                    on_poll(child)
                try:
                    child.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        if requested is not None:
            terminate_and_reap()
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            # Includes callback, receipt-writing and asynchronous Python exceptions.
            if child is not None:
                terminate_and_reap()
            record = {
                'status': 'supervision_failed' if error else
                          ('interrupted' if requested is not None else 'process_exited'),
                'child_pid': None if child is None else child.pid,
                'returncode': None if child is None else child.returncode,
                'requested_signal': requested,
                'termination_escalated': escalated,
                'owned_child_terminal_and_reaped': child is None or child.returncode is not None,
                'output_lock_inherited_by_child': child is not None,
                'device_lock_inherited_by_child': child is not None,
                'device_lock_path': '/home/lxq/device.lock',
                'device_lock_held_until_owned_child_terminal': child is None or child.returncode is not None,
                'linux_parent_death_signal': 'SIGKILL via checked exec_guard',
                'error': error,
                'finished_at_unix': time.time(),
            }
            write_receipt(receipt_path, record)
        finally:
            for sig, old in previous.items():
                signal.signal(sig, old)
    return record
