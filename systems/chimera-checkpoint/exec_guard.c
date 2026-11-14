#define _GNU_SOURCE
#include <sys/prctl.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
// No CUDA calls. PID is unchanged by execvp; the supervisor retains exact ownership.
int main(int argc, char **argv) {
    if (argc < 3) return 124;
    char *end = NULL;
    long expected = strtol(argv[1], &end, 10);
    if (!end || *end || expected <= 1 || getppid() != (pid_t)expected) return 125;
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) { perror("PR_SET_PDEATHSIG"); return 126; }
    // Covers death before or during prctl; no orphan can proceed into the builder.
    if (getppid() != (pid_t)expected) return 125;
    sigset_t empty;
    sigemptyset(&empty);
    if (sigprocmask(SIG_SETMASK, &empty, NULL) != 0) return 126;
    execvp(argv[2], argv + 2);
    perror("execvp");
    return errno == ENOENT ? 127 : 126;
}
