"""Project actual profile commands and selected raw counters into a report appendix."""
from pathlib import Path
import csv
import hashlib
import json
import shlex

ROOT = Path('/home/lxq/graduate/ai-workspace/research/multivector')
PLANS = ROOT / 'chimera/profile-v2/firstdelivery-R10-profile-plans'
METRICS = [
    'gpu__time_duration.sum',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
    'l1tex__throughput.avg.pct_of_peak_sustained_active',
    'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'l1tex__t_sector_hit_rate.pct',
    'lts__t_sector_hit_rate.pct',
    'sm__warps_active.avg.per_cycle_active',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'gpc__cycles_elapsed.avg.per_second',
    'lts__t_sectors_op_read.sum.per_second',
    'lts__t_sectors_op_write.sum.per_second',
]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

out = ['# LoTTE R10 瓶颈采证：实际命令与原始计数', '',
       '本附录保留实际命令、原始单位和值；profiling时间不进入正式QPS。', '',
       '两系统的部分achieved occupancy超过硬件或寄存器给出的上限，相关占用率判为无效，不用于瓶颈归类。L2请求率按原单位保留，不等同LPDDR流量。', '']
for plan in sorted(PLANS.iterdir()):
    if not plan.is_dir():
        continue
    out += ['## ' + plan.name, '']
    for phase in sorted(plan.iterdir()):
        process = phase / 'process.json'
        if not phase.is_dir() or not process.exists() or not phase.name.startswith(('nsys', 'sol-', 'memory-', 'compute-', 'latency-')):
            continue
        meta = json.loads(process.read_text())
        out += ['### ' + phase.name, '', '```sh', shlex.join(meta['command']), '```', '',
                f'命令记录：`{process.relative_to(ROOT)}`，SHA256 `{sha(process)}`。', '']
        p = phase / 'raw.csv'
        if not p.exists():
            continue
        rows = list(csv.reader(p.open()))
        if len(rows) < 3:
            continue
        header, units = rows[:2]
        out += [f'原始CSV：`{p.relative_to(ROOT)}`，SHA256 `{sha(p)}`。', '']
        for index, values in enumerate(rows[2:]):
            assert len(header) == len(units) == len(values)
            d = dict(zip(header, values)); u = dict(zip(header, units))
            out += [f"Kernel {index}: `{d.get('Kernel Name', 'unknown')}`", '', '| 原始metric | 原始unit | 原始value |', '|---|---|---:|']
            for key in METRICS:
                if key in d:
                    out.append(f'| `{key}` | {u[key]} | {d[key]} |')
            out += ['']
(ROOT / 'report/lotte-r10-profile-commands.md').write_text('\n'.join(out))
print('Wrote report/lotte-r10-profile-commands.md')
