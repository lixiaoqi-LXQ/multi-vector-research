"""Render the six accepted full-LoTTE rows; never uses calibration throughput."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text())
    assert data['scope'] == 'full_LoTTE_4bit_Top10_B1_three_rounds'
    rows = data['records']
    labels = ['Chimera', 'PLAID native', 'PLAID gpu-all']
    targets = [.90, .95]
    lookup = {(r['label'], r['target_recall10']): r for r in rows}
    assert len(rows) == len(lookup) == 6
    for label in labels:
        for target in targets:
            r = lookup[label, target]
            assert r['formal_output_accepted'] and r['performance_window_accepted']
            assert len(r['rounds']) == 3
            assert all(x['queries'] == 2931 for x in r['rounds'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), layout='constrained')
    colors = ['#315b8a', '#cc8540', '#378579']
    width = .23
    for j, (label, color) in enumerate(zip(labels, colors)):
        x = np.arange(2) + (j - 1) * width
        for axis, metric, ylabel in zip(axes, ['qps', 'p99_ms'],
                                        ['QPS', 'Mean of round p99 (ms)']):
            values, low, high = [], [], []
            for target in targets:
                samples = [r[metric] for r in lookup[label, target]['rounds']]
                mean = statistics.mean(samples)
                values.append(mean)
                low.append(mean - min(samples))
                high.append(max(samples) - mean)
            bars = axis.bar(x, values, width, color=color, label=label,
                            yerr=[low, high], capsize=3, linewidth=.8,
                            error_kw={'elinewidth': .9})
            for i, (bar, target) in enumerate(zip(bars, targets)):
                r = lookup[label, target]
                qualified = r['qualification']
                if not qualified:
                    bar.set_hatch('///')
                    bar.set_alpha(.45)
                if metric == 'qps':
                    text = f"{values[i]:.1f}"
                    if not qualified:
                        text += f"\nR10={r['full_recall10']:.4f}\nbelow target"
                    axis.annotate(text, (bar.get_x() + width / 2, values[i] + high[i]),
                                  xytext=(0, 5), textcoords='offset points',
                                  ha='center', va='bottom', fontsize=8)
            axis.set_ylabel(ylabel)
            axis.set_xticks([0, 1], ['0.90', '0.95'])
            axis.set_xlabel('Recall@10 target')
            axis.grid(axis='y', alpha=.18)
            axis.set_axisbelow(True)
            axis.margins(y=.27)
    axes[0].set_title('Completed synchronous searches')
    axes[1].set_title('Tail latency')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside upper center', ncols=3, frameon=False)
    fig.supxlabel('Full LoTTE · 4bit · B1 · 2,931 queries × 3 rounds\n'
                  'Whiskers: observed three-round range; hatched bars: full-query target missed.',
                  fontsize=9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['svg', 'png']:
        fig.savefig(out.with_suffix('.' + suffix), dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
