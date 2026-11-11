#!/usr/bin/env python3
"""Plot measured CSVs only: python plot.py RESULTS_DIRECTORY."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory

    def read(name):
        path = root / f'{name}.csv'
        if not path.exists():
            print(f'Skipping absent {path}')
            return pd.DataFrame()
        df = pd.read_csv(path)
        # Mixing devices or runtime versions makes the comparisons misleading.
        if len(df[['gpu', 'cpu', 'cuda_runtime', 'cuda_driver']].drop_duplicates()) > 1:
            raise ValueError(f'{path}: mixed environments; plot each run separately')
        return df[df.correct.eq('pass')].copy()

    def finish(filename, ylabel):
        plt.ylabel(ylabel)
        plt.grid(True, alpha=.25)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(root / filename, dpi=160)
        plt.close()

    df = read('transfer')
    if not df.empty:
        plt.figure()
        for (strategy, direction), group in df.groupby(['strategy', 'direction']):
            group = group.sort_values('bytes')
            plt.plot(group.bytes, group.GB_s, 'o-', label=f'{strategy} {direction}')
        plt.xscale('log', base=2)
        plt.xlabel('Transfer size (bytes)')
        finish('bandwidth.png', 'Bandwidth (decimal GB/s, CUDA event interval)')
    df = read('memory')
    if not df.empty:
        plt.figure()
        for strategy, group in df.groupby('strategy'):
            group = group.sort_values('bytes')
            for column, style in [('first_ms', '--'), ('repeat_ms', '-')]:
                plt.plot(group.bytes, group[column], style, label=f'{strategy} {column}')
        plt.xscale('log', base=2)
        plt.xlabel('Input size (bytes)')
        finish('memory.png', 'Kernel time (ms)')
    df = read('spmv')
    if not df.empty:
        pivot = df.pivot(index='matrix', columns='strategy', values='repeat_ms')
        pivot.index = [Path(p).name for p in pivot.index]
        pivot.plot.bar(figsize=(10, 5), rot=30)
        plt.xlabel('Matrix')
        finish('spmv.png', 'Steady-state cuSPARSE SpMV (ms)')
    df = read('amortization')
    reports = []
    if not df.empty:
        plt.figure(figsize=(10, 6))
        for (matrix, strategy), group in df.groupby(['matrix', 'strategy']):
            group = group.sort_values('iterations')
            plt.plot(group.iterations, group.total_host_ms, 'o-',
                     label=f'{Path(matrix).name}: {strategy}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('SpMV count (movement included; allocation excluded)')
        finish('amortization.png', 'Measured batch total (host ms)')
        for matrix, group in df.groupby('matrix'):
            table = group.pivot(index='iterations', columns='strategy', values='total_host_ms').sort_index()
            if 'hbm' not in table:
                continue
            for other in ['host', 'managed']:
                if other not in table:
                    continue
                valid = table[['hbm', other]].dropna()
                winners = valid.index[valid.hbm < valid[other]]
                if len(winners):
                    n = winners[0]
                    prior = valid.index[valid.index < n]
                    status = (f'observed crossover in ({prior[-1]}, {n}]'
                              if len(prior) else f'HBM already faster at first measured count {n}')
                else:
                    status = 'no measured count favors HBM'
                reports.append(f'{matrix}: HBM vs {other}: {status}')
    if reports:
        (root / 'crossover.txt').write_text('\n'.join(reports) + '\n')
        print('\n'.join(reports))


if __name__ == '__main__':
    main()
