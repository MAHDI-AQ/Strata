"""Summarize native benchmark JSON and optionally compare exact output token IDs."""
import argparse
import json
from pathlib import Path


def requests(report):
    return {key: item['tokens'] for run in report['runs'] for key, item in run['requests'].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reference = requests(json.loads(args.reference.read_text())) if args.reference else None
    results = {}
    success = True
    for path in args.reports:
        report = json.loads(path.read_text())
        rows = report['runs']
        actual = requests(report)
        total = sum(map(len, actual.values()))
        wall = sum(row['wall_s'] for row in rows)
        span = sum(row['common_decode_s'] for row in rows)
        overlap = sum((row['aggregate_common_decode_tps'] or 0) * row['common_decode_s'] for row in rows)
        result = dict(complete=report['complete'] and report.get('exit_code') == 0,
                      sha256=report['sha256'], requests=len(actual), tokens=total,
                      wall_tps=total / wall if wall else None,
                      decode_tps=overlap / span if span else None)
        if reference is not None:
            differing = sorted(key for key in actual.keys() | reference.keys() if actual.get(key) != reference.get(key))
            result.update(exact_match=not differing, differing_requests=differing)
            success &= not differing
        success &= result['complete']
        results[path.stem] = result
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
