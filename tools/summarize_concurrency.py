"""Summarize native benchmark JSON and optionally compare exact output token IDs."""
import argparse
import json
from pathlib import Path


def requests(report):
    return {key: item for run in report['runs'] for key, item in run['requests'].items()}


def finish_reason(item):
    fields = item.get('done', '').split()
    return fields[5] if len(fields) > 5 and fields[0] == 'DONE' else None


def compare(actual, reference):
    differences = []
    for key in sorted(actual.keys() | reference.keys()):
        a, b = actual.get(key), reference.get(key)
        if a is None or b is None:
            differences.append(dict(request=key, error='missing request'))
            continue
        left, right = a['tokens'], b['tokens']
        first = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), None)
        if first is None and len(left) != len(right):
            first = min(len(left), len(right))
        if first is not None or finish_reason(a) != finish_reason(b) or finish_reason(a) is None:
            differences.append(dict(request=key, first_token_mismatch=first,
                                    actual_length=len(left), reference_length=len(right),
                                    actual_finish=finish_reason(a), reference_finish=finish_reason(b)))
    return differences


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
        total = sum(len(item['tokens']) for item in actual.values())
        wall = sum(row['wall_s'] for row in rows)
        span = sum(row['common_decode_s'] for row in rows)
        overlap = sum((row['aggregate_common_decode_tps'] or 0) * row['common_decode_s'] for row in rows)
        result = dict(complete=report['complete'] and report.get('exit_code') == 0,
                      sha256=report['sha256'], requests=len(actual), tokens=total,
                      wall_tps=total / wall if wall else None,
                      decode_tps=overlap / span if span else None)
        if reference is not None:
            differences = compare(actual, reference)
            differing = [item['request'] for item in differences]
            result.update(exact_match=not differing, differing_requests=differing, differences=differences)
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
