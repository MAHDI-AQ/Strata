"""Native-protocol concurrency benchmark. Loads ONE model; requires explicit config.

Reports aggregate output/wall time, per-request TTFT, and the token rate in the
interval where all batch members are decoding. Keeps raw token IDs for parity.
Model loading and a separate warmup are excluded. No server/preset router needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--exe', type=Path)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--requests', type=int, choices=range(1, 5), default=4,
                    help='number of essay requests; capacity remains the configured concurrency')
    ap.add_argument('--background-repeats', type=int, default=0)
    ap.add_argument('--workload', type=Path, help='JSON cases with tokens, max_new and optional native sampling fields')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--stock', action='store_true', help='omit fork-only options for the upstream reference')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    values = {}
    it = iter(cfg['args'])
    # Local serving configs use key/value options, except these boolean switches.
    flags = {'--vision', '--no-prefill-borrow'}
    for key in it:
        values[key] = None if key in flags else next(it)
    values.pop('--vision', None)
    if args.strict:
        values.update({'--expert-cache': '6000', '--prefill': '256',
                       '--concurrent-prefill': '256', '--short-read': '0',
                       '--adapt-every': '0', '--suffix-draft': '0',
                       '--prompt-cache': '0', '--pcie-frac': '0', '--no-prefill-borrow': None})
    for setting in args.set:
        key, value = setting.split('=', 1)
        values['--' + key.removeprefix('--')] = value
    values['--vram-reserve-mib'] = '2560'
    concurrency = int(values.get('--concurrency', '1'))
    if args.stock:
        concurrency = 1
        for key in ('--concurrency', '--concurrent-prefill', '--batch-rows', '--batch-policy', '--batch-graphs', '--batch-padding', '--batch-parallel'):
            values.pop(key, None)
    native_args = [v for k, value in values.items() for v in ([k] if value is None else [k, value])]
    exe = str(args.exe or cfg['exe'])
    env = dict(os.environ)
    env.update(cfg.get('env', {}))
    env['PATH'] = os.pathsep.join(cfg.get('lib_dirs', []) + [env['PATH']])
    env['STRATA_CONCURRENT_PROFILE'] = '1'
    if args.strict:
        env.update({k: '1' for k in ('STRATA_NO_IQ512', 'STRATA_NO_IQ256', 'STRATA_NO_IQ4NL')})
    p = Path(cfg['tokenizer'])
    vocab = json.loads((p / 'vocab.json').read_text(encoding='utf-8'))
    names = [''] * len(vocab)
    for name, index in vocab.items():
        names[index] = name
    tok = Tokenizer(names, (p / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                    json.loads((p / 'token_type.json').read_text(encoding='utf-8')))
    template = ChatTemplate(ROOT / 'serve/chat_template.jinja')
    base = ('Write me an essay on the general theory of relativity to a postgraduate level. '
            'This should be a lengthy thorough essay covering the history leading up to the '
            'derivation of the theory, followed by a reasonable indepth into the theory itself, '
            'key predictions, the underlying thought processes, etc. Leave all equations out of this for now.')
    background = ('The station recorded ordinary weather and filed its daily observations in the archive. ' *
                  args.background_repeats)
    cases = []
    for focus in ('historical development', 'physical intuition', 'experimental predictions', 'conceptual foundations'):
        prompt = background + base + '\nGive particular attention to ' + focus + '.'
        rendered = template.render([{'role': 'user', 'content': prompt}], enable_thinking=False)
        cases.append(dict(tokens=tok.encode(rendered, parse_special=True), max_new=args.tokens, sampling=''))
    if args.workload:
        cases = json.loads(args.workload.read_text(encoding='utf-8'))
    else:
        cases = cases[:args.requests]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'exe': exe, 'sha256': hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
              'args': native_args, 'strict': args.strict, 'concurrency': concurrency,
              'prompt_lengths': [len(c['tokens']) for c in cases], 'workload': cases,
              'environment': {k: v for k, v in env.items() if k.startswith('STRATA_')},
              'runs': [], 'complete': False}
    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    with args.output.with_suffix('.stderr.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([exe, '--serve', *native_args], cwd=cfg.get('cwd', ROOT), env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                text=True, encoding='utf-8', bufsize=1)
        q = queue.Queue()
        def read():
            for line in proc.stdout:
                q.put((time.perf_counter(), line.strip()))
            q.put((time.perf_counter(), None))
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        def receive():
            stamp, line = q.get(timeout=180)
            if line is None:
                raise RuntimeError(f'engine exited: {proc.wait(timeout=5)}')
            return stamp, line
        def send(line):
            proc.stdin.write(line + '\n')
            proc.stdin.flush()
        try:
            report['startup'] = []
            while True:
                _, line = receive()
                report['startup'].append(line)
                if line.startswith('READY '):
                    multi = 'multiplex' in line
                    print(line, flush=True)
                    break
            def batch(ids, maximum, rid_start):
                start = time.perf_counter()
                members = {rid_start + i: {'tokens': [], 'times': []} for i in range(len(ids))}
                for rid, prompt in zip(members, ids):
                    limit = maximum if maximum is not None else prompt['max_new']
                    sampling = prompt.get('sampling', '')
                    send(f'{"CGEN " + str(rid) if multi else "GEN"} {limit}{sampling} ' + ','.join(map(str, prompt['tokens'])))
                done = 0
                while done < len(members):
                    stamp, line = receive()
                    rid = rid_start
                    if multi:
                        if not line.startswith('R '):
                            continue
                        _, number, line = line.split(' ', 2)
                        rid = int(number)
                    item = members[rid]
                    if line.startswith('T '):
                        item['tokens'].append(int(line[2:]))
                        item['times'].append(stamp - start)
                    elif line.startswith('DONE '):
                        item['done'] = line
                        item['wall_s'] = stamp - start
                        item['ttft_s'] = item['times'][0] if item['times'] else None
                        done += 1
                    elif line.startswith('ERR'):
                        raise RuntimeError(line)
                wall = max(x['wall_s'] for x in members.values())
                begin = max(x['times'][0] for x in members.values() if x['times'])
                end = min(x['times'][-1] for x in members.values() if x['times'])
                # Exclude another 0.25s at either edge of the common decode interval.
                begin += 0.25
                end -= 0.25
                overlap_tokens = sum(sum(begin <= t <= end for t in x['times']) for x in members.values())
                result = {'requests': members, 'wall_s': wall,
                          'aggregate_wall_tps': sum(len(x['tokens']) for x in members.values()) / wall,
                          'common_decode_s': max(0, end - begin),
                          'aggregate_common_decode_tps': overlap_tokens / (end - begin) if end > begin else None}
                return result
            report['warmup'] = batch(cases[:concurrency], 32, 1)
            save()
            for repetition in range(args.repeat):
                for offset in range(0, len(cases), concurrency):
                    result = batch(cases[offset:offset + concurrency], None, 100 + repetition * len(cases) + offset)
                    report['runs'].append(result)
                    save()
                    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}), flush=True)
            report['complete'] = True
        finally:
            if proc.poll() is None:
                try:
                    send('QUIT')
                    proc.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    proc.kill()
                    proc.wait()
            report['exit_code'] = proc.returncode
            save()


if __name__ == '__main__':
    main()
