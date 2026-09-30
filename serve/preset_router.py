"""Local single-resident preset router. Switching waits for all active HTTP requests."""
import argparse
import collections
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs


class Presets:
    def __init__(self, config):
        self.config = config
        self.entries = config['models']
        self.configs = {name: json.loads(Path(path).read_text(encoding='utf-8')) for name, path in self.entries.items()}
        self.port = config.get('backend_port', 8081)
        self.cv = threading.Condition()
        self.pending = collections.deque()
        self.current = None
        self.active = 0
        self.switching = False
        self.closed = False
        self.proc = None
        self.log = None

    def stop_backend(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(self.proc.pid), '/T', '/F'], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    import signal
                    os.killpg(self.proc.pid, signal.SIGTERM)
                try: self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
            self.proc = None
        if self.log is not None:
            self.log.close(); self.log = None
        self.current = None

    def load(self, name):
        self.stop_backend()
        cfg = self.configs[name]
        log_path = Path(self.config['log_dir']) / (name + '.server.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open('a', encoding='utf-8')
        command = [self.config.get('python', sys.executable), '-u', str(Path(__file__).with_name('server.py')),
                   '--engine', 'strata', '--config', self.entries[name], '--host', '127.0.0.1', '--port', str(self.port)]
        kwargs = dict(stdout=self.log, stderr=self.log, cwd=cfg['cwd'])
        if os.name == 'nt': kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else: kwargs['start_new_session'] = True
        self.proc = subprocess.Popen(command, **kwargs)
        print('Loading preset:', name, flush=True)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.closed: raise RuntimeError('Router is shutting down')
            if self.proc.poll() is not None: raise RuntimeError(f'{name} failed to start; see {log_path}')
            conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
            try:
                conn.request('GET', '/v1/models')
                response = conn.getresponse()
                data = json.loads(response.read())
                if response.status == 200 and any(m['id'] == name for m in data.get('data', [])):
                    self.current = name
                    print('Ready preset:', name, flush=True)
                    return
            except (OSError, ValueError, http.client.HTTPException): pass
            finally: conn.close()
            time.sleep(.5)
        raise RuntimeError(f'{name} startup timed out; see {log_path}')

    def acquire(self, name):
        if name not in self.entries: raise ValueError('Unknown model ID: ' + str(name))
        ticket = object()
        with self.cv:
            self.pending.append(ticket)
            try:
                while True:
                    if self.closed: raise RuntimeError('Router is shutting down')
                    if self.pending[0] is ticket and not self.switching:
                        alive = self.proc is not None and self.proc.poll() is None
                        if self.current == name and alive:
                            self.pending.popleft(); self.active += 1; self.cv.notify_all(); return
                        if self.active == 0:
                            self.switching = True
                            break
                    self.cv.wait(timeout=.5)
            except BaseException:
                self.pending.remove(ticket); self.cv.notify_all(); raise
        try:
            self.load(name)
        except BaseException:
            self.stop_backend()
            with self.cv:
                self.switching = False; self.pending.remove(ticket); self.cv.notify_all()
            raise
        with self.cv:
            self.switching = False; self.pending.popleft(); self.active += 1; self.cv.notify_all()

    def release(self):
        with self.cv:
            self.active -= 1; self.cv.notify_all()

    def models(self):
        models = []
        for name, cfg in self.configs.items():
            args = cfg['args']
            context = int(args[args.index('--max-context') + 1])
            models.append({'id': name, 'object': 'model', 'owned_by': 'local',
                           'status': {'value': 'loaded' if name == self.current else 'unloaded'},
                           'meta': {'n_ctx': context}, 'architecture': {'input_modalities': ['text', 'image'] if cfg.get('vision') else ['text'], 'output_modalities': ['text']}})
        return {'object': 'list', 'data': models}


def handler(manager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, fmt, *args): pass
        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path.rstrip('/')
            if path in ('/v1/models', '/models'):
                self.send_json(200, manager.models()); return
            if path in ('/health', '/healthz'):
                self.send_json(200, {'status': 'ok', 'loaded_model': manager.current}); return
            if path in ('/router/status', ''):
                with manager.cv:
                    status = dict(loaded_model=manager.current, active_requests=manager.active,
                                  queued_requests=len(manager.pending), switching=manager.switching)
                self.send_json(200, status); return
            if path == '/props':
                name = parse_qs(urlsplit(self.path).query).get('model', [manager.current or next(iter(manager.entries))])[0]
                if name not in manager.entries:
                    self.send_json(404, {'error': {'message': 'model not found'}}); return
                cfg = manager.configs[name]; args = cfg['args']
                self.send_json(200, {'model_alias':name, 'default_generation_settings':{'n_ctx':int(args[args.index('--max-context')+1]),'params':{'n_predict':-1}},
                                     'modalities':{'vision':bool(cfg.get('vision'))}, 'models_autoload':True,
                                     'total_slots':int(args[args.index('--concurrency')+1]), 'is_sleeping':manager.current != name}); return
            self.send_json(404, {'error': {'message': 'not found'}})

        def do_POST(self):
            path = urlsplit(self.path).path.rstrip('/')
            if path not in ('/v1/chat/completions', '/v1/messages'):
                self.send_json(404, {'error': {'message': 'not found'}}); return
            acquired = False; conn = None; headers_sent = False
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32*1024*1024: raise ValueError('Request body must be 1 byte to 32 MiB')
                body = self.rfile.read(size)
                request = json.loads(body)
                if not isinstance(request, dict): raise ValueError('Request body must be a JSON object')
                name = request.get('model')
                if not isinstance(name, str): raise ValueError('A model ID is required')
                manager.acquire(name); acquired = True
                conn = http.client.HTTPConnection('127.0.0.1', manager.port, timeout=1800)
                headers = {'Content-Type':'application/json'}
                for key in ('anthropic-version', 'anthropic-beta'):
                    if key in self.headers: headers[key] = self.headers[key]
                conn.request('POST', self.path, body=body, headers=headers)
                response = conn.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close'); self.end_headers(); headers_sent = True
                self.close_connection = True
                while True:
                    chunk = response.read1(65536)
                    if not chunk: break
                    self.wfile.write(chunk); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception as exc:
                if not headers_sent:
                    self.send_json(400 if isinstance(exc, (ValueError, json.JSONDecodeError)) else 503,
                                   {'error': {'type':'invalid_request_error' if isinstance(exc,ValueError) else 'server_error', 'message':str(exc)}})
                else: self.close_connection = True
            finally:
                if conn: conn.close()
                if acquired: manager.release()
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    manager = Presets(config)
    # Never start model processes if either listening port already belongs to another service.
    import socket
    with socket.socket() as probe: probe.bind(('127.0.0.1', manager.port))
    server = ThreadingHTTPServer(('127.0.0.1',config.get('port',8080)),handler(manager))
    server.daemon_threads = True
    print('Preset router ready:',server.server_address,flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        with manager.cv:
            manager.closed = True; manager.cv.notify_all()
            while manager.switching: manager.cv.wait(timeout=.5)
        server.server_close(); manager.stop_backend()


if __name__ == '__main__': main()
