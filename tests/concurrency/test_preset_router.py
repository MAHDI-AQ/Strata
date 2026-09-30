"""No model loading: verify preset-switch ordering and streamed HTTP forwarding."""
import concurrent.futures
import http.client
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from serve.preset_router import Presets, handler, available_backend_port


class FakePresets(Presets):
    def __init__(self):
        self.entries = {'one':'one', 'four':'four'}
        self.configs = {name:{'args':['--max-context',str(ctx),'--concurrency',str(c)]} for name,ctx,c in [('one',196608,1),('four',98304,4)]}
        self.cv = threading.Condition()
        import collections
        self.pending = collections.deque()
        self.current = None; self.active = 0; self.switching = False; self.closed = False
        self.proc = None; self.loads = []
    def load(self,name):
        self.loads.append(name); self.current = name
        self.proc = SimpleNamespace(poll=lambda:None)
    def stop_backend(self): self.proc = None; self.current = None


class RouterTests(unittest.TestCase):
    def test_busy_private_port_uses_another_port_without_sharing(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            chosen = available_backend_port(occupied.getsockname()[1])
            self.assertNotEqual(chosen, occupied.getsockname()[1])
            with socket.socket() as replacement:
                replacement.bind(('127.0.0.1', chosen))

    def test_same_preset_has_four_concurrent_leases(self):
        m = FakePresets()
        for _ in range(4): m.acquire('four')
        self.assertEqual(m.active,4); self.assertEqual(m.loads,['four'])
        for _ in range(4): m.release()

    def test_switch_waits_and_cannot_be_starved_by_old_preset(self):
        m = FakePresets(); m.acquire('one')
        switched = threading.Event(); release_switch = threading.Event(); old_admitted = threading.Event()
        def new():
            m.acquire('four'); switched.set(); release_switch.wait(3); m.release()
        def old():
            m.acquire('one'); old_admitted.set(); m.release()
        a = threading.Thread(target=new); a.start()
        deadline = time.monotonic()+3
        while not m.pending and time.monotonic()<deadline: time.sleep(.005)
        b = threading.Thread(target=old); b.start()
        self.assertFalse(switched.wait(.05)); self.assertFalse(old_admitted.is_set())
        m.release(); self.assertTrue(switched.wait(3)); self.assertFalse(old_admitted.is_set())
        release_switch.set(); a.join(3); b.join(3)
        self.assertTrue(old_admitted.is_set()); self.assertEqual(m.loads,['one','four','one'])

    def test_unknown_model_does_not_load(self):
        m = FakePresets()
        with self.assertRaises(ValueError): m.acquire('missing')
        self.assertEqual(m.loads,[])

    def test_failed_load_releases_switch_and_queue(self):
        m = FakePresets()
        original = m.load
        def fail(name): raise RuntimeError('simulated startup failure')
        m.load = fail
        with self.assertRaises(RuntimeError): m.acquire('one')
        self.assertFalse(m.switching); self.assertEqual(len(m.pending),0); self.assertEqual(m.active,0)
        m.load = original; m.acquire('four'); m.release()
        self.assertEqual(m.current,'four')

    def test_models_and_sse_forwarding(self):
        class Backend(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
                self.wfile.write(('data: '+json.dumps({'model':req['model'],'text':'hello'})+'\n\ndata: [DONE]\n\n').encode())
                self.wfile.flush()
        m=FakePresets()
        backend=ThreadingHTTPServer(('127.0.0.1',0),Backend); m.port=backend.server_port
        router=ThreadingHTTPServer(('127.0.0.1',0),handler(m))
        for server in [backend,router]: threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            conn=http.client.HTTPConnection('127.0.0.1',router.server_port,timeout=3)
            conn.request('GET','/v1/models'); models=json.loads(conn.getresponse().read())
            self.assertEqual([v['id'] for v in models['data']],['one','four'])
            conn.request('POST','/v1/chat/completions',body=json.dumps({'model':'four','stream':True}),headers={'Content-Type':'application/json'})
            response=conn.getresponse(); data=response.read().decode(); conn.close()
            self.assertEqual(response.status,200); self.assertIn('"model": "four"',data); self.assertIn('data: [DONE]',data)
        finally:
            for server in [router,backend]: server.shutdown(); server.server_close()


if __name__ == '__main__': unittest.main()
