"""CPU-only tests. No subprocess, model, CUDA context or network connection is created."""
import io
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, EngineDied, ByteTokenizer, MockEngine, Service


class Output:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        while True:
            line = self.queue.get()
            if line is None:
                return
            yield line


def engine():
    value = StrataEngine.__new__(StrataEngine)
    value._request_local = threading.local()
    value._write_lock = threading.Lock()
    value._channels_lock = threading.Lock()
    value._channels = {}
    value._request_number = 0
    value._shared_progress = {}
    value._shared_queue_wait = {}
    value.multiplex = True
    value.ended = False
    value.lines = queue.Queue()
    value.proc = SimpleNamespace(stdin=io.StringIO(), stdout=Output(), poll=lambda: None)
    value.info = {"concurrency": 4}
    thread = threading.Thread(target=value._pump, daemon=True)
    thread.start()
    return value, thread


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("mocked operation timed out")
        time.sleep(0.001)


class MultiplexTests(unittest.TestCase):
    def test_interleaved_live_rates_are_aggregate_and_monotonic(self):
        tok = ByteTokenizer()
        mock = MockEngine(tok, "ok", max_context=4096)
        mock.info = {"concurrency": 4}
        template = Path(__file__).resolve().parents[2] / "serve" / "chat_template.jinja"
        service = Service(mock, tok, ChatTemplate(template))
        service.active_requests = 2
        for rid in (1, 2):
            service.live_requests[rid] = dict(started=100.0, first_token=None, generated=0,
                prompt_tokens=10, max_tokens=100, phase="reading", tail="", tool=None)
        with patch("serve.server.time.time", return_value=100.0):
            service._note(1, [], 1)
        with patch("serve.server.time.time", return_value=101.0):
            service._note(51, [], 1)
            service._note(1, [], 2)  # a new request's smaller counter must not reset the total
        with patch("serve.server.time.time", return_value=102.0):
            service._note(101, [], 1)
            service._note(51, [], 2)
            self.assertEqual(service.status["generated"], 152)
            self.assertEqual(service._tok_s(), 75.5)
            metrics = service.metrics()
            self.assertEqual(metrics["live"]["rate_scope"], "aggregate")
            self.assertEqual(metrics["live"]["active_requests"], 2)
            self.assertEqual([r["generated"] for r in metrics["active_requests"]], [101, 51])
        with patch("serve.server.time.time", return_value=200.0):
            self.assertEqual(service._tok_s(), 0.0)

    def test_status_reports_capacity_and_active_requests(self):
        tok = ByteTokenizer()
        mock = MockEngine(tok, "ok", max_context=4096)
        mock.info = {"concurrency": 4}
        template = Path(__file__).resolve().parents[2] / "serve" / "chat_template.jinja"
        service = Service(mock, tok, ChatTemplate(template))
        service.status.update(busy=True, active_requests=2, queued=1)
        status = service.v1_status()
        self.assertEqual(status["concurrency"], {"serving": 4, "requested": 4})
        self.assertEqual(status["activity"]["in_flight"], 3)
        self.assertEqual(status["activity"]["requests"], 2)

    def setUp(self):
        self.engine, self.pump = engine()

    def tearDown(self):
        self.engine.proc.stdout.queue.put(None)
        self.pump.join(2)
        self.assertFalse(self.pump.is_alive())

    def emit(self, text):
        self.engine.proc.stdout.queue.put(text + "\n")

    def test_four_streams_and_timings_are_isolated(self):
        result, errors = {}, []
        def consume(index):
            try:
                tokens = list(self.engine.generate([index], 2, {}, threading.Event()))
                result[index] = (tokens, self.engine.last.copy())
            except Exception as exc:
                errors.append(exc)
        workers = []
        for index in range(1, 5):
            worker = threading.Thread(target=consume, args=(index,))
            workers.append(worker)
            worker.start()
            wait_for(lambda: len(self.engine._channels) == index)
        for index in (4, 2, 1, 3):
            self.emit(f"R {index} T {index * 10}")
        for index in (1, 3, 4, 2):
            self.emit(f"R {index} T {index * 10 + 1}")
            self.emit(f"R {index} DONE 2 {index} 1.0 {index * 10}.0 length 1 1 0")
        for worker in workers:
            worker.join(2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        for index in range(1, 5):
            tokens, timings = result[index]
            self.assertEqual(tokens, [index * 10, index * 10 + 1])
            self.assertEqual(timings["prompt_tokens"], index)
            self.assertEqual(timings["decode_ms"], index * 10)
        self.assertEqual(self.engine._channels, {})

    def test_close_cancels_only_its_request(self):
        first = self.engine.generate([1], 8, {}, threading.Event())
        second = self.engine.generate([2], 8, {}, threading.Event())
        self.emit("R 1 T 11")
        # Register before emitting: the pump deliberately ignores unknown/retired identities.
        a = threading.Thread(target=lambda: next(first))
        a.start()
        wait_for(lambda: 1 in self.engine._channels)
        self.emit("R 1 T 12")
        a.join(2)
        first.close()
        self.assertIn("CSTOP 1\n", self.engine.proc.stdin.getvalue())
        result = []
        b = threading.Thread(target=lambda: result.extend(second))
        b.start()
        wait_for(lambda: 2 in self.engine._channels)
        self.emit("R 1 T 999")
        self.emit("R 1 DONE 1 1 0 1 cancel 0 0 0")
        self.emit("R 2 T 22")
        self.emit("R 2 DONE 1 1 0 1 length 0 0 0")
        b.join(2)
        self.assertEqual(result, [22])
        self.assertNotIn("CSTOP 2", self.engine.proc.stdin.getvalue())

    def test_early_close_drains_the_done_before_the_channel_goes(self):
        # Service.run's stop-token break closes the generator while the engine's DONE is already
        # queued: the drain must deliver it, and no CSTOP is needed once DONE was consumed.
        got = []
        gen = self.engine.generate([1], 8, {}, threading.Event())
        worker = threading.Thread(target=lambda: got.append(next(gen)))
        worker.start()
        wait_for(lambda: 1 in self.engine._channels)
        self.emit("R 1 T 42")
        worker.join(2)
        self.assertEqual(got, [42])                  # the stop token: the consumer stops here
        self.emit("R 1 DONE 1 3 12.5 44.0 stop 1 1 5")
        gen.close()
        self.assertEqual(self.engine.last.get("prompt_ms"), 12.5)
        self.assertEqual(self.engine.last.get("reused"), 5)
        self.assertNotIn("CSTOP 1\n", self.engine.proc.stdin.getvalue())
        self.assertEqual(self.engine._channels, {})

    def test_slow_client_does_not_block_other_streams(self):
        slow, fast = queue.Queue(maxsize=256), queue.Queue(maxsize=256)
        self.engine._channels = {1: slow, 2: fast}
        for i in range(257):
            self.emit(f"R 1 T {i}")
        self.emit("R 2 T 42")
        self.assertEqual(fast.get(timeout=2).strip(), "T 42")
        self.assertTrue(slow.get(timeout=2).startswith("ERR "))
        self.assertIn("CSTOP 1", self.engine.proc.stdin.getvalue())

    def test_engine_exit_wakes_every_waiter(self):
        channels = {i: queue.Queue(maxsize=256) for i in range(4)}
        self.engine._channels = channels
        self.engine.proc.stdout.queue.put(None)
        for channel in channels.values():
            self.assertIsNone(channel.get(timeout=2))
        self.pump.join(2)
        self.assertFalse(self.engine.alive())

    def test_legacy_statistics_remain_visible_across_threads(self):
        self.engine.multiplex = False
        self.engine.last = {"generated": 7}
        self.engine.progress = (3, 10)
        values = []
        thread = threading.Thread(target=lambda: values.append((self.engine.last, self.engine.progress)))
        thread.start(); thread.join()
        self.assertEqual(values, [({"generated": 7}, (3, 10))])


if __name__ == "__main__":
    unittest.main()
