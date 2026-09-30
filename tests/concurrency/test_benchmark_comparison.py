import unittest

from tools.summarize_concurrency import compare


def response(tokens, reason='length', elapsed=1):
    return {'tokens': tokens, 'done': f'DONE {len(tokens)} 20 0 {elapsed} {reason} 0 0 0'}


class ComparisonTests(unittest.TestCase):
    def test_timing_does_not_affect_parity(self):
        self.assertEqual(compare({'1': response([2, 3])}, {'1': response([2, 3], elapsed=9)}), [])

    def test_token_mismatch_and_truncation(self):
        for candidate, position in [([2, 4], 1), ([2], 1), ([2, 3, 4], 2)]:
            with self.subTest(candidate=candidate):
                self.assertEqual(compare({'1': response(candidate)}, {'1': response([2, 3])})[0]['first_token_mismatch'], position)

    def test_finish_and_missing_request(self):
        self.assertTrue(compare({'1': response([2], 'stop')}, {'1': response([2])}))
        self.assertTrue(compare({'1': response([2])}, {}))
        self.assertTrue(compare({'1': {'tokens': [2]}}, {'1': {'tokens': [2]}}))
