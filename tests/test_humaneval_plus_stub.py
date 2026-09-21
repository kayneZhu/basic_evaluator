#!/usr/bin/env python3
"""HumanEval+ is a marked stub, not a fake implementation."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.humaneval_plus_adaptor import STUB_MESSAGE, HumanEvalPlusAdaptor


class TestHumanEvalPlusStub(unittest.TestCase):
    def test_factory_key_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            AdaptorFactory.create_adaptor("humaneval_plus", "unused.jsonl")
        self.assertIn("STUB", str(ctx.exception))
        self.assertIn("HumanEval+", str(ctx.exception))
        self.assertIn("sandbox", str(ctx.exception).lower())

    def test_alias_and_direct_class(self):
        with self.assertRaises(NotImplementedError):
            AdaptorFactory.create_adaptor("humaneval+", "unused.jsonl")
        with self.assertRaises(NotImplementedError):
            HumanEvalPlusAdaptor("unused.jsonl")
        self.assertIn("evalplus", STUB_MESSAGE.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
