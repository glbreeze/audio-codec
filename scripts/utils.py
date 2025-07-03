import torch
import argparse
from types import SimpleNamespace

class MeanAverager:
    def __init__(self):
        self.sums = {}
        self.count = 0

    def update(self, metrics: dict):
        for k, v in metrics.items():
            v = v.item() if torch.is_tensor(v) else v
            if k not in self.sums:
                self.sums[k] = 0.0
            self.sums[k] += v
        self.count += 1

    def average(self):
        return {k: v / self.count for k, v in self.sums.items()} if self.count > 0 else {}

def namespace_to_dict(ns):
    if isinstance(ns, (SimpleNamespace, argparse.Namespace)):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns