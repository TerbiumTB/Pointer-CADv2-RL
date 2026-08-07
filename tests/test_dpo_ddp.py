import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from tests.test_pointercad_predict import empty_brep_batch, lightweight_pointercad


class DPOConditionalDDPTest(unittest.TestCase):
    def test_zero_anchor_allows_repeated_ddp_iterations_without_unused_scan(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is unavailable")
        model = lightweight_pointercad()
        model.brep = nn.Linear(2, 2)
        model.parameter = nn.Linear(2, 2)
        for parameter in model.parameters():
            parameter.requires_grad = False
        conditional_modules = (
            model.brep,
            model.parameter,
            model.parameter_projection,
            model.pointer_projection,
        )
        for module in conditional_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        model.parameter_tau.requires_grad = True

        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = Path(temporary) / "ddp-init"
            try:
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"file://{rendezvous}",
                    rank=0,
                    world_size=1,
                )
            except RuntimeError as exc:
                self.skipTest(f"cannot initialize local Gloo process group: {exc}")
            try:
                ddp = DistributedDataParallel(
                    model, find_unused_parameters=False
                )
                input_ids = torch.tensor([[42, 43]])
                for _ in range(2):
                    outputs = ddp(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        breps=empty_brep_batch(1),
                        parameter_maps=None,
                        ensure_conditional_parameter_usage=True,
                    )
                    (outputs[5] * 0.0).backward()
                    for module in conditional_modules:
                        for parameter in module.parameters():
                            self.assertIsNotNone(parameter.grad)
                    ddp.zero_grad(set_to_none=True)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
