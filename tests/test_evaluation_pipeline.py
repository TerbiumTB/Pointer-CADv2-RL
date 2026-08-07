import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import test as evaluation
from rl.cad_environment import empty_brep_graph
from rl.rollout_generator import _graph_to_payload


class _FailingEvaluationGenerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, tasks, graph_payloads):
        raise RuntimeError("expected evaluation generation failure")


class EvaluationTaskLoadingTest(unittest.TestCase):
    def test_source_fallback_selects_final_model_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            model_root = dataset_root / "0001" / "00000042"
            (model_root / "json").mkdir(parents=True)
            (model_root / "prompt_exp.txt").write_text(
                "build a part", encoding="utf-8"
            )
            (model_root / "json" / "00000042_00001.json").write_text(
                "{}", encoding="utf-8"
            )
            final_target = model_root / "json" / "00000042_00003.json"
            final_target.write_text("{}", encoding="utf-8")
            split_path = root / "train_val_test.json"
            split_path.write_text(
                json.dumps(
                    {
                        "train": [],
                        "validation": [],
                        "test": [
                            "0001_00000042_00003",
                            "0001_00000042_00001",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            tasks = evaluation._load_tasks_from_source(
                {
                    "dataset": {
                        "dataset_dir": str(dataset_root),
                        "split_filepath": str(split_path),
                        "split": "test",
                        "prompt_variant": "exp",
                    }
                }
            )

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task_id, "0001_00000042_exp")
            self.assertEqual(
                tasks[0].target_cad_path,
                "0001/00000042/json/00000042_00003.json",
            )


class EvaluationGeneratorTest(unittest.TestCase):
    def test_empty_action_fails_only_one_argmax_batch_item(self):
        class Inputs(dict):
            def to(self, _device):
                return self

        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.return_value = 151673
        processor = Mock(tokenizer=tokenizer)
        processor.apply_chat_template.side_effect = lambda messages, **_: (
            f"rendered:{messages[1]['content'][1]['text']}"
        )
        processor.return_value = Inputs()
        model = Mock()
        model.predict.return_value = (
            torch.tensor([[11, 151673], [12, 151673]]),
            [
                {"length": torch.tensor([]), "angle": torch.tensor([])},
                {"length": torch.tensor([]), "angle": torch.tensor([])},
            ],
            [torch.tensor([1]), torch.tensor([])],
            [torch.tensor([0]), torch.tensor([])],
            [torch.tensor([0]), torch.tensor([])],
        )
        generator = evaluation.EvaluationGenerator(
            model=model,
            processor=processor,
            device=torch.device("cpu"),
            config={
                "generation": {
                    "cpu_workers_per_gpu": 2,
                    "max_input_length": 32,
                    "max_generation_steps": 16,
                }
            },
        )
        tasks = [
            evaluation.EvaluationTask(
                task_id=f"task-{index}",
                split="test",
                chunk="0001",
                model_id=str(index),
                prompt_variant="exp",
                prompt=f"prompt-{index}",
                target_cad_path=f"target-{index}.json",
            )
            for index in range(2)
        ]
        payloads = [_graph_to_payload(empty_brep_graph()) for _ in tasks]

        actions, errors, _ = generator.generate(tasks, payloads)

        self.assertIsNotNone(actions[0])
        self.assertIsNone(errors[0])
        self.assertIsNone(actions[1])
        self.assertIn("without a CAD action", errors[1])
        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs["mode"], "argmax")


class EvaluationResumeTest(unittest.TestCase):
    def test_rank_shards_are_merged_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            result = {"status": True, "termination_reason": "model_end"}
            evaluation._append_result(
                output_dir / "results-r00000.jsonl", "task-a", result
            )
            evaluation._append_result(
                output_dir / "results-r00001.jsonl", "task-b", result
            )
            merged = evaluation._read_result_shards(output_dir)
            self.assertEqual(set(merged), {"task-a", "task-b"})

            evaluation._append_result(
                output_dir / "results-r00001.jsonl",
                "task-a",
                {"status": False},
            )
            with self.assertRaisesRegex(ValueError, "Conflicting results"):
                evaluation._read_result_shards(output_dir)

    def test_incomplete_final_jsonl_record_is_discarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            shard = output_dir / "results-r00000.jsonl"
            result = {"status": True, "termination_reason": "model_end"}
            evaluation._append_result(shard, "task-a", result)
            with shard.open("ab") as file:
                file.write(b'{"task_id":"incomplete"')

            merged = evaluation._read_result_shards(output_dir)

            self.assertEqual(merged, {"task-a": result})
            self.assertTrue(shard.read_bytes().endswith(b"\n"))


class EvaluationDistributedContextTest(unittest.TestCase):
    @patch.object(evaluation.dist, "init_process_group")
    def test_evaluation_uses_long_distributed_timeout(self, init_process_group):
        environment = {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}
        with patch.dict(os.environ, environment, clear=True):
            context = evaluation._distributed_context()

        self.assertEqual(context, (1, 2, 1))
        init_process_group.assert_called_once_with(
            backend="nccl",
            timeout=evaluation.timedelta(
                seconds=evaluation.DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS
            ),
        )

    @patch.object(evaluation.dist, "init_process_group")
    def test_evaluation_rejects_invalid_distributed_timeout(
        self, init_process_group
    ):
        environment = {
            "WORLD_SIZE": "2",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS": "0",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                evaluation._distributed_context()

        init_process_group.assert_not_called()


class EvaluationCPUWorkerPipelineTest(unittest.TestCase):
    def test_generation_failure_is_persisted_as_one_task_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            source_root = output_dir / "source"
            source_root.mkdir()
            task = evaluation.EvaluationTask(
                task_id="task-a",
                split="test",
                chunk="0001",
                model_id="0002",
                prompt_variant="exp",
                prompt="test prompt",
                target_cad_path="missing.json",
            )
            config = {
                "dataset": {"dataset_dir": str(source_root)},
                "generation": {
                    "batch_size": 1,
                    "cpu_workers_per_gpu": 1,
                    "batch_wait_seconds": 0.0,
                    "max_episode_steps": 2,
                    "base_seed": 0,
                },
                "execution": {
                    "strict": True,
                    "surf_u_samples": 2,
                    "surf_v_samples": 2,
                    "curv_u_samples": 2,
                    "build_timeout_seconds": 1.0,
                    "cpu_threads_per_worker": 1,
                },
                "metrics": {"version": "test", "enabled": []},
                "outputs": {"save_step": False, "save_stl": False},
            }

            evaluation._run_rank_evaluation(
                generator=_FailingEvaluationGenerator(),
                tasks=[task],
                config=config,
                output_dir=output_dir,
                rank=0,
            )

            result = evaluation._read_result_shards(output_dir)["task-a"]
            self.assertFalse(result["status"])
            self.assertEqual(result["termination_reason"], "generation_error")
            self.assertIn("expected evaluation", result["error_message"])


if __name__ == "__main__":
    unittest.main()
