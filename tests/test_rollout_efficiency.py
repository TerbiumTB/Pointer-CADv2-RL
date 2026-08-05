import unittest
import multiprocessing as mp
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

from rl.cad_environment import empty_brep_graph
from measurements.chamfer_distance import chamfer_distance
from rl.cad_environment import evaluate_models
from rl.rollout_generator import (
    _LazyMesh,
    _RolloutJob,
    RolloutGenerator,
    build_runtime_config,
    decode_step_generation,
    _graph_from_payload,
    _graph_to_payload,
    _run_batched_jobs,
    _task_rank,
)
from rl.prompts import prompt_message
from rl.rollout_store import RolloutStore
from rl.schemas import EpisodeRecord, TrajectoryRecord, canonical_json


def _send_empty_graph(output_queue):
    output_queue.put(_graph_to_payload(empty_brep_graph()))


class _FailingBatchGenerator:
    def __init__(self, store, source_dataset_root):
        import torch

        self.store = store
        self.source_dataset_root = source_dataset_root
        self.device = torch.device("cpu")

    def _generate_actions(self, **kwargs):
        raise RuntimeError("expected generation failure")


class LazyMeshTest(unittest.TestCase):
    def test_builds_value_only_once(self):
        value = object()
        builder = Mock(return_value=value)
        lazy = _LazyMesh(builder)

        self.assertIs(lazy.get(), value)
        self.assertIs(lazy.get(), value)
        builder.assert_called_once_with()
        self.assertGreaterEqual(lazy.build_time_seconds, 0.0)

    def test_caches_build_failure(self):
        error = RuntimeError("mesh failed")
        builder = Mock(side_effect=error)
        lazy = _LazyMesh(builder)

        with self.assertRaisesRegex(RuntimeError, "mesh failed"):
            lazy.get()
        with self.assertRaisesRegex(RuntimeError, "mesh failed"):
            lazy.get()
        builder.assert_called_once_with()


class CachedMetricMeshTest(unittest.TestCase):
    @patch("rl.cad_environment.topology_counts")
    @patch("rl.cad_environment.chamfer_distance_from_meshes")
    def test_reuses_prediction_mesh_across_metrics(
        self, chamfer_from_meshes, topology_counts
    ):
        topology_counts.return_value = {
            "vertex_count": 1,
            "edge_count": 1,
            "face_count": 1,
        }
        chamfer_from_meshes.return_value = 0.25
        prediction_mesh = Mock(is_watertight=True)
        target_mesh = object()
        prediction_builder = Mock(return_value=prediction_mesh)
        target_builder = Mock(return_value=target_mesh)
        prediction_cache = _LazyMesh(prediction_builder)
        target_cache = _LazyMesh(target_builder)
        prediction = Mock(seq=[object()])
        target = Mock(seq=[object()])

        result = evaluate_models(
            prediction=prediction,
            target=target,
            enabled_metrics=["chamfer_distance", "watertight"],
            chamfer_points=128,
            prediction_mesh_factory=prediction_cache.get,
            target_mesh_factory=target_cache.get,
        )

        self.assertEqual(result["chamfer_distance"], 0.25)
        self.assertTrue(result["watertight"])
        prediction_builder.assert_called_once_with()
        target_builder.assert_called_once_with()
        chamfer_from_meshes.assert_called_once_with(
            prediction_mesh,
            target_mesh,
            points=128,
        )


class ChamferCompatibilityTest(unittest.TestCase):
    @patch("measurements.chamfer_distance.chamfer_distance_from_meshes")
    @patch("measurements.chamfer_distance.create_mesh")
    def test_model_api_delegates_to_mesh_api(self, create_mesh, mesh_metric):
        prediction = object()
        target = object()
        prediction_mesh = object()
        target_mesh = object()
        create_mesh.side_effect = [prediction_mesh, target_mesh]
        mesh_metric.return_value = 0.5

        result = chamfer_distance(prediction, target, points=256)

        self.assertEqual(result, 0.5)
        self.assertEqual(
            create_mesh.call_args_list,
            [call(prediction), call(target)],
        )
        mesh_metric.assert_called_once_with(
            prediction_mesh,
            target_mesh,
            points=256,
            type="uniform",
            normalize=True,
        )


class ParallelInfrastructureTest(unittest.TestCase):
    def test_empty_action_fails_only_its_batch_item(self):
        import torch

        class Inputs(dict):
            def to(self, _device):
                return self

        tokenizer = Mock(pad_token_id=151643)
        tokenizer.convert_tokens_to_ids.side_effect = {
            "<|cad_start|>": 100,
            "<|cad_pad|>": 101,
        }.__getitem__
        tokenizer.decode.return_value = "plan"
        processor = Mock(tokenizer=tokenizer)
        processor.apply_chat_template.return_value = "rendered"
        processor.return_value = Inputs()
        model = Mock()
        model.predict.return_value = (
            torch.tensor(
                [
                    [11, 100, 101, 102],
                    [11, 151643, 151643, 151643],
                ]
            ),
            [
                {"length": [], "angle": []},
                {"length": [], "angle": []},
            ],
            [torch.tensor([1]), torch.tensor([])],
            [torch.tensor([0]), torch.tensor([])],
            [torch.tensor([0]), torch.tensor([])],
            [
                {
                    "plan": [-0.1, -0.2, -0.3],
                    "label": [-0.4],
                    "parameter": [0.0],
                    "pointer": [0.0],
                },
                {
                    "plan": [-0.1],
                    "label": [],
                    "parameter": [],
                    "pointer": [],
                },
            ],
        )
        generator = RolloutGenerator(
            model=model,
            processor=processor,
            device=torch.device("cpu"),
            store=Mock(),
            source_dataset_root=Path("source"),
            config={
                "generation": {
                    "cpu_workers_per_gpu": 2,
                    "max_input_length": 32,
                    "max_generation_steps": 16,
                    "temperatures": {},
                }
            },
        )

        decoded, errors, _ = generator._generate_actions(
            prompts=["first", "second"],
            graphs=[empty_brep_graph(), empty_brep_graph()],
            sampling_generators=[None, None],
        )

        self.assertIsNotNone(decoded[0])
        self.assertIsNone(errors[0])
        self.assertIsNone(decoded[1])
        self.assertIn("without a CAD action", errors[1])
        model.predict.assert_called_once()

    def test_decode_ignores_padding_after_shorter_batched_generation(self):
        import torch

        tokenizer = Mock(pad_token_id=151643)
        tokenizer.convert_tokens_to_ids.side_effect = {
            "<|cad_start|>": 100,
            "<|cad_pad|>": 101,
        }.__getitem__
        tokenizer.decode.return_value = "plan"

        decoded = decode_step_generation(
            tokenizer=tokenizer,
            generated_ids=torch.tensor(
                [11, 100, 101, 102, 151643, 151643]
            ),
            parameter_map={"length": [], "angle": []},
            labels=torch.tensor([1]),
            parameters=torch.tensor([0]),
            pointers=torch.tensor([0]),
            behavior_log_probs={
                "plan": [-0.1, -0.2, -0.3],
                "label": [-0.4],
                "parameter": [0.0],
                "pointer": [0.0],
            },
        )

        self.assertEqual(decoded[0], "plan")
        self.assertEqual(decoded[1], [11, 100, 102])
        tokenizer.decode.assert_called_once_with([11])

    def test_batched_prompt_rendering_produces_strings(self):
        import torch

        processor = Mock()
        processor.apply_chat_template.side_effect = (
            lambda messages, **_: f"rendered:{messages[1]['content'][1]['text']}"
        )
        generator = RolloutGenerator(
            model=Mock(),
            processor=processor,
            device=torch.device("cpu"),
            store=Mock(),
            source_dataset_root=Path("source"),
            config={"generation": {"cpu_workers_per_gpu": 2}},
        )

        rendered = [
            generator._render_prompt(prompt)
            for prompt in ("first", "second")
        ]

        self.assertEqual(rendered, ["rendered:first", "rendered:second"])
        processor.apply_chat_template.assert_any_call(
            prompt_message("first"),
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_dgl_graph_crosses_spawn_process_boundary(self):
        context = mp.get_context("spawn")
        output_queue = context.Queue()
        process = context.Process(
            target=_send_empty_graph, args=(output_queue,)
        )
        process.start()
        payload = output_queue.get(timeout=10.0)
        process.join(timeout=10.0)
        graph = _graph_from_payload(payload)

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(graph.num_nodes(), 0)
        self.assertEqual(graph.num_edges(), 0)
        output_queue.close()

    def test_rank_shards_use_independent_part_names(self):
        record = TrajectoryRecord(
            trajectory_id="trajectory",
            task_id="task",
            rollout_index=0,
            seed=0,
            termination_reason="rollout_error",
            num_generated_steps=0,
            valid=False,
            final_cad_path=None,
            final_mesh_path=None,
            execution_error="test",
            metrics_json=canonical_json({}),
            generation_time_seconds=0.0,
            execution_time_seconds=0.0,
            metrics_version="test",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rollouts"
            first = RolloutStore.create(
                root, {"test": True}, writer_id="r0000"
            )
            second = RolloutStore.create(
                root, {"test": True}, exist_ok=True, writer_id="r0001"
            )
            first.append([record], [])
            second.append([record], [])

            names = sorted(
                path.name for path in first.trajectories_dir.glob("*.parquet")
            )
            self.assertEqual(
                names,
                ["part-r0000-000000.parquet", "part-r0001-000000.parquet"],
            )

    def test_task_rank_is_stable_and_bounded(self):
        self.assertEqual(_task_rank("task-a", 8), _task_rank("task-a", 8))
        self.assertIn(_task_rank("task-b", 8), range(8))

    def test_parallel_knobs_are_versioned_in_runtime_config(self):
        config = {
            "generation": {
                "batch_size": 8,
                "cpu_workers_per_gpu": 12,
                "batch_wait_seconds": 0.03,
                "max_episode_steps": 64,
            },
            "execution": {
                "cpu_threads_per_worker": 1,
                "strict": True,
            },
        }
        with tempfile.NamedTemporaryFile() as checkpoint:
            runtime = build_runtime_config(config, Path(checkpoint.name))

        self.assertEqual(runtime["generation"]["batch_size"], 8)
        self.assertEqual(runtime["generation"]["cpu_workers_per_gpu"], 12)
        self.assertEqual(runtime["generation"]["batch_wait_seconds"], 0.03)
        self.assertEqual(runtime["execution"]["cpu_threads_per_worker"], 1)

    def test_cpu_worker_pipeline_returns_generation_failure(self):
        config = {
            "generation": {
                "batch_size": 1,
                "cpu_workers_per_gpu": 1,
                "batch_wait_seconds": 0.0,
                "max_episode_steps": 2,
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
            "outputs": {"save_step": False, "save_mesh": False},
        }
        episode = EpisodeRecord(
            task_id="task",
            split="train",
            source_dataset="test",
            chunk="0000",
            model_id="0001",
            prompt_variant="exp",
            prompt="test prompt",
            target_cad_path="missing.json",
            source_step_ids=["00001"],
            num_steps=1,
            operation_types=["ExtrudeFeature"],
            preprocessing_version="test",
        )
        job = _RolloutJob(
            episode=episode,
            rollout_index=0,
            seed=7,
            trajectory_id="trajectory",
        )
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / "source"
            source_root.mkdir()
            store = RolloutStore.create(
                Path(temporary) / "rollouts", {"test": True}
            )
            generator = _FailingBatchGenerator(store, source_root)
            results = []

            _run_batched_jobs(
                generator=generator,
                jobs=[job],
                config=config,
                on_result=lambda trajectory, steps: results.append(
                    (trajectory, steps)
                ),
            )

        self.assertEqual(len(results), 1)
        trajectory, steps = results[0]
        self.assertEqual(trajectory.termination_reason, "generation_error")
        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main()
