import unittest
from unittest.mock import Mock, call, patch

from measurements.chamfer_distance import chamfer_distance
from rl.cad_environment import evaluate_models
from rl.rollout_generator import _LazyMesh


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


if __name__ == "__main__":
    unittest.main()
