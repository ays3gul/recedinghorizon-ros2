import os
import subprocess
import time
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation
import ros2_node

try:
    from ament_index_python.packages import get_package_share_directory
    _AMENT_INDEX = True
except ImportError:
    _AMENT_INDEX = False

try:
    from gazebo_msgs.srv import SpawnModel, DeleteModel
    _GAZEBO_MSGS = True
except ImportError:
    _GAZEBO_MSGS = False
    print("[SDFSpawner] gazebo_msgs not found — spawn/delete calls will be skipped. "
          "Install with: sudo apt install ros-jazzy-gazebo-msgs")

# Fallback SDF path used when the ament_index lookup fails (e.g. Gz Harmonic run)
_FALLBACK_SDF_PATH = (
    "/home/ayse/Desktop/RecedingHorizon/src/simulation_environment/sdfs/"
)
# Gz Harmonic world name (must match worlds/bunny_gz.sdf)
_GZ_WORLD = "bunny_world"


def _euler_to_quat(r, p, y):
    q = Rotation.from_euler("xyz", [r, p, y]).as_quat()  # [x, y, z, w]
    return q


def _gz_available():
    """True if the 'gz' CLI is on PATH (Gz Harmonic is running)."""
    return subprocess.call(["which", "gz"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


class SDFSpawner:
    def __init__(self, model_name="box"):
        self.model_name = model_name
        node = ros2_node.get_node()
        self._use_gz = False

        if _AMENT_INDEX:
            try:
                pkg_path = get_package_share_directory("simulation_environment")
                self.model_path = pkg_path + "/sdfs/"
            except Exception:
                self.model_path = _FALLBACK_SDF_PATH
        else:
            self.model_path = _FALLBACK_SDF_PATH

        if _GAZEBO_MSGS:
            self.spawn_client = node.create_client(SpawnModel,
                                                   "/gazebo/spawn_sdf_model")
            self.delete_client = node.create_client(DeleteModel,
                                                    "gazebo/delete_model")
            node.get_logger().info("[SDFSpawner] Waiting for Gazebo spawn service (5 s)...")
            if not self.spawn_client.wait_for_service(timeout_sec=5.0):
                node.get_logger().info(
                    "[SDFSpawner] /gazebo/spawn_sdf_model not available — "
                    "trying Gz Harmonic path."
                )
                self.spawn_client = None
                self.delete_client = None
                if _gz_available():
                    self._use_gz = True
                    node.get_logger().info("[SDFSpawner] Gz CLI found — spawning via 'gz service'.")
                else:
                    node.get_logger().info("[SDFSpawner] No simulator found; spawn calls will be skipped.")
        else:
            self.spawn_client = None
            self.delete_client = None
            if _gz_available():
                self._use_gz = True

    def _make_pose(self, pos):
        q = _euler_to_quat(0, 0, 0)
        return Pose(
            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            orientation=Quaternion(x=float(q[0]), y=float(q[1]),
                                   z=float(q[2]), w=float(q[3])),
        )

    def _spawn_gz(self, model_name, sdf_content, x, y, z):
        """Spawn a model in Gz Harmonic using gz service directly."""
        import os as _os, json as _json, tempfile as _tmp
        _env = _os.environ.copy()
        _gz_libs = [
            "/opt/ros/jazzy/opt/gz_transport_vendor/lib",
            "/opt/ros/jazzy/opt/gz_msgs_vendor/lib",
            "/opt/ros/jazzy/opt/gz_common_vendor/lib",
            "/opt/ros/jazzy/opt/gz_math_vendor/lib",
            "/opt/ros/jazzy/opt/gz_utils_vendor/lib",
            "/opt/ros/jazzy/opt/gz_sim_vendor/lib",
            "/opt/ros/jazzy/opt/gz_plugin_vendor/lib",
            "/opt/ros/jazzy/opt/sdformat_vendor/lib",
            "/opt/ros/jazzy/opt/gz_fuel_tools_vendor/lib",
        ]
        _existing = _env.get("LD_LIBRARY_PATH", "")
        _new_paths = ":".join(p for p in _gz_libs if p not in _existing)
        _env["LD_LIBRARY_PATH"] = _new_paths + ":" + _existing
        _env["GZ_PARTITION"] = _os.environ.get("GZ_PARTITION", "")
        req = _json.dumps({
            "name": model_name,
            "allow_renaming": False,
            "pose": {"position": {"x": x, "y": y, "z": z}},
            "sdf": sdf_content,
            "sdf_version": "1.6",
            "world_name": _GZ_WORLD,
        })
        try:
            result = subprocess.run(
                [
                    "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz",
                    "service",
                    "-s", f"/world/{_GZ_WORLD}/create",
                    "--reqtype", "gz.msgs.EntityFactory",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "5000",
                    "--req", f"sdf: '{sdf_content}' name: '{model_name}' "
                             f"pose {{ position {{ x: {x} y: {y} z: {z} }} }}",
                ],
                timeout=10,
                capture_output=True,
                text=True,
                env=_env,
            )
            if result.returncode != 0:
                print(f"[SDFSpawner] Gz spawn failed for {model_name}: {result.stderr.strip()}")
            else:
                print(f"[SDFSpawner] Spawned {model_name} at ({x},{y},{z})")
        except Exception as e:
            print(f"[SDFSpawner] Gz spawn exception for {model_name}: {e}")

    def _spawn(self, model_name, sdf_path, pose):
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        # Gz Harmonic path
        if self._use_gz:
            if not os.path.exists(sdf_path):
                print(f"[SDFSpawner] SDF not found: {sdf_path}")
                return
            try:
                sdf_content = open(sdf_path).read()
            except Exception as e:
                print(f"[SDFSpawner] Cannot read {sdf_path}: {e}")
                return
            self._spawn_gz(model_name, sdf_content, x, y, z)
            return

        # Gazebo Classic path
        if not _GAZEBO_MSGS or self.spawn_client is None:
            print(f"[SDFSpawner] skipping spawn of {model_name} (no simulator)")
            return
        try:
            req = SpawnModel.Request()
            req.model_name = model_name
            req.model_xml = open(sdf_path, "r").read()
            req.robot_namespace = ""
            req.initial_pose = pose
            req.reference_frame = "world"
            future = self.spawn_client.call_async(req)
            while not future.done():
                time.sleep(0.01)
        except Exception as e:
            print("Service call failed:", e)

    def spawn_box(self, pos, id):
        pos = pos - np.array([0.0, 0.0, 0.024])
        self._spawn(f"box_{id}", self.model_path + "box.sdf", self._make_pose(pos))

    def spawn_sized_box(self, pos, id, size_name):
        pos = pos - np.array([0.0, 0.0, 0.024])
        self._spawn(f"box_{size_name}_{id}",
                    self.model_path + f"box_{size_name}.sdf",
                    self._make_pose(pos))

    def spawn_named_model(self, pos, id, model_file):
        self._spawn(f"{model_file}_{id}",
                    self.model_path + f"{model_file}.sdf",
                    self._make_pose(pos))

    def spawn_bar(self, pos, id):
        pos = pos - np.array([0.0, 0.0, 0.024])
        self._spawn(f"bar_{id}", self.model_path + "bar.sdf", self._make_pose(pos))

    def delete_box(self):
        if not _GAZEBO_MSGS or self.delete_client is None:
            print("[SDFSpawner] delete_box skipped (no Gazebo Classic client)")
            return
        if not self.delete_client.wait_for_service(timeout_sec=5.0):
            print("[SDFSpawner] delete_model service not available — skipping delete.")
            return
        try:
            req = DeleteModel.Request()
            req.model_name = "box"
            future = self.delete_client.call_async(req)
            while not future.done():
                time.sleep(0.01)
        except Exception as e:
            print("delete box failed:", e)
