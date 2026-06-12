import rospy
import rospkg
import numpy as np

from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler


class SDFSpawner:
    def __init__(self, model_name="box"):
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.model_name = model_name
        rospack = rospkg.RosPack()
        self.model_path = rospack.get_path("simulation_environment") + "/sdfs/"

    def spawn_box(self, pos, id):
        pos = pos - np.array([0.0, 0.0, 0.024])
        box_pose = Pose(
            position=Point(*pos),
            orientation=Quaternion(*quaternion_from_euler(0, 0, 0)),
        )
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            spawner(
                model_name=f"box_{id}",
                model_xml=open(self.model_path + "box.sdf", "r").read(),
                robot_namespace="",
                initial_pose=box_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def spawn_sized_box(self, pos, id, size_name):
        """
        Spawn an occluder box of a given size from a dedicated SDF file.
        size_name in {"small", "medium", "large"} -> box_<size_name>.sdf,
        which must live in simulation_environment/sdfs/ alongside box.sdf.

        Used to build a monotonic occlusion-difficulty gradient (easy/hard/
        extreme) where the box size — not just its position — increases the
        fraction of the ROI hidden from the camera.
        """
        pos = pos - np.array([0.0, 0.0, 0.024])
        box_pose = Pose(
            position=Point(*pos),
            orientation=Quaternion(*quaternion_from_euler(0, 0, 0)),
        )
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            spawner(
                model_name=f"box_{size_name}_{id}",
                model_xml=open(self.model_path + f"box_{size_name}.sdf", "r").read(),
                robot_namespace="",
                initial_pose=box_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def spawn_named_model(self, pos, id, model_file):
        """
        Spawn any occluder defined by an SDF file (model_file without the
        .sdf extension), placed at world position pos. General-purpose helper
        used to build structured occluders (half-enclosure, tunnel) from thin
        panel primitives. Each panel SDF must live in
        simulation_environment/sdfs/ alongside box.sdf.

        Note: spawn_box applies a -0.024 m Z pivot correction for the legacy
        box model. The panel SDFs are authored with their geometric centre at
        the origin, so NO pivot offset is applied here — pos is the panel
        centre in world coordinates.
        """
        box_pose = Pose(
            position=Point(*pos),
            orientation=Quaternion(*quaternion_from_euler(0, 0, 0)),
        )
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            spawner(
                model_name=f"{model_file}_{id}",
                model_xml=open(self.model_path + f"{model_file}.sdf", "r").read(),
                robot_namespace="",
                initial_pose=box_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def spawn_bar(self, pos, id):
        pos = pos - np.array([0.0, 0.0, 0.024])
        box_pose = Pose(
            position=Point(*pos),
            orientation=Quaternion(*quaternion_from_euler(0, 0, 0)),
        )
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            spawner(
                model_name=f"bar_{id}",
                model_xml=open(self.model_path + "bar.sdf", "r").read(),
                robot_namespace="",
                initial_pose=box_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def delete_box(self):
        rospy.wait_for_service("gazebo/delete_model")
        delete_model_service = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        try:
            delete_model_service(model_name="box")
        except Exception as e:
            print("delete box failed")
