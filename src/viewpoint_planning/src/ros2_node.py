"""
Shared ROS 2 node context.

Call ros2_node.init() once at program start (in __main__).
All other modules call ros2_node.get_node() to access the singleton node.
The node spins in a background thread so subscriptions and service callbacks
are processed without blocking the planning loop.
"""
import signal
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

_node: Node = None
_executor: MultiThreadedExecutor = None
_spin_thread: threading.Thread = None


def _spin_loop():
    """Spin the executor; log if it exits unexpectedly."""
    global _spin_thread
    try:
        _executor.spin()
    except Exception as e:
        print(f"[ros2_node] Spin thread exited with error: {e}")
    else:
        if rclpy.ok():
            print("[ros2_node] WARNING: spin thread exited while context is still ok")


def init(node_name: str = "rh_node") -> Node:
    global _node, _executor, _spin_thread
    if not rclpy.ok():
        rclpy.init()

    # Override rclpy's SIGINT handler so the context stays alive during long
    # GPU computations.  We still raise KeyboardInterrupt so the user can Ctrl+C.
    def _sigint_keep_alive(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_keep_alive)

    _node = Node(node_name)
    _executor = MultiThreadedExecutor()
    _executor.add_node(_node)
    _spin_thread = threading.Thread(target=_spin_loop, daemon=True)
    _spin_thread.start()
    return _node


def get_node() -> Node:
    if _node is None:
        raise RuntimeError(
            "ros2_node not initialized — call ros2_node.init() before creating "
            "any ROS 2 objects."
        )
    return _node


def shutdown() -> None:
    global _executor, _spin_thread, _node
    if _executor is not None:
        _executor.shutdown(timeout_sec=1.0)
    if _node is not None:
        _node.destroy_node()
        _node = None
    if rclpy.ok():
        rclpy.shutdown()
