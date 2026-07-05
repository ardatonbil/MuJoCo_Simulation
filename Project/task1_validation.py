import numpy as np
import mujoco
import matplotlib.pyplot as plt
from src.kinematics import (
    forward_kinematics,
    rotation_to_quaternion,
    quaternion_error_angle,
)


def get_mujoco_ground_truth(model, data, q):
    """Set joint configuration and read ground truth from MuJoCo."""
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    site_id = model.site('attachment_site').id
    pos  = data.site_xpos[site_id].copy()
    quat = rotation_to_quaternion(data.site_xmat[site_id].reshape(3, 3))
    return pos, quat


def validate_fk(model, data, n_samples=100, seed=42):
    """
    Compare FK against MuJoCo ground truth over random configurations.
    Returns position errors [m] and orientation errors [rad].
    """
    rng = np.random.default_rng(seed)

    # Sample random configs within joint limits
    lo = model.jnt_range[:6, 0]
    hi = model.jnt_range[:6, 1]

    pos_errors = np.zeros(n_samples)
    ori_errors = np.zeros(n_samples)

    for i in range(n_samples):
        q = rng.uniform(lo, hi)

        # Ground truth from MuJoCo
        pos_gt, quat_gt = get_mujoco_ground_truth(model, data, q)

        # Your FK
        T        = forward_kinematics(q, model)
        pos_fk   = T[:3, 3]
        quat_fk  = rotation_to_quaternion(T[:3, :3])

        pos_errors[i] = np.linalg.norm(pos_fk - pos_gt)
        ori_errors[i] = quaternion_error_angle(quat_fk, quat_gt)

    return pos_errors, ori_errors


def plot_fk_validation(pos_errors, ori_errors, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(pos_errors, 'o-', markersize=3, linewidth=0.8)
    ax1.set_ylabel('Position error [m]')
    ax1.set_title('FK validation: position error vs MuJoCo ground truth')
    ax1.set_yscale('log')
    ax1.grid(True)

    ax2.plot(np.degrees(ori_errors), 'o-', markersize=3, linewidth=0.8, color='orange')
    ax2.set_ylabel('Orientation error [deg]')
    ax2.set_xlabel('Configuration index')
    ax2.set_title('FK validation: orientation error vs MuJoCo ground truth')
    ax2.set_yscale('log')
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def print_summary(pos_errors, ori_errors):
    print("\n--- FK Validation Summary ---")
    print(f"Position error  [m]   : max={pos_errors.max():.2e}  "
          f"mean={pos_errors.mean():.2e}  min={pos_errors.min():.2e}")
    print(f"Orientation error [deg]: max={np.degrees(ori_errors).max():.2e}  "
          f"mean={np.degrees(ori_errors).mean():.2e}  "
          f"min={np.degrees(ori_errors).min():.2e}")
    if pos_errors.max() < 1e-6:
        print("\n✓ FK is correct — errors are numerical precision only.")
    else:
        print("\n✗ FK has errors — check your transformation chain.")


if __name__ == "__main__":
    # Load position mode model (ground truth comes from position mode)
    xml_path = "z1_files/z1_joint_position_mode/z1_pos.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    pos_errors, ori_errors = validate_fk(model, data, n_samples=100, seed=42)
    print_summary(pos_errors, ori_errors)
    plot_fk_validation(pos_errors, ori_errors, save_path="task1_fk_validation.png")