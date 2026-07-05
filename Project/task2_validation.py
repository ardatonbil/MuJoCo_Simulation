import numpy as np
import mujoco
import matplotlib.pyplot as plt
from src.jacobian import jacobian


def validate_jacobian(model, data, n_trials=20, seed=42):
    rng          = np.random.default_rng(seed)
    dq_magnitude = 1e-5
    n_links      = 6
    error_table  = np.zeros((n_trials, n_links))
    perturb_indices = np.zeros(n_trials, dtype=int)

    for trial in range(n_trials):
        q           = rng.uniform(model.jnt_range[:6, 0], model.jnt_range[:6, 1])
        perturb_idx = int(rng.integers(0, 6))
        perturb_indices[trial] = perturb_idx
        dq          = np.zeros(6)
        dq[perturb_idx] = dq_magnitude
        q_perturbed = q + dq

        for link_idx in range(1, n_links + 1):
            body_name = f'link0{link_idx}'
            body_id   = model.body(body_name).id
            p_body    = model.body_ipos[body_id].copy()

            J            = jacobian(link_idx, p_body, q, model)
            dp_estimated = J[:3, :] @ dq

            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            p_before = (data.xpos[body_id] +
                        data.xmat[body_id].reshape(3, 3) @ p_body)

            data.qpos[:6] = q_perturbed
            mujoco.mj_forward(model, data)
            p_after  = (data.xpos[body_id] +
                        data.xmat[body_id].reshape(3, 3) @ p_body)

            dp_actual = p_after - p_before
            error_table[trial, link_idx - 1] = np.linalg.norm(dp_estimated - dp_actual)

    return error_table, perturb_indices


def plot_error_table(error_table, save_path=None):
    n_trials, n_links = error_table.shape
    link_labels = [f'link0{i+1}' for i in range(n_links)]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(error_table, aspect='auto', cmap='viridis',
                   interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('error norm [m]', fontsize=11)

    ax.set_xticks(range(n_links))
    ax.set_xticklabels(link_labels, fontsize=10)
    ax.set_xlabel('Moving body COM', fontsize=11)
    ax.set_ylabel('Trial', fontsize=11)
    ax.set_title('Jacobian COM displacement prediction error [m]', fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    xml_path = "z1_files/z1_joint_torque_mode/z1_torque.xml"
    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)

    print("Running Jacobian validation (20 trials)...\n")
    error_table, perturb_indices = validate_jacobian(model, data, n_trials=20, seed=42)
    plot_error_table(error_table, save_path="task2_jacobian_validation.png")