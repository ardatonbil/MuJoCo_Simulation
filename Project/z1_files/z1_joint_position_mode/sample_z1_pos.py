import os
import numpy as np
import time
import mujoco
import mujoco.viewer


MODEL_PATH = r"scene_pos.xml"

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Number of joints:", model.njnt)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print("joint", i, name)

print("\nNumber of actuators:", model.nu)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print("actuator", i, name)

print("\nNumber of bodies:", model.nbody)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print("body", i, name)

print("\nNumber of sites:", model.nsite)
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print("site", i, name)

site_name = "attachment_site"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

if site_id == -1:
    raise ValueError(f"Site '{site_name}' not found in model.")

print("\nattachment_site id =", site_id)

 # --------------------------------
 # ADD YOUR IMAGINATION HERE
 # --------------------------------

mujoco.mj_forward(model, data)

print("attachment_site position =", data.site_xpos[site_id].copy())

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # --------------------------------
        # ADD YOUR IMAGINATION HERE
        # --------------------------------

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)