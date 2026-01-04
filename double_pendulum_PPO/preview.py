import mujoco
import mujoco.viewer
   
while True:
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as v:

        # show joints
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        # enable transparency rendering
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
    # close viewer, loop reloads the file