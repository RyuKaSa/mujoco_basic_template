import time

import mujoco
import mujoco.viewer

# import model from xml file
model = mujoco.MjModel.from_xml_path("Pendulum.xml")
# Static state which contains this
"""
masses
joint limits
geom shapes
actuator definitions
"""


data = mujoco.MjData(model)
# Dynamic state which contains this
"""
data.qpos         - joint positions (angles/displacements)
data.qvel         - joint velocities
data.ctrl         - actuator commands (RL or code inputs)
data.sensordata   - sensor readings
data.xquat        - body orientations
data.time         - simulation time
"""


# initial angles if not set in xml
data.qpos[0] = 1


# set the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    # loop the viewing window
    while viewer.is_running():

        # computation can go here, to not affect the duration of the step per while iteration
        # basically control logic, setting angles on actuators, etc

        # will need the duration of the physic step
        step_start = time.time()

        # physic step
        mujoco.mj_step(model, data)

        # send the newly computated step into the viewer for rendering
        viewer.sync()

        # compute step duration, and sleep for reminder of time until next step increment
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)

