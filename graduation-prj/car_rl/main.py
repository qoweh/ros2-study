import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("car.xml") # XML 로드
data = mujoco.MjData(model) # 시뮬레이션 상태
viewer = mujoco.viewer.launch_passive(model, data) # viewer 실행

state = np.hstack([
    data.body("car").xpos,
    data.body("car").cvel,
    data.body("car").xquat
])

start = time.time()
while viewer.is_running():
    
    data.ctrl[0] = 1.0
        
    mujoco.mj_step(model, data) # 물리 시뮬레이션 1 step
    viewer.sync()
    time.sleep(0.01)
    if time.time() - start > 10: break
    
viewer.close()