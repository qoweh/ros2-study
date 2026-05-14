# CarEnv, MuJoCo, Gymnasium

`car_env.py`는 이 프로젝트에서 제일 중요한 파일이다. MuJoCo 장면을 강화학습 알고리즘이 이해할 수 있는 Gymnasium 환경으로 바꿔주는 어댑터 역할을 한다.

## MuJoCo 쪽: `model`과 `data`

`CarEnv.__init__()`의 처음 부분에서 이 두 객체가 만들어진다.

```python
self.model = mujoco.MjModel.from_xml_path(str(xml_path))
self.data = mujoco.MjData(self.model)
```

`model`은 XML에서 읽은 정적인 구조다.

- body, geom, joint, camera 이름
- timestep 같은 simulation option
- joint의 qpos/qvel 주소
- body hierarchy

`data`는 실행 중 바뀌는 상태다.

- 현재 joint position인 `qpos`
- 현재 joint velocity인 `qvel`
- mocap body 위치와 회전
- 계산된 body pose
- contact, sensor, actuator 관련 runtime 값

간단히 말하면 `model`은 설계도이고, `data`는 현재 상태다.

## XML과 joint 주소 매핑

`car.xml`에는 이런 joint들이 있다.

```xml
<joint name="car_x" type="slide" axis="1 0 0"/>
<joint name="car_y" type="slide" axis="0 1 0"/>
<joint name="car_yaw" type="hinge" axis="0 0 1"/>
```

그리고 바퀴 조향과 회전을 위한 joint도 있다.

`car_env.py`는 joint 이름을 이용해 MuJoCo 내부 주소를 찾는다.

```python
joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
self._qpos_adr[joint_name] = int(self.model.jnt_qposadr[joint_id])
self._qvel_adr[joint_name] = int(self.model.jnt_dofadr[joint_id])
```

이 매핑이 필요한 이유는 `self.data.qpos[...]`, `self.data.qvel[...]`가 단순 배열이기 때문이다. 이름으로 바로 접근하는 것이 아니라, 이름으로 한 번 주소를 찾아두고 그 주소를 계속 쓴다.

## 이 환경은 `mj_step()` 중심이 아니다

이 프로젝트의 차량은 MuJoCo actuator와 contact dynamics로 움직이는 차라기보다, 코드에서 직접 bicycle model 비슷하게 pose를 갱신하는 차다.

`step()` 안에서 직접 계산하는 값은 다음과 같다.

- steering angle
- speed
- x position
- y position
- yaw
- wheel spin

그 후 `_sync_pose()`가 MuJoCo의 `qpos`, `qvel`에 값을 넣고 `mujoco.mj_forward()`를 호출한다.

```python
mujoco.mj_forward(self.model, self.data)
```

`mj_forward()`는 현재 `qpos`, `qvel`을 기준으로 body 위치, geom 위치 같은 파생 값을 다시 계산한다. 그래서 viewer가 새 pose를 제대로 보여줄 수 있다.

만약 motor actuator를 두고 물리엔진이 힘과 토크로 차를 밀게 만들었다면 보통 `mujoco.mj_step()`을 썼을 것이다. 지금 코드는 학습용으로 이해하기 쉽게 "차량 운동학은 Python에서 계산하고, MuJoCo는 시각화와 scene 상태 동기화에 가깝게 사용"하는 형태다.

## Gymnasium 쪽: 환경의 표준 모양

Gymnasium 환경은 보통 아래 속성과 함수를 가진다.

- `action_space`: agent가 낼 수 있는 action 범위
- `observation_space`: agent가 받을 observation 모양과 범위
- `reset()`: episode 시작
- `step(action)`: action 하나를 적용하고 다음 상태 반환
- `close()`: 정리

이 프로젝트의 action space는 2차원이다.

```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

의미는 다음과 같다.

| index | 이름 | 의미 |
| --- | --- | --- |
| `action[0]` | throttle | 양수면 전진 가속, 음수면 감속 또는 후진 |
| `action[1]` | steering command | 양수와 음수로 좌우 조향 |

PPO 입장에서 action은 그냥 `[-1, 1]` 범위의 숫자 2개다. 이 숫자에 실제 차량 의미를 부여하는 것은 `CarEnv.step()`이다.

## Observation 10차원

`get_obs()`는 PPO가 볼 observation을 만든다.

| index | 값 | 의미 |
| --- | --- | --- |
| 0 | local goal x / max_distance | 차량 좌표계 기준 목표의 앞뒤 방향 |
| 1 | local goal y / max_distance | 차량 좌표계 기준 목표의 좌우 방향 |
| 2 | distance / max_distance | 목표까지 거리 |
| 3 | cos(yaw_error) | 목표 방향과 현재 yaw 차이의 cos |
| 4 | sin(yaw_error) | 목표 방향과 현재 yaw 차이의 sin |
| 5 | speed / max_speed | 현재 속도 |
| 6 | steering_angle / max_steer | 현재 조향각 |
| 7 | last_action[0] | 직전 throttle |
| 8 | last_action[1] | 직전 steering command |
| 9 | remaining episode ratio | 남은 시간 비율 |

`yaw_error`를 그냥 각도 하나로 넣지 않고 `cos`, `sin`으로 넣는 이유는 각도의 경계 문제 때문이다. 예를 들어 `+pi`와 `-pi`는 거의 같은 방향인데 숫자로는 크게 떨어져 있다. `cos`, `sin` 표현은 이런 끊김을 줄여준다.

## `reset()` 흐름

`reset()`은 새 episode를 시작한다.

```text
super().reset(seed=seed)
  -> Gymnasium의 랜덤 생성기 준비

mujoco.mj_resetData(model, data)
  -> MuJoCo runtime 상태 초기화

내부 변수 초기화
  -> speed, steering_angle, wheel_spin, step_count, last_action

start_pose 생성
  -> x=0, y=0, yaw는 작은 랜덤값

_sync_pose(...)
  -> MuJoCo data.qpos/qvel에 시작 pose 반영

goal 결정
  -> options로 받은 goal이 있으면 사용
  -> 없으면 random/fixed 설정에 따라 goal 생성

_set_goal(...)
  -> goal mocap body 위치와 방향 갱신

prev_distance, prev_yaw_error 저장
  -> 다음 step에서 progress reward 계산하려고 저장

return obs, info
```

`reset()`이 `obs`만 반환하지 않고 `(obs, info)`를 반환하는 것은 Gymnasium의 새 API 형태다.

## `step(action)` 흐름

`step()`은 agent의 action 하나를 환경에 적용한다.

```text
action을 numpy array로 변환하고 shape 확인
  -> action_space 범위로 clip

이전 거리와 이전 yaw error 저장
  -> progress reward 계산용

action[0]을 throttle로 해석
action[1]을 steering command로 해석

steering command를 실제 steering angle target으로 변환
  -> max_steer_rate로 한 step에 너무 급하게 꺾지 못하게 제한

speed 업데이트
  -> throttle로 가속
  -> drag로 감속
  -> max speed와 reverse speed로 제한

pose 업데이트
  -> x += speed * cos(yaw) * dt
  -> y += speed * sin(yaw) * dt
  -> yaw_rate = speed / wheelbase * tan(steering_angle)
  -> yaw += yaw_rate * dt

_sync_pose(...)
  -> MuJoCo data와 viewer에 반영될 상태 갱신

성공, 이탈, 시간 초과 판단
  -> terminated 또는 truncated 결정

reward 계산
  -> distance progress, yaw progress, action penalty, success bonus 등

return obs, reward, terminated, truncated, info
```

## `terminated`와 `truncated`

Gymnasium에서는 episode 종료 이유를 둘로 나눈다.

`terminated`는 환경의 자연스러운 종료다.

- 성공했다.
- 바깥으로 나갔다.
- 실패 조건을 만났다.

`truncated`는 시간 제한 같은 외부 이유로 잘린 종료다.

- `max_episode_steps`에 도달했다.
- 성공도 실패도 아니지만 더 오래 돌리지 않기로 했다.

이 프로젝트에서는 다음과 같이 결정한다.

```python
terminated = success or out_of_bounds
truncated = self.step_count >= self.max_episode_steps and not terminated
```

평가 루프에서는 둘 중 하나라도 true가 되면 episode를 끝낸다.

```python
while not (terminated or truncated):
    ...
```

## `info`는 무엇인가

`info`는 학습 알고리즘의 필수 입력이라기보다, 사람이 분석하고 디버깅하기 위한 부가 정보다.

이 프로젝트의 `info`에는 다음 값들이 들어 있다.

- `goal`
- `car_pose`
- `distance_to_goal`
- `yaw_error`
- `speed`
- `steering_angle`
- `episode_step`
- `success`
- `position_success`
- `pose_success`
- `out_of_bounds`

`main.py`의 scripted controller는 이 `info`를 적극적으로 사용한다. 반면 PPO 모델은 `obs`만 보고 행동한다. 그래서 `info`는 학습의 직접 입력이 아니라 평가와 디버깅 도구라고 보는 편이 좋다.

