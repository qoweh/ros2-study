# FAQ와 디버깅 메모

헷갈리기 쉬운 질문을 바로 답할 수 있게 모아둔 파일이다.

## 왜 `while viewer.is_running()`를 쓰나

viewer 창이 살아 있는 동안만 루프를 돌리기 위해서다.

`while True`로 쓰면 창을 닫아도 Python loop가 계속 돌 수 있다. `viewer.is_running()`을 조건으로 쓰면 사용자가 창을 닫았을 때 자연스럽게 종료된다.

## 왜 `test.py`는 `while not (terminated or truncated)`를 쓰나

`test.py`는 viewer가 없는 `--headless` 모드도 지원하기 때문이다. viewer가 없을 때는 `viewer.is_running()`을 쓸 수 없다.

그래서 episode가 끝났는지를 중심 조건으로 둔다.

```python
while not (terminated or truncated):
```

viewer가 있을 때만 loop 내부에서 viewer 상태를 따로 확인한다.

## `terminated`와 `truncated`는 뭐가 다른가

`terminated`는 환경 자체의 종료다.

- 성공
- 실패
- 바깥 이탈

`truncated`는 시간 제한 때문에 끊긴 종료다.

- 최대 step 수 도달

이 프로젝트에서는 성공하거나 arena 밖으로 나가면 `terminated`, 시간이 다 되면 `truncated`다.

## 왜 `obs, info = env.reset()`인가

Gymnasium의 새 API가 reset에서 observation과 info를 같이 반환하기 때문이다.

```python
obs, info = env.reset()
```

예전 Gym 코드에서는 `obs = env.reset()` 형태도 많다. 자료를 보다 보면 둘이 섞여 있어서 헷갈릴 수 있다.

## 왜 `env.step()`은 5개를 반환하나

Gymnasium API는 step에서 5개를 반환한다.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

예전 Gym은 `obs, reward, done, info` 4개를 반환했다. 지금은 `done`을 `terminated`와 `truncated`로 나눴다고 보면 된다.

## 왜 `action, _ = model.predict(...)`에서 `_`가 있나

Stable-Baselines3의 `predict()`는 action과 추가 state를 반환한다.

```python
action, state = model.predict(obs)
```

현재 모델은 recurrent policy가 아니므로 추가 state를 쓰지 않는다. 그래서 관례적으로 `_`에 받는다.

## 왜 학습할 때 viewer를 안 켜나

학습은 많은 step을 빠르게 처리해야 한다. viewer를 켜고 `viewer.sync()`와 `time.sleep()`을 넣으면 학습 속도가 크게 느려진다.

그래서 보통:

- 학습: viewer 없이 headless
- 평가와 디버깅: viewer 사용

이렇게 나눈다.

## 왜 `time.sleep(env.dt)`가 있나

viewer에서 사람이 보기 좋은 속도로 맞추기 위해서다.

`env.dt`는 `car.xml`의 timestep에서 온다.

```xml
<option gravity="0 0 -9.81" timestep="0.05"/>
```

0.05초씩 자면 대략 초당 20 step으로 보인다.

## 왜 `mujoco.mj_forward()`를 쓰고 `mj_step()`을 안 쓰나

현재 차량 움직임은 Python 코드에서 직접 계산한다.

```python
x_pos += self.speed * np.cos(yaw) * self.dt
y_pos += self.speed * np.sin(yaw) * self.dt
yaw += yaw_rate * self.dt
```

그 다음 MuJoCo data에 pose를 넣고 `mj_forward()`로 파생 상태를 갱신한다.

즉 지금은 "물리 엔진이 차를 밀어서 움직인다"보다 "Python이 차 pose를 계산하고 MuJoCo가 보여준다"에 가깝다.

## 왜 goal body가 `mocap="true"`인가

goal은 물리적으로 충돌하거나 움직이는 물체가 아니라, 목표 위치와 방향을 보여주는 marker다.

`mocap` body는 코드에서 위치와 회전을 직접 바꾸기 편하다.

```python
self.data.mocap_pos[self._goal_mocap_id] = ...
self.data.mocap_quat[self._goal_mocap_id] = ...
```

그래서 reset마다 목표 marker를 새 위치로 옮길 수 있다.

## 왜 local goal을 observation에 넣나

전역 좌표의 goal과 차량 pose를 따로 넣는 것보다, 차량 기준으로 목표가 어디에 있는지 넣으면 agent가 배우기 쉽다.

차량 입장에서는 중요한 질문이 이것이다.

```text
목표가 내 앞에 있나, 왼쪽에 있나, 오른쪽에 있나?
```

`_goal_local()`은 바로 이 정보를 만든다.

## 왜 yaw를 그냥 넣지 않고 `cos`, `sin`으로 넣나

각도는 `pi`와 `-pi` 근처에서 숫자가 갑자기 튄다.

실제로는 거의 같은 방향인데 숫자로는 큰 차이처럼 보일 수 있다. 그래서 `cos(yaw_error)`, `sin(yaw_error)`로 넣으면 각도 경계의 끊김이 줄어든다.

## 모델이 실패할 때 어디부터 보면 좋나

먼저 평가를 headless로 여러 episode 돌린다.

```bash
python car_rl/test.py --episodes 20 --headless
```

그 다음 화면으로 실패 episode를 본다.

```bash
python car_rl/test.py --episodes 5
```

확인할 순서는 다음이 좋다.

1. 목표 위치까지는 가는가
2. 목표 근처에서 속도를 줄이는가
3. yaw만 못 맞추는가
4. arena 밖으로 자주 나가는가
5. 특정 goal 방향에서만 실패하는가

## reward를 바꿨는데 더 나빠지면

강화학습에서는 흔한 일이다. reward를 바꾸면 agent가 보는 "좋은 행동"의 정의가 바뀐다.

이럴 때는 한 번에 여러 항목을 바꾸기보다 하나씩 바꾸는 게 좋다.

- success bonus만 변경
- near-goal speed penalty만 변경
- yaw reward만 변경
- action penalty만 변경

그리고 같은 seed와 같은 `--timesteps`로 비교해야 원인을 보기 쉽다.

## 처음 공부할 때의 추천 루틴

1. `main.py`로 scripted controller를 본다.
2. `car_env.py`의 `reset()`과 `step()`을 같이 읽는다.
3. `test.py --headless`로 모델 평가 숫자를 본다.
4. `test.py`로 viewer 움직임을 본다.
5. `train.py --timesteps 10000`처럼 짧게 학습을 돌려본다.
6. PPO 문서를 읽고 `n_steps`, `gamma`, `ent_coef`를 하나씩 바꿔본다.

이 순서로 보면 "화면에서 차가 움직인다"와 "PPO가 observation/action/reward를 통해 배운다"가 점점 이어진다.

