# Viewer loop와 시간 흐름

질문한 `while viewer.is_running()`는 MuJoCo viewer를 쓸 때 꽤 중요한 패턴이다. 단순히 `while True`로 돌려도 화면은 한동안 보일 수 있지만, 창을 닫거나 프로그램을 멈출 때 문제가 생기기 쉽다.

## `viewer.is_running()`의 뜻

`mujoco.viewer.launch_passive(env.model, env.data)`는 passive viewer를 띄운다. passive viewer는 시뮬레이션을 자동으로 진행하지 않는다. Python 코드가 `env.step()`으로 상태를 바꾸고, `viewer.sync()`로 그 상태를 viewer에 밀어 넣어야 한다.

`viewer.is_running()`은 viewer 창이 아직 살아 있는지 확인하는 함수다.

```python
while viewer.is_running():
    action = scripted_controller(env, info)
    obs, reward, terminated, truncated, info = env.step(action)
    viewer.sync()
    time.sleep(env.dt)
```

이렇게 쓰면 사용자가 viewer 창을 닫았을 때 while loop도 자연스럽게 끝난다.

## 왜 `while True`가 아닌가

`while True`로 쓰면 viewer 창을 닫아도 Python loop가 계속 돌 수 있다.

문제가 되는 지점은 다음과 같다.

- 닫힌 viewer에 계속 `viewer.sync()`를 호출할 수 있다.
- 사용자는 창을 닫았는데 프로세스가 끝나지 않을 수 있다.
- CPU를 계속 쓰는 loop가 남을 수 있다.
- GUI 자원 정리가 깔끔하지 않을 수 있다.

그래서 viewer가 있는 루프에서는 "창이 살아 있는 동안만 반복한다"는 조건을 loop 조건에 넣는다.

## `main.py`의 루프

`main.py`는 사람이 눈으로 보는 데모이므로 viewer가 필수다.

```python
while viewer.is_running():
    action = scripted_controller(env, info)
    obs, reward, terminated, truncated, info = env.step(action)
    viewer.sync()
    time.sleep(env.dt)

    if terminated or truncated:
        ...
        break
```

이 loop에는 종료 조건이 두 종류 있다.

- viewer 창을 닫으면 `viewer.is_running()`이 false가 되어 종료된다.
- episode가 성공하거나 시간 초과되면 `break`로 종료된다.

즉, 사람이 닫아도 끝나고, 환경이 끝났다고 말해도 끝난다.

## `test.py`의 루프는 왜 다르게 생겼나

`test.py`의 `rollout_episode()`는 viewer가 있을 수도 있고 없을 수도 있다.

```python
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if viewer is not None:
        if not viewer.is_running():
            return interrupted result
        viewer.sync()
        time.sleep(env.dt)
```

여기서는 loop 조건이 `viewer.is_running()`이 아니다. 이유는 `--headless` 모드에서는 viewer가 아예 없기 때문이다. viewer가 없을 때도 episode는 돌아야 한다.

그래서 `test.py`는 episode 종료 조건을 loop의 중심으로 둔다.

```python
while not (terminated or truncated):
```

그리고 viewer가 있을 때만 안쪽에서 viewer 상태를 확인한다.

```python
if viewer is not None:
    if not viewer.is_running():
        return interrupted result
```

이 구조 덕분에 같은 평가 코드가 두 모드에서 모두 작동한다.

- `python car_rl/test.py`: viewer를 보면서 평가
- `python car_rl/test.py --headless`: viewer 없이 빠르게 평가

## `train.py`에는 왜 viewer loop가 없는가

학습에서는 viewer를 켜지 않는다. PPO가 빨리 많은 경험을 모아야 하기 때문이다.

`train.py`에서는 이런 코드를 직접 쓰지 않는다.

```python
while ...:
    action = ...
    env.step(action)
```

대신 다음 한 줄이 내부적으로 그 일을 한다.

```python
model.learn(total_timesteps=args.timesteps, callback=callback)
```

Stable-Baselines3의 `learn()` 내부에서는 대략 이런 일이 반복된다.

```text
obs = env.reset()
for timestep in total_timesteps:
    action = policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer에 transition 저장
    episode가 끝나면 env.reset()
    충분히 모이면 PPO update
```

학습 중 viewer를 켜면 화면 동기화와 sleep 때문에 매우 느려진다. 그래서 학습은 headless로 돌리고, 학습이 끝난 모델을 `test.py`에서 viewer로 확인한다.

## `viewer.sync()`의 역할

`env.step()`은 MuJoCo `data`를 바꾼다. 하지만 viewer가 그 변경을 화면에 반영하려면 동기화가 필요하다.

```python
viewer.sync()
```

이 호출은 "지금 Python 쪽 model/data 상태를 viewer 화면에 반영해줘"에 가깝다.

passive viewer에서는 Python 코드가 simulation loop의 주인이다. viewer는 상태를 보여주는 창이고, 언제 한 step 진행할지는 Python loop가 정한다.

## `time.sleep(env.dt)`의 역할

`env.dt`는 `car.xml`의 timestep에서 온다.

```xml
<option gravity="0 0 -9.81" timestep="0.05"/>
```

즉 `env.dt`는 0.05초다. 1초에 20 step 정도다.

```python
time.sleep(env.dt)
```

이 코드는 viewer가 사람이 보기 좋은 속도로 움직이게 한다. 이 sleep이 없으면 Python loop가 가능한 한 빨리 돌아서 차가 순간이동하듯 보이거나 CPU를 많이 쓸 수 있다.

다만 학습할 때는 sleep을 넣으면 안 된다. 학습은 실제 시간처럼 천천히 볼 필요가 없고, 가능한 빠르게 많은 step을 처리하는 것이 좋다.

## 루프를 고르는 기준

viewer가 필수인 데모라면:

```python
while viewer.is_running():
    ...
```

headless와 viewer 모드를 둘 다 지원하는 평가라면:

```python
while not (terminated or truncated):
    ...
    if viewer is not None:
        if not viewer.is_running():
            ...
```

학습이라면:

```python
model.learn(...)
```

이 세 가지를 구분하면 loop 모양이 왜 다른지 훨씬 덜 헷갈린다.

