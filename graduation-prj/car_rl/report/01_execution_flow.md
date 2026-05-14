# 실행 흐름

이 프로젝트는 파일마다 역할이 꽤 분명하다. 중요한 점은 `car_env.py`가 환경이고, 나머지 파일들은 그 환경을 서로 다른 방식으로 사용하는 진입점이라는 것이다.

## 전체 구조

```text
car.xml
  -> MuJoCo 장면 정의

car_env.py
  -> car.xml을 읽어서 Gymnasium 환경으로 감싼다
  -> reset(), step(), observation, reward, done 조건을 제공한다

main.py
  -> 사람이 직접 짠 scripted_controller로 환경을 움직인다
  -> 학습 없이 viewer에서 동작 확인

train.py
  -> PPO가 CarEnv를 많이 반복 실행하면서 정책을 학습한다
  -> 모델 zip과 평가 로그를 저장한다

test.py
  -> 저장된 PPO 모델을 불러와 여러 episode를 평가한다
  -> viewer를 켜거나, headless로 빠르게 돌릴 수 있다
```

## 1. `main.py`: 학습 없이 동작 감 잡기

`main.py`는 강화학습 모델을 쓰지 않는다. 대신 `scripted_controller()`라는 사람이 짠 규칙으로 차를 움직인다.

실행 흐름은 다음과 같다.

```text
parse_args()
  -> goal 생성
  -> CarEnv(goal_mode="fixed", fixed_goal=goal) 생성
  -> mujoco.viewer.launch_passive(env.model, env.data)로 viewer 생성
  -> env.reset(seed=..., options={"goal": goal})
  -> while viewer.is_running():
       info를 보고 scripted_controller가 action 계산
       env.step(action)
       viewer.sync()
       time.sleep(env.dt)
       성공하거나 시간 초과면 break
  -> viewer.close()
  -> env.close()
```

여기서 핵심은 `scripted_controller()`가 `info`를 본다는 점이다. `info` 안에는 현재 목표, 현재 차량 pose, 목표까지 거리, yaw 오차, 속도 등이 들어 있다. 이 값으로 "목표가 멀면 목표 방향으로 조향하고, 가까우면 목표 yaw에 맞춰 정렬한다"라는 간단한 제어를 한다.

이 파일의 목적은 학습이 아니다. 환경이 제대로 움직이는지, 목표 지점이 보이는지, reward나 종료 조건이 이상하지 않은지 눈으로 확인하는 데 가깝다.

## 2. `train.py`: PPO 학습

`train.py`는 사람이 직접 루프를 돌리는 파일처럼 보이지 않는다. 핵심 반복은 Stable-Baselines3의 `model.learn()` 안에 들어 있다.

실행 흐름은 다음과 같다.

```text
parse_args()
  -> runs 디렉토리 준비
  -> env = Monitor(CarEnv(goal_mode="random"))
  -> eval_env = Monitor(CarEnv(goal_mode="random"))
  -> EvalCallback 생성
  -> PPO("MlpPolicy", env, hyperparameters...) 생성
  -> model.learn(total_timesteps=..., callback=callback)
       내부에서 env.reset()
       내부에서 model이 action 선택
       내부에서 env.step(action) 반복
       일정 step마다 PPO 업데이트
       eval_freq마다 eval_env로 평가
  -> model.save(...)
  -> env.close()
  -> eval_env.close()
```

`train.py`에는 `viewer`가 없다. 학습은 화면을 보는 것보다 빠르게 많은 step을 쌓는 것이 중요하기 때문이다. 화면을 매번 그리면 학습 속도가 크게 느려진다.

`Monitor`는 episode reward, episode length 같은 학습 통계를 Stable-Baselines3가 알기 쉽게 감싸주는 wrapper다. `EvalCallback`은 학습 도중 일정 간격으로 평가를 돌리고, 가장 좋은 모델을 `runs/best_model`에 저장한다.

## 3. `test.py`: 저장된 모델 평가

`test.py`는 `train.py`로 만든 모델을 불러와서 실제로 써보는 파일이다.

실행 흐름은 다음과 같다.

```text
parse_args()
  -> model-path 존재 확인
  -> goal 옵션 해석
  -> CarEnv 생성
  -> PPO.load(model_path)
  -> headless가 아니면 viewer 생성
  -> episodes만큼 반복:
       rollout_episode(...)
       결과 출력
  -> summary 출력
  -> viewer.close()
  -> env.close()
```

`rollout_episode()` 내부는 강화학습 정책을 사용하는 가장 전형적인 평가 루프다.

```text
obs, info = env.reset(...)
terminated = False
truncated = False

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

학습된 모델은 `obs`만 보고 `action`을 낸다. `info`를 직접 보고 조작하는 `main.py`의 scripted controller와 달리, PPO policy는 학습 중 `observation_space`에 들어 있던 10차원 관측값만 사용한다.

## 4. `car_env.py`: 모든 진입점이 공유하는 환경

`CarEnv`는 Gymnasium 환경이다. 강화학습 알고리즘이 기대하는 표준 인터페이스를 제공한다.

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

`main.py`, `train.py`, `test.py`는 모두 결국 이 두 함수를 반복 호출한다.

`CarEnv.__init__()`에서는 다음 작업을 한다.

- `car.xml`을 MuJoCo model로 읽는다.
- `MjData`를 만든다.
- goal mode와 episode 길이를 설정한다.
- action space와 observation space를 정의한다.
- XML에 있는 joint 이름을 찾아 `qpos`, `qvel` 주소로 매핑한다.
- goal body가 mocap body인지 확인한다.

`reset()`은 새 episode를 시작한다.

- MuJoCo data를 초기화한다.
- 속도, 조향각, step count를 초기화한다.
- 차의 시작 pose를 잡는다.
- goal을 직접 받은 값으로 쓰거나 랜덤 샘플링한다.
- goal marker의 위치와 방향을 MuJoCo data에 반영한다.
- 첫 observation과 info를 반환한다.

`step(action)`은 한 tick을 진행한다.

- action을 `[-1, 1]` 범위로 자른다.
- throttle과 steering command로 해석한다.
- steering rate limit을 적용한다.
- 속도, 위치, yaw를 업데이트한다.
- MuJoCo data의 joint 값들을 새 pose에 맞춘다.
- 성공, 바깥 이탈, 시간 초과를 판단한다.
- reward를 계산한다.
- 다음 observation과 info를 반환한다.

## 제일 중요한 차이

`main.py`와 `test.py`에서는 우리가 직접 `while` 루프를 볼 수 있다. 사람이 보기 위한 viewer도 직접 관리한다.

`train.py`에서는 반복문이 `model.learn()` 안으로 들어간다. 그래서 코드에서 `while`이 보이지 않지만, 실제로는 Stable-Baselines3가 내부에서 `env.step()`을 수십만 번 호출하고 있다.

## 실행 목적별 선택

- 환경이 잘 보이는지 확인하고 싶다: `python car_rl/main.py`
- 저장된 모델 성능을 빠르게 숫자로 보고 싶다: `python car_rl/test.py --headless`
- 저장된 모델이 어떻게 움직이는지 보고 싶다: `python car_rl/test.py`
- 새 모델을 학습하고 싶다: `python car_rl/train.py --timesteps 120000`

