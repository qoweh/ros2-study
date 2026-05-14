# car_rl report

이 디렉토리는 `car_rl`를 처음부터 다시 읽을 수 있게 만든 설명서 모음이다.

`car_rl`는 MuJoCo 장면, Gymnasium 환경, Stable-Baselines3 PPO 학습, 학습된 모델 평가가 한 번에 들어 있는 작은 강화학습 프로젝트다. 처음 보면 파일 수는 적어도 흐름이 살짝 헷갈린다. 특히 `viewer.is_running()`, `terminated`, `truncated`, `model.predict()`, `env.step()` 같은 부분은 "왜 꼭 이렇게 쓰지?"라는 질문이 자연스럽게 생긴다.

## 추천 읽는 순서

1. `01_execution_flow.md`
   - 프로그램이 어떤 순서로 실행되는지 먼저 잡는다.
   - `main.py`, `train.py`, `test.py`가 서로 어떻게 다른지 설명한다.

2. `02_car_env_mujoco_gymnasium.md`
   - `CarEnv`가 MuJoCo와 Gymnasium 사이에서 어떤 역할을 하는지 설명한다.
   - `reset()`, `step()`, `observation`, `action`, `info`를 코드 흐름 기준으로 읽는다.

3. `03_viewer_loop_and_time.md`
   - `while viewer.is_running()`를 왜 쓰는지 따로 정리한다.
   - 렌더링 루프와 학습 루프가 왜 다르게 생겼는지 설명한다.

4. `04_reward_and_task_design.md`
   - 보상 함수가 왜 이렇게 생겼는지 설명한다.
   - 강화학습에서 reward shaping이 왜 중요한지 같이 정리한다.

5. `05_ppo_core.md`
   - PPO의 핵심 원리와 이 프로젝트에서 쓰인 설정을 설명한다.
   - PPO 대신 SAC, TD3, DQN 등을 쓰면 무엇이 달라지는지도 비교한다.

6. `06_training_testing_models.md`
   - 학습, 평가, 저장된 모델, 로그 파일이 어떤 생명주기를 가지는지 정리한다.

7. `07_faq_debugging.md`
   - 자주 헷갈리는 질문을 짧고 직접적으로 모아둔 파일이다.

## 현재 핵심 파일

- `car.xml`: MuJoCo 장면 정의. 바닥, 목표 지점, 차체, 바퀴, 카메라가 들어 있다.
- `car_env.py`: Gymnasium 환경. 강화학습 알고리즘이 직접 상대하는 핵심 파일이다.
- `main.py`: PPO 없이 사람이 짠 규칙 기반 controller로 차를 움직이는 데모다.
- `train.py`: PPO로 정책을 학습한다.
- `test.py`: 저장된 PPO 모델을 불러와 평가한다.
- `car_ppo_nav.zip`: 기본 평가 스크립트가 읽는 학습 완료 모델이다.
- `runs/best_model/best_model.zip`: 학습 중 `EvalCallback`이 저장한 가장 성능이 좋았던 모델이다.

## 제일 짧은 정신 모델

강화학습에서 매 step마다 일어나는 일은 거의 항상 이 모양이다.

```text
obs -> policy/model -> action -> env.step(action) -> next_obs, reward, done, info
```

`car_rl`에서는 여기에 MuJoCo viewer가 붙으면 화면 갱신이 추가된다.

```text
obs -> action -> env.step(action) -> viewer.sync() -> sleep -> repeat
```

학습할 때는 사람이 보는 화면이 필요하지 않으므로 `viewer.sync()`와 `sleep`이 없다. 대신 Stable-Baselines3의 `model.learn()`이 내부에서 빠르게 `env.step()`을 반복한다.

## 바로 실행해보는 명령

프로젝트 루트에서 실행하는 기준이다.

```bash
python car_rl/main.py
```

```bash
python car_rl/test.py --episodes 5 --headless
```

```bash
python car_rl/test.py --model-path car_rl/car_ppo_nav.zip
```

```bash
python car_rl/train.py --timesteps 50000 --eval-freq 10000 --eval-episodes 5
```

