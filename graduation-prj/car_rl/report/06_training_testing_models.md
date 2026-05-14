# 학습, 평가, 모델 파일

`car_rl`에는 모델을 만드는 흐름과 모델을 써보는 흐름이 분리되어 있다.

## 모델을 만드는 파일: `train.py`

학습 명령 예시는 다음과 같다.

```bash
python car_rl/train.py --timesteps 120000
```

학습이 끝나면 기본적으로 다음 파일이 생기거나 갱신된다.

```text
car_rl/car_ppo_nav.zip
```

`--model-path`를 바꾸면 다른 위치에 저장할 수 있다.

```bash
python car_rl/train.py --model-path car_rl/my_model --timesteps 50000
```

Stable-Baselines3는 `.zip`을 자동으로 붙여 저장한다. 그래서 위 명령의 결과는 보통 `car_rl/my_model.zip`이다.

## 학습 중 평가: `EvalCallback`

`train.py`에는 `EvalCallback`이 들어 있다.

```python
callback = EvalCallback(
    eval_env,
    best_model_save_path=str(args.log_dir / "best_model"),
    log_path=str(args.log_dir / "eval"),
    eval_freq=args.eval_freq,
    n_eval_episodes=args.eval_episodes,
    deterministic=True,
    render=False,
)
```

이 callback은 학습 중간중간 현재 모델을 평가한다.

- `eval_freq` step마다 평가한다.
- `eval_episodes`개 episode를 돌려 평균 성능을 본다.
- 이전보다 좋은 모델이면 `runs/best_model/best_model.zip`에 저장한다.
- 평가 결과는 `runs/eval` 아래에 저장된다.

최종 모델과 best model은 다를 수 있다.

- 최종 모델: 마지막까지 학습한 모델
- best model: 중간 평가에서 가장 좋았던 모델

강화학습은 마지막 모델이 항상 제일 좋지는 않다. 그래서 best model을 따로 저장하는 것이 중요하다.

## 로그 파일

현재 학습 스크립트는 TensorBoard 로그를 켠다.

```python
tensorboard_log=str(args.log_dir / "tb")
```

기본 위치는 다음이다.

```text
car_rl/runs/tb
```

TensorBoard를 보면 episode reward, loss, value estimate, entropy 같은 학습 곡선을 볼 수 있다.

```bash
tensorboard --logdir car_rl/runs/tb
```

학습이 잘 되고 있는지 볼 때는 단일 episode 결과보다 곡선을 보는 편이 좋다. 강화학습은 episode마다 랜덤성이 있어서 한두 번 성공하거나 실패한 결과만으로 판단하면 흔들린다.

## 모델을 써보는 파일: `test.py`

기본 평가 명령은 다음이다.

```bash
python car_rl/test.py
```

이 명령은 기본 모델을 읽는다.

```text
car_rl/car_ppo_nav.zip
```

화면 없이 빠르게 평가하려면 `--headless`를 붙인다.

```bash
python car_rl/test.py --episodes 20 --headless
```

특정 모델을 평가하려면 `--model-path`를 쓴다.

```bash
python car_rl/test.py --model-path car_rl/runs/best_model/best_model.zip --episodes 20 --headless
```

## 랜덤 goal 평가와 고정 goal 평가

아무 goal 옵션도 주지 않으면 goal은 random이다.

```bash
python car_rl/test.py --episodes 10 --headless
```

특정 goal만 보고 싶으면 세 값을 모두 준다.

```bash
python car_rl/test.py --goal-x 1.8 --goal-y -1.2 --goal-yaw 1.57
```

`goal-x`, `goal-y`, `goal-yaw` 중 하나만 주면 에러가 난다. 세 값이 함께 있어야 하나의 pose goal이 되기 때문이다.

## `model.predict()`의 의미

`test.py`에서 실제 모델 사용은 이 한 줄이다.

```python
action, _ = model.predict(obs, deterministic=True)
```

의미는 다음과 같다.

- 현재 observation을 모델에 넣는다.
- 모델이 action을 예측한다.
- `deterministic=True`이므로 평가 때는 가장 그럴듯한 action을 고른다.

PPO policy는 확률적 policy다. 학습 중에는 여러 action을 샘플링하면서 탐색한다. 평가 때는 보통 deterministic하게 실행해서 흔들림을 줄인다.

반환값의 두 번째 값 `_`는 hidden state 자리다. recurrent policy가 아니면 보통 필요하지 않으므로 `_`로 버린다.

## 평가 출력 읽는 법

`test.py`는 episode마다 다음을 출력한다.

```text
episode= 1 success= True distance= 0.321 yaw_error= 0.12 reward= 25.7 steps= 34
```

중요하게 볼 값은 다음이다.

- `success`: 성공 조건을 만족했는가
- `distance`: 마지막에 목표 위치와 얼마나 가까운가
- `yaw_error`: 마지막에 목표 yaw와 얼마나 차이 나는가
- `reward`: episode 동안 받은 reward 합
- `steps`: 몇 step 만에 끝났는가

마지막 summary는 성공률과 평균 reward, 평균 최종 거리를 보여준다.

```text
summary: success_rate=0.40, mean_reward=53.333, mean_final_distance=0.637
```

`success_rate`가 가장 직관적인 지표이고, `mean_final_distance`는 실패했더라도 목표에 가까이 갔는지 보여준다.

## 학습과 평가는 왜 분리하는가

학습은 빠르게 많이 돌려야 한다.

```text
model.learn(...)
```

평가는 사람이 이해할 수 있게 봐야 한다.

```text
model.predict(obs) -> env.step(action) -> print/render
```

두 목적이 다르기 때문에 파일도 나뉘어 있다.

- `train.py`: 성능을 만들기 위한 파일
- `test.py`: 만들어진 성능을 확인하기 위한 파일
- `main.py`: 학습 전 환경과 controller 감을 확인하기 위한 파일

이 구분이 잡히면 "모델을 만드는 코드"와 "모델을 써먹는 코드"를 헷갈리지 않게 된다.

## 모델 파일을 고를 때

일단 빠르게 확인하려면:

```bash
python car_rl/test.py --model-path car_rl/car_ppo_nav.zip
```

학습 중 가장 좋았던 모델을 보고 싶으면:

```bash
python car_rl/test.py --model-path car_rl/runs/best_model/best_model.zip
```

새 실험을 여러 개 비교하려면 `--model-path`와 `--log-dir`를 실험 이름별로 나누는 것이 좋다.

```bash
python car_rl/train.py \
  --timesteps 120000 \
  --model-path car_rl/experiments/reward_v2/final_model \
  --log-dir car_rl/experiments/reward_v2/runs
```

