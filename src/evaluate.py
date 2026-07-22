import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wrappers import make_env

model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/mario_ppo_v5.zip"
num_episodes = 10
max_steps_per_episode = 3000  # teto de segurança, ~50s de jogo por episódio

env = DummyVecEnv([lambda: make_env()])
env = VecFrameStack(env, n_stack=4)

model = PPO.load(model_path)

max_x_per_episode = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    max_x = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done_array, info = env.step(action)
        done = done_array[0]
        max_x = max(max_x, info[0].get("x_pos", 0))
        steps += 1

        if steps % 200 == 0:
            print(f"  [debug] step {steps} | x_pos: {info[0].get('x_pos')} | lives: {info[0].get('lives')}")

    max_x_per_episode.append(max_x)
    status = "morreu" if done else "atingiu limite de steps (provavelmente preso)"
    print(f"Episódio {episode + 1}: distância máxima = {max_x} ({status}, {steps} steps)")

env.close()

media = sum(max_x_per_episode) / len(max_x_per_episode)
print(f"\nDistância média ao longo de {num_episodes} episódios: {media:.1f}")
print(f"Melhor episódio: {max(max_x_per_episode)}")
print(f"Pior episódio: {min(max_x_per_episode)}")