from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import imageio
from wrappers import make_env

env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
env = VecFrameStack(env, n_stack=4)

model = PPO.load("../models/mario_ppo_v1")

obs = env.reset()
writer = imageio.get_writer("../videos/mario_ppo_v1.mp4", fps=60)

num_steps = 1800  # 30 segundos de conteúdo a 60fps

for step in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # env.render(mode="rgb_array") não funciona direto em VecEnv com render_mode=None,
    # então pegamos o frame cru do ambiente original por baixo do wrapper.
    frame = env.get_images()[0]
    writer.append_data(frame)

    if done[0]:
        obs = env.reset()

writer.close()
env.close()
print("Vídeo salvo em videos/mario_ppo_v1.mp4")