import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import imageio
from wrappers import make_env

MODELS_DIR = "../models"
VIDEOS_DIR = "../videos"


def get_latest_model(base_name="mario_ppo"):
    existing = [f for f in os.listdir(MODELS_DIR) if f.startswith(base_name) and f.endswith(".zip")]

    version_numbers = {}
    for filename in existing:
        name_without_ext = filename.replace(".zip", "")
        version_str = name_without_ext.replace(f"{base_name}_v", "")
        if version_str.isdigit():
            version_numbers[int(version_str)] = filename

    if not version_numbers:
        raise FileNotFoundError("Nenhum modelo encontrado em models/")

    latest_version = max(version_numbers.keys())
    return os.path.join(MODELS_DIR, version_numbers[latest_version])


# Se um caminho foi passado por linha de comando, usa ele.
# Senão, cai no comportamento antigo: pega o modelo mais recente.
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = get_latest_model()

model_name = os.path.splitext(os.path.basename(model_path))[0]
print(f"Carregando modelo: {model_path}")

os.makedirs(VIDEOS_DIR, exist_ok=True)

env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
env = VecFrameStack(env, n_stack=4)

model = PPO.load(model_path)

obs = env.reset()
writer = imageio.get_writer(f"{VIDEOS_DIR}/{model_name}.mp4", fps=60)

num_steps = 1800

for step in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame = env.get_images()[0]
    writer.append_data(frame)

    if done[0]:
        obs = env.reset()

writer.close()
env.close()
print(f"Vídeo salvo em videos/{model_name}.mp4")