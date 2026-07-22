from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from wrappers import make_env
import os

MODELS_DIR = "../models"
CHECKPOINTS_DIR = "../models/checkpoints"


def get_next_model_name(base_name="mario_ppo"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(MODELS_DIR) if f.startswith(base_name) and f.endswith(".zip")]

    version_numbers = []
    for filename in existing:
        name_without_ext = filename.replace(".zip", "")
        version_str = name_without_ext.replace(f"{base_name}_v", "")
        if version_str.isdigit():
            version_numbers.append(int(version_str))

    next_version = max(version_numbers, default=0) + 1
    return f"{base_name}_v{next_version}"


model_name = get_next_model_name()

env = DummyVecEnv([lambda: make_env()])
env = VecFrameStack(env, n_stack=4)

model = PPO("CnnPolicy", env, verbose=1, ent_coef=0.01)

checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=CHECKPOINTS_DIR,
    name_prefix=model_name,
)

model.learn(total_timesteps=200_000, callback=checkpoint_callback)

model.save(f"{MODELS_DIR}/{model_name}")

env.close()
print(f"Treino finalizado e modelo salvo em models/{model_name}.zip")
print(f"Checkpoints salvos em {CHECKPOINTS_DIR}/")