from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wrappers import make_env

# 1. Cria o ambiente vetorizado (mesmo rodando só 1 instância)
env = DummyVecEnv([lambda: make_env()])

# 2. Empilha os últimos 4 frames, pra o agente perceber movimento
env = VecFrameStack(env, n_stack=4)

# 3. Cria o modelo PPO com política CNN (adequada pra observação em imagem)
model = PPO("CnnPolicy", env, verbose=1)

# 4. Treina por um número de timesteps (prova de conceito: começamos pequeno)
model.learn(total_timesteps=100_000)

# 5. Salva o modelo treinado, pra poder carregar e testar depois
model.save("../models/mario_ppo_v2")

env.close()
print("Treino finalizado e modelo salvo em models/mario_ppo_v2")