import stable_retro as retro
import imageio
import numpy as np

env = retro.make(game='SuperMarioWorld-Snes-v0', render_mode=None)
obs, info = env.reset()

# Descobre o índice do botão RIGHT dinamicamente (evita depender da ordem fixa)
buttons = env.buttons
print("Botões disponíveis:", buttons)

right_index = buttons.index('RIGHT')

# Monta a ação: todos os botões soltos, exceto RIGHT pressionado
action = np.zeros(env.action_space.n, dtype=np.int8)
action[right_index] = 1

writer = imageio.get_writer('mario_right.mp4', fps=60)

num_steps = 1800  # 30 segundos de conteúdo a 60fps
total_reward = 0

for step in range(num_steps):
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    writer.append_data(obs)

    if terminated or truncated:
        print(f"Episódio terminou no step {step}. Resetando...")
        obs, info = env.reset()

    if step % 300 == 0:
        print(f"Step {step} | Recompensa acumulada: {total_reward:.1f}")

writer.close()
env.close()

print(f"\nFinalizado! Recompensa acumulada: {total_reward:.1f}")
print("Vídeo salvo como mario_right.mp4")