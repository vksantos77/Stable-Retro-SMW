import stable_retro as retro
import imageio
import time

env = retro.make(game='SuperMarioWorld-Snes-v0', render_mode=None)
obs, info = env.reset()

writer = imageio.get_writer('mario_random_5min.mp4', fps=60)

start_time = time.time()
duration_seconds = 5 * 60  # 5 minutos
total_reward = 0
step_count = 0

while time.time() - start_time < duration_seconds:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    writer.append_data(obs)
    step_count += 1

    if terminated or truncated:
        obs, info = env.reset()

    if step_count % 600 == 0:  # a cada ~10s de jogo
        elapsed = time.time() - start_time
        print(f"[{elapsed:.0f}s] Steps: {step_count} | Recompensa acumulada: {total_reward:.1f}")

writer.close()
env.close()

print(f"\nFinalizado! Total de steps: {step_count}")
print(f"Recompensa acumulada final: {total_reward:.1f}")
print("Vídeo salvo como mario_random_5min.mp4")