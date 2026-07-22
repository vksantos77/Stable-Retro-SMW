import stable_retro as retro

# Registra a pasta customizada como fonte adicional de integrações
retro.data.Integrations.add_custom_path(
    "/home/ghostpunk/projetos/Stable-Retro/custom_integrations"
)

env = retro.make(
    game="SuperMarioWorld-Snes-v0",
    inttype=retro.data.Integrations.ALL,
    render_mode=None,
)

obs, info = env.reset()
print("Info no reset:", info)

buttons = env.buttons
right_index = buttons.index("RIGHT")

import numpy as np
action = np.zeros(len(buttons), dtype=np.int8)
action[right_index] = 1

for step in range(60):
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 10 == 0:
        print(f"Step {step} | x_pos: {info.get('x_pos')} | reward: {reward}")

env.close()