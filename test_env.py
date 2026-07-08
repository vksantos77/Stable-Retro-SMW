import stable_retro as retro

env = retro.make(game='SuperMarioWorld-Snes-v0', render_mode=None)
obs, info = env.reset()
print("Ambiente carregado com sucesso!")
print("Formato da observação (imagem):", obs.shape)
print("Ações possíveis:", env.action_space)
env.close()