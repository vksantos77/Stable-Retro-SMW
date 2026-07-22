"""
Wrappers de pré-processamento para o ambiente SuperMarioWorld-Snes-v0.

reduz o espaço de ações a combinações úteis e transforma a observação (imagem) para um formato mais leve de processar: grayscale, resize e frame stacking.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converte o espaço de ações MultiBinary(12) em um espaço Discrete pequeno, com combinações de botões úteis para jogar (evita que o agente precise aprender a ignorar sozinho combinações iúteis, como cima+baixo ao mesmo tempo).
    """

    def __init__(self, env):
        super().__init__(env)

        buttons = env.unwrapped.buttons
         # Cada combo é uma lista de nomes de botões pressionados juntos.
        combos = [
            [],                      # não fazer nada
            ["RIGHT"],               # andar pra direita
            ["RIGHT", "A"],          # correr pra direita (A = correr/pegar no SMW)
            ["RIGHT", "B"],          # pular pra direita (B = pulo)
            ["RIGHT", "A", "B"],     # correr e pular pra direita
            ["B"],                   # pular parado
            ["LEFT"],                # andar pra esquerda
            ["LEFT", "B"],           # pular pra esquerda
            ["DOWN"],                # agachar
        ]

        self._actions = []
        for combo in combos:
            action = np.zeros(len(buttons), dtype=np.int8)
            for button_name in combo:
                if button_name in buttons:
                    action[buttons.index(button_name)] = 1
            self._actions.append(action)

        self.action_space = spaces.Discrete(len(self._actions))

    def action(self, act):
        return self._actions[act]

class PreprocessFrameWrapper(gym.ObservationWrapper):
    """
    Converte a observação para grayscale e reduz a solução.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:,:,None]

class XPositionRewardWrapper(gym.Wrapper):
    def __init__(self, env, death_penalty=-15.0, step_penalty=-0.1):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self._prev_x = None
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x = info.get("x_pos", 0)
        self._prev_lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)

        x_pos = info.get("x_pos", self._prev_x)
        lives = info.get("lives", self._prev_lives)

        died = lives < self._prev_lives

        if died:
            reward = self.death_penalty
            terminated = True  # força o fim do episódio na primeira morte
        else:
            reward = (x_pos - self._prev_x) + self.step_penalty

        self._prev_x = x_pos
        self._prev_lives = lives

        return obs, float(reward), terminated, truncated, info
    
    def __init__(self, env, death_penalty=-15.0, step_penalty=-0.1):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self._prev_x = None
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x = info.get("x_pos", 0)
        self._prev_lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)

        x_pos = info.get("x_pos", self._prev_x)
        lives = info.get("lives", self._prev_lives)

        died = lives < self._prev_lives

        if died:
            reward = self.death_penalty
        else:
            reward = (x_pos - self._prev_x) + self.step_penalty

        self._prev_x = x_pos
        self._prev_lives = lives

        return obs, float(reward), terminated, truncated, info
    
def make_env(game="SuperMarioWorld-Snes-v0", state=None, render_mode=None):
    import stable_retro as retro

    retro.data.Integrations.add_custom_path(
        "/home/ghostpunk/projetos/Stable-Retro/custom_integrations"
    )

    kwargs = {
        "game": game,
        "render_mode": render_mode,
        "inttype": retro.data.Integrations.ALL,
    }
    if state is not None:
        kwargs["state"] = state

    env = retro.make(**kwargs)
    env = DiscreteActionWrapper(env)
    env = XPositionRewardWrapper(env)
    env = PreprocessFrameWrapper(env)
    return env
if __name__ == "__main__":
    env = make_env()
    obs,info = env.reset()
    print("Observação (após  wrappers):", obs.shape)
    print("Espaço de ações (após wrappers):", env.action_space)

    action = env.action_space.sample()
    obs,reward, terminated, truncated, info = env.step(action)
    print("Step executado. Recomensa:", reward)

    env.close()
