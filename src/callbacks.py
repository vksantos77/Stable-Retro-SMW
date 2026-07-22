import os
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wrappers import make_env


class VideoRecorderCallback(BaseCallback):
    """
    A cada `record_freq` timesteps, grava um vídeo curto do agente
    jogando com a política atual, usando um ambiente separado
    (com render_mode='rgb_array') para não afetar a velocidade
    do ambiente de treino principal.
    """

    def __init__(self, record_freq=20_000, video_length=600, videos_dir="../videos/training_progress", verbose=0):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_length = video_length
        self.videos_dir = videos_dir
        os.makedirs(self.videos_dir, exist_ok=True)

        # Ambiente separado, só para gravação (não usado no treino)
        self._eval_env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
        self._eval_env = VecFrameStack(self._eval_env, n_stack=4)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_freq == 0:
            self._record_video()
        return True  # retornar False cancelaria o treino; sempre continuamos

    def _record_video(self):
        video_path = os.path.join(self.videos_dir, f"step_{self.num_timesteps}.mp4")
        writer = imageio.get_writer(video_path, fps=60)

        obs = self._eval_env.reset()
        for _ in range(self.video_length):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self._eval_env.step(action)
            frame = self._eval_env.get_images()[0]
            writer.append_data(frame)
            if done[0]:
                obs = self._eval_env.reset()

        writer.close()
        if self.verbose:
            print(f"[VideoRecorderCallback] Vídeo salvo: {video_path}")

    def _on_training_end(self):
        self._eval_env.close()