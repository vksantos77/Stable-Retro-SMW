# Roadmap — Mario World RL

Plano de ação para treinar um agente de Reinforcement Learning jogando Super Mario World.

## Status atual

- [x] Ambiente WSL2 + Python 3.10 configurado
- [x] Stack instalada: stable-retro, gymnasium, stable-baselines3, PyTorch (CPU), opencv-python
- [x] ROM importada e validada (hash correto)
- [x] Ambiente testado (`test_env.py` rodando sem erro)
- [x] Teste com ações aleatórias (baseline "burro")
- [x] Script de ação fixa (andar só pra direita) como segundo baseline

---

## Fase 1 — Estrutura do projeto

- [ ] Organizar pastas do projeto:
  ```
  Stable-Retro/
  ├── src/
  │   ├── wrappers.py       # pré-processamento de observação e ações
  │   ├── train.py          # script de treino
  │   ├── play.py           # roda um modelo treinado
  │   └── evaluate.py       # avalia performance de um modelo salvo
  ├── models/                # checkpoints salvos (gitignored)
  ├── logs/                  # logs de treino / tensorboard (gitignored)
  ├── videos/                # gravações de gameplay (gitignored)
  ├── requirements.txt
  ├── .gitignore
  └── README.md
  ```
- [ ] Criar `requirements.txt` com as versões exatas das libs instaladas (`pip freeze > requirements.txt`)
- [ ] Subir o projeto pro GitHub (repositório privado ou público sem a ROM)

## Fase 2 — Pré-processamento do ambiente

- [ ] Aplicar **grayscale** na observação (reduz de 3 canais pra 1)
- [ ] Aplicar **resize** da imagem (ex: 224x256 → 84x84, padrão em RL de jogos Atari/retro)
- [ ] Implementar **frame stacking** (empilhar últimos 4 frames, pra rede perceber movimento/velocidade)
- [ ] Restringir o **espaço de ações**: de `MultiBinary(12)` pra um conjunto discreto de combinações úteis (andar, pular, correr, pular+correr, etc) — acelera muito o aprendizado
- [ ] Testar o ambiente com os wrappers aplicados (confirmar shape da observação e que as ações restritas funcionam)

## Fase 3 — Função de recompensa

- [ ] Investigar a recompensa **default** da integração do jogo no stable-retro (olhar o `data.json`/`scenario.json` da integração)
- [ ] Decidir se a recompensa default serve ou se precisa customizar
- [ ] Se customizar: montar recompensa baseada em avanço de posição X (via RAM), com penalidade por morte e bônus por completar a fase
- [ ] Testar a função de recompensa isoladamente (rodar alguns episódios e conferir se os valores fazem sentido)

## Fase 4 — Treino inicial (prova de conceito)

- [ ] Escolher **uma única fase** simples pra começar (ex: Donut Plains 1 ou Yoshi's Island 1)
- [ ] Configurar o modelo PPO do stable-baselines3 com hiperparâmetros default
- [ ] Rodar um treino curto (ex: 100k timesteps) só pra validar que o pipeline completo funciona ponta a ponta (ambiente → wrappers → recompensa → PPO → checkpoint salvo)
- [ ] Salvar o modelo treinado e testar ele jogando (`play.py`)

## Fase 5 — Monitoramento e iteração

- [ ] Configurar **Tensorboard** pra acompanhar curva de recompensa média ao longo do treino
- [ ] Configurar checkpoints automáticos (salvar modelo a cada N timesteps)
- [ ] Rodar treinos mais longos (ex: 1M+ timesteps) e comparar curvas
- [ ] Ajustar hiperparâmetros (learning rate, batch size, n_steps, gamma) conforme os resultados
- [ ] Documentar o que funcionou e o que não funcionou (vale registrar isso, é ótimo material de portfólio)

## Fase 6 — Generalização

- [ ] Treinar em múltiplas fases (não só uma), pra ver se o agente generaliza ou decora só uma fase
- [ ] Avaliar performance em fases que o agente nunca viu no treino
- [ ] (Opcional) Testar outros algoritmos além do PPO (ex: A2C, DQN) pra comparar resultados

## Fase 7 — Apresentação dos resultados

- [ ] Gravar vídeo comparando: ações aleatórias vs. ação fixa (baseline) vs. agente treinado
- [ ] Atualizar o README com resultados, gráficos de treino, e instruções de como rodar o modelo final
- [ ] (Opcional) Montar um post/artigo curto documentando o processo, bom pra portfólio

---

## Próximo passo imediato

A Fase 2 (pré-processamento) é o próximo bloco de código a escrever — os wrappers de grayscale, resize, frame stacking e restrição de ações. É a base que todo o resto (recompensa, treino) vai usar.