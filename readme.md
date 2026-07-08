# Mario World RL

Projeto de Reinforcement Learning para treinar um agente jogar Super Mario World (SNES) usando `stable-retro` e `stable-baselines3`.

## Stack

- **Python 3.10**
- **stable-retro** — emulador de SNES com interface Gym para RL
- **gymnasium** — padrão de interface para ambientes de RL
- **stable-baselines3** — implementações de algoritmos de RL (PPO)
- **PyTorch (CPU)** — backend de rede neural
- **OpenCV** — pré-processamento de imagem (grayscale, resize)

## Pré-requisitos

- WSL2 com Ubuntu 24.04 (ou Linux nativo)
- Uma ROM legal de Super Mario World (SNES), formato `.smc`, obtida a partir de um cartucho que você possui. **A ROM não é distribuída neste repositório** por questões de direitos autorais.

## Setup do zero

### 1. Instalar Python 3.10

O Ubuntu 24.04 vem com Python 3.12 por padrão. Para instalar o 3.10:

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

### 2. Clonar o repositório e criar o ambiente virtual

```bash
git clone <url-do-seu-repositorio>
cd Stable-Retro
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências de sistema (necessárias para compilar o stable-retro)

```bash
sudo apt install build-essential cmake python3-dev
```

Para renderização em tempo real (`render_mode='human'`), também é necessário:

```bash
sudo apt install libglu1-mesa freeglut3-dev mesa-common-dev
```

### 4. Instalar as dependências Python

```bash
pip install --upgrade pip

# PyTorch CPU-only (evita download gigante da versão CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install stable-baselines3
pip install stable-retro gymnasium opencv-python numpy imageio imageio-ffmpeg
```

> **Nota:** se você tiver GPU NVIDIA configurada com CUDA no WSL, pode instalar a versão completa do PyTorch (`pip install torch`) para acelerar o treino. A versão CPU-only é suficiente para começar.

### 5. Importar a ROM

Coloque o arquivo `.smc` da ROM em uma pasta acessível (ex: `Downloads` do Windows, acessível via `/mnt/c/Users/SEU_USUARIO/Downloads/` no WSL) e rode:

```bash
python -m stable_retro.import "/caminho/para/pasta/com/a/rom/"
```

Saída esperada:
```
Importing SuperMarioWorld-Snes-v0
Imported 1 games
```

> O nome do jogo registrado é `SuperMarioWorld-Snes-v0` (com o sufixo `-v0`).

### 6. Testar o ambiente

Crie um arquivo `test_env.py`:

```python
import stable_retro as retro

env = retro.make(game='SuperMarioWorld-Snes-v0', render_mode=None)
obs, info = env.reset()
print("Ambiente carregado com sucesso!")
print("Formato da observação (imagem):", obs.shape)
print("Ações possíveis:", env.action_space)
env.close()
```

Rode:

```bash
python test_env.py
```

Saída esperada:
```
Ambiente carregado com sucesso!
Formato da observação (imagem): (224, 256, 3)
Ações possíveis: MultiBinary(12)
```

## Troubleshooting

### `FileNotFoundError: No romfiles found for game`

Reimporte a ROM usando o módulo novo diretamente:
```bash
python -m stable_retro.import "/caminho/para/pasta/"
```
Certifique-se de usar `import stable_retro as retro` no script (não `import retro`, que é o pacote antigo e pode apontar para outro diretório de dados).

### `ImportError: Library "GLU" not found`

O WSL não tem suporte gráfico OpenGL por padrão. Use `render_mode=None` para rodar sem interface visual (necessário para treino). Para renderização em tempo real, instale as libs gráficas (passo 3) e configure um servidor X (ex: VcXsrv) no Windows.

### `Imported 0 games`

O hash MD5 da ROM não corresponde ao esperado pela integração do `stable-retro`. Verifique se a ROM é um dump limpo, sem header, da versão USA. Confirme com:
```bash
md5sum "/caminho/para/rom.smc"
```

## Estrutura do projeto

```
Stable-Retro/
├── venv/                  # ambiente virtual (não versionado)
├── test_env.py            # teste básico de carregamento do ambiente
├── test_random.py         # roda ações aleatórias e salva vídeo/gif
├── wrappers.py            # pré-processamento (grayscale, resize, frame stacking)
├── train.py               # script de treino com PPO
├── play.py                # carrega modelo treinado e roda o agente
├── models/                # modelos treinados salvos (não versionado)
└── README.md
```

## Roadmap

- [x] Configurar ambiente WSL2 + Python 3.10
- [x] Instalar stable-retro, stable-baselines3, PyTorch, gymnasium
- [x] Importar ROM e validar carregamento do ambiente
- [x] Testar ações aleatórias
- [ ] Definir/ajustar função de recompensa
- [ ] Pré-processar observações (grayscale, resize, frame stacking)
- [ ] Restringir espaço de ações a combinações úteis
- [ ] Treinar agente com PPO (fase única)
- [ ] Avaliar e iterar hiperparâmetros
- [ ] Generalizar para múltiplas fases

## Licença / Aviso legal

Este repositório não inclui a ROM do jogo. Para rodar o projeto, é necessário possuir uma cópia legal de Super Mario World e extrair a ROM a partir do próprio cartucho.