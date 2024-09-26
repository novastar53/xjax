# Exploring Jax

# Getting Started

### Apple Silicon

I had to pin libomp to 11.1.0 to avoid segfaults in pytorch.

```bash
curl -sL -o libomp.rb https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew install ./libomp.rb
```

## Configure Environment

```bash
# Configure environment
source environment.sh
make

# Activate venv
source .venv/bin/activate

# Configure Jupyter 
jupyter lab --generate config

# Open the config file 
vim ~/.jupyter/jupyter_lab_config.py
```
Add or modify the following lines 

```py
c.ServerApp.ip = '0.0.0.0'
c.Serverapp.open_browser = False
c.ServerApp.port = 8888
```

```sh
# Launch jupyter
jupyter lab
```
