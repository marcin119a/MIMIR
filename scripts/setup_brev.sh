#!/bin/bash                                                                                          
set -euo pipefail                                                                                    
                                                                                                     
# ── Unsloth installer — macOS · Linux · WSL ──────────────────────────────────                      
UNSLOTH_ENV="${UNSLOTH_ENV:-unsloth}"                                                                
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"                                                             
HOST="127.0.0.1"                                                                                     
PORT="${UNSLOTH_PORT:-8888}"
NGINX_PORT="${NGINX_PORT:-80}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[unsloth]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
die()   { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# ── Detect OS / environment ───────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        *) die "Unsupported OS: $(uname -s)" ;;
    esac
}

OS=$(detect_os)
info "Detected OS: $OS"

# ── Detect CUDA ───────────────────────────────────────────────────────────────
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
elif command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
fi

if [[ -n "$CUDA_VERSION" ]]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    ok "NVIDIA GPU detected — CUDA $CUDA_VERSION"
else
    warn "No NVIDIA GPU detected — installing CPU-only build"
    CUDA_MAJOR=""
fi

# ── Ensure conda / mamba ──────────────────────────────────────────────────────
ensure_conda() {
    if command -v mamba &>/dev/null; then
        CONDA_CMD="mamba"; return
    fi
    if command -v conda &>/dev/null; then
        CONDA_CMD="conda"; return
    fi

    info "Installing Miniforge3..."
    local installer
    case "$OS" in
        macos)
            local arch
            arch=$(uname -m)
            installer="Miniforge3-MacOSX-${arch}.sh"
            ;;
        linux|wsl)
            installer="Miniforge3-Linux-x86_64.sh"
            ;;
    esac

    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/${installer}" \
        -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge.sh

    export PATH="$HOME/miniforge3/bin:$PATH"

    CONDA_CMD="mamba"
    ok "Miniforge3 installed"
}

ensure_conda

# Initialize conda for this non-interactive shell
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── Create / activate conda env ───────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$UNSLOTH_ENV"; then
    info "Conda env '$UNSLOTH_ENV' already exists — skipping creation"
else
    info "Creating conda env '$UNSLOTH_ENV' (Python $PYTHON_VERSION)..."
    $CONDA_CMD create -n "$UNSLOTH_ENV" python="$PYTHON_VERSION" -y
fi

conda activate "$UNSLOTH_ENV"

# ── Install PyTorch ───────────────────────────────────────────────────────────
info "Installing PyTorch..."
if [[ -n "$CUDA_MAJOR" ]]; then
    case "$CUDA_MAJOR" in
        12) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        11) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)  TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    esac
    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" -q
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi
ok "PyTorch installed"

# ── Install Unsloth ───────────────────────────────────────────────────────────
info "Installing Unsloth..."
if [[ -n "$CUDA_MAJOR" ]]; then
    pip install "unsloth[cu${CUDA_MAJOR}xx-torch260]" -q 2>/dev/null \
        || pip install "unsloth[colab-new]" -q
else
    pip install "unsloth[cpu]" -q 2>/dev/null \
        || pip install unsloth -q
fi
ok "Unsloth installed"

# ── Install Unsloth Studio (Jupyter + UI) ────────────────────────────────────
info "Installing Unsloth Studio..."
pip install unsloth-studio -q 2>/dev/null || pip install jupyterlab ipywidgets -q
ok "Unsloth Studio installed"

# ── Install & configure nginx ─────────────────────────────────────────────────
install_nginx() {
    if command -v nginx &>/dev/null; then
        ok "nginx already installed"
        return
    fi
    info "Installing nginx..."
    case "$OS" in
        macos) brew install nginx -q ;;
        linux|wsl)
            if command -v apt-get &>/dev/null; then
                sudo apt-get install -y -qq nginx
            elif command -v yum &>/dev/null; then
                sudo yum install -y -q nginx
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y -q nginx
            else
                die "Cannot install nginx — unknown package manager"
            fi
            ;;
    esac
    ok "nginx installed"
}

configure_nginx() {
    local conf_dir
    case "$OS" in
        macos) conf_dir="$(brew --prefix)/etc/nginx/servers" ;;
        *)     conf_dir="/etc/nginx/conf.d" ;;
    esac

    local conf_file="${conf_dir}/unsloth.conf"
    info "Writing nginx config to ${conf_file}..."

    sudo tee "$conf_file" > /dev/null <<NGINX
server {
    listen ${NGINX_PORT};
    listen [::]:${NGINX_PORT};

    # WebSocket support (required by Jupyter)
    location / {
        proxy_pass         http://${HOST}:${PORT};
        proxy_http_version 1.1;
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
    }
}
NGINX

    ok "nginx config written"
}

start_nginx() {
    info "Starting nginx..."
    sudo nginx -t || die "nginx configuration test failed"
    case "$OS" in
        macos)
            brew services restart nginx 2>/dev/null \
                || nginx -s reload 2>/dev/null \
                || nginx
            ;;
        linux|wsl)
            if sudo systemctl restart nginx 2>/dev/null; then
                :
            elif sudo service nginx restart 2>/dev/null; then
                :
            else
                sudo nginx
            fi
            ;;
    esac
    ok "nginx running on port ${NGINX_PORT}"
}

install_nginx
configure_nginx

# Remove default nginx site that conflicts on port 80
sudo rm -f /etc/nginx/sites-enabled/default

start_nginx

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ "$NGINX_PORT" == "80" ]]; then
    ACCESS_URL="http://localhost"
else
    ACCESS_URL="http://localhost:${NGINX_PORT}"
fi

echo ""
ok "Installation complete!"
echo ""
info "Unsloth Studio available at ${ACCESS_URL}"
echo ""

# ── Create systemd service for persistent Jupyter ─────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")
JUPYTER_BIN="${CONDA_BASE}/envs/${UNSLOTH_ENV}/bin/jupyter"
UNSLOTH_BIN="${CONDA_BASE}/envs/${UNSLOTH_ENV}/bin/unsloth"

if command -v unsloth &>/dev/null && unsloth --help 2>&1 | grep -q studio; then
    LAUNCH_CMD="${UNSLOTH_BIN} studio -H ${HOST} -p ${PORT}"
else
    LAUNCH_CMD="${JUPYTER_BIN} lab --ip=${HOST} --port=${PORT} --no-browser --NotebookApp.token='' --NotebookApp.password=''"
fi

info "Creating systemd service for Unsloth Studio..."
sudo tee /etc/systemd/system/unsloth-studio.service > /dev/null <<SERVICE
[Unit]
Description=Unsloth Studio (JupyterLab)
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}
Environment=PATH=${CONDA_BASE}/envs/${UNSLOTH_ENV}/bin:${CONDA_BASE}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=${LAUNCH_CMD}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable unsloth-studio
sudo systemctl restart unsloth-studio
ok "Unsloth Studio service started"
