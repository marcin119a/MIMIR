#!/bin/bash
set -euo pipefail

# ── Unsloth installer — macOS · Linux · WSL ──────────────────────────────────
UNSLOTH_ENV="${UNSLOTH_ENV:-unsloth}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
HOST="127.0.0.1"
PORT="${UNSLOTH_PORT:-8888}"
NGINX_PORT="${NGINX_PORT:-80}"
VENV_DIR="${HOME}/.venvs/${UNSLOTH_ENV}"

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

# ── Ensure uv ─────────────────────────────────────────────────────────────────
ensure_uv() {
    if command -v uv &>/dev/null; then
        ok "uv already installed"
        return
    fi
    info "Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || die "uv installation failed"
    ok "uv installed"
}

ensure_uv
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# ── Create / activate venv ────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    info "Virtual env '$UNSLOTH_ENV' already exists — skipping creation"
else
    info "Creating virtual env '$UNSLOTH_ENV' (Python $PYTHON_VERSION)..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# ── Install PyTorch ───────────────────────────────────────────────────────────
info "Installing PyTorch..."
if [[ -n "$CUDA_MAJOR" ]]; then
    case "$CUDA_MAJOR" in
        12) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        11) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)  TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    esac
    uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
else
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
ok "PyTorch installed"

# ── Install Unsloth ───────────────────────────────────────────────────────────
info "Installing Unsloth..."
if [[ -n "$CUDA_MAJOR" ]]; then
    uv pip install "unsloth[cu${CUDA_MAJOR}xx-torch260]" 2>/dev/null \
        || uv pip install "unsloth[colab-new]"
else
    uv pip install "unsloth[cpu]" 2>/dev/null \
        || uv pip install unsloth
fi
ok "Unsloth installed"

# ── Install Unsloth Studio (Jupyter + UI) ────────────────────────────────────
info "Installing Unsloth Studio..."
uv pip install unsloth-studio 2>/dev/null || uv pip install jupyterlab ipywidgets
ok "Unsloth Studio installed"

# ── Create the dedicated studio venv that `unsloth studio` expects ────────────
STUDIO_VENV="${HOME}/.unsloth/studio/unsloth_studio"
if [[ ! -x "${STUDIO_VENV}/bin/python" ]]; then
    info "Creating Unsloth Studio venv at ${STUDIO_VENV}..."
    mkdir -p "${HOME}/.unsloth/studio"
    uv venv "${STUDIO_VENV}" --python "${PYTHON_VERSION}"
    ok "Studio venv created"
fi

# Find and run setup.sh bundled with the installed unsloth_cli package
STUDIO_SETUP=$(python -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('studio')
if spec and spec.submodule_search_locations:
    p = pathlib.Path(list(spec.submodule_search_locations)[0]) / 'setup.sh'
    if p.is_file():
        print(p)
" 2>/dev/null || true)

if [[ -n "${STUDIO_SETUP}" ]]; then
    info "Running Unsloth Studio setup..."
    bash "${STUDIO_SETUP}" || warn "Studio setup encountered issues — see above"
    ok "Unsloth Studio setup complete"
else
    warn "studio/setup.sh not found — skipping Studio dependency install"
fi

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
JUPYTER_BIN="${VENV_DIR}/bin/jupyter"
UNSLOTH_BIN="${VENV_DIR}/bin/unsloth"

if [[ -x "${UNSLOTH_BIN}" ]] && "${UNSLOTH_BIN}" --help 2>&1 | grep -q studio; then
    LAUNCH_CMD="${UNSLOTH_BIN} studio -H ${HOST} -p ${PORT}"
elif [[ -x "${JUPYTER_BIN}" ]]; then
    LAUNCH_CMD="${JUPYTER_BIN} lab --ip=${HOST} --port=${PORT} --no-browser --NotebookApp.token='' --NotebookApp.password=''"
else
    die "Neither unsloth nor jupyter found in ${VENV_DIR}/bin — installation incomplete"
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
Environment=PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
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
