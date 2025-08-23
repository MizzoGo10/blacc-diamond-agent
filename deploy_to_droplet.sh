#!/bin/bash

# 🤖 Ultimate Architect Agent - One-Click Droplet Deployment
# ================================================================
# This script deploys the complete Ultimate Architect system to your Digital Ocean droplet
# with intelligent RPC caching, rate limiting, and multi-provider failover

set -e  # Exit on any error

echo "🤖 Ultimate Architect Agent - Droplet Deployment Starting..."
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/opt/app"
SERVICE_NAME="ultimate-architect"
USER="ubuntu"
RUST_VERSION="1.75.0"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    print_error "This script should not be run as root. Please run as ubuntu user with sudo privileges."
    exit 1
fi

print_header "🚀 Step 1: System Update and Dependencies"
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_status "Installing essential packages..."
sudo apt install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    nginx \
    htop \
    unzip \
    jq \
    screen \
    tmux

print_header "🦀 Step 2: Rust Installation"
if ! command -v rustc &> /dev/null; then
    print_status "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
    source ~/.cargo/env
    rustup component add clippy rustfmt
else
    print_success "Rust already installed"
    rustc --version
fi

print_header "⚡ Step 3: Solana CLI Installation"
if ! command -v solana &> /dev/null; then
    print_status "Installing Solana CLI..."
    sh -c "$(curl -sSfL https://release.solana.com/stable/install)"
    export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
    echo 'export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"' >> ~/.bashrc
else
    print_success "Solana CLI already installed"
    solana --version
fi

print_header "📁 Step 4: Application Directory Setup"
print_status "Creating application directory..."
sudo mkdir -p $APP_DIR/{src,config,logs,wallets,backups,uploads,models}
sudo chown -R $USER:$USER $APP_DIR
chmod 755 $APP_DIR

print_status "Copying application files..."
cp -r src/* $APP_DIR/src/
cp -r web $APP_DIR/
cp .env.production $APP_DIR/.env
cp Cargo.toml $APP_DIR/

print_header "🔧 Step 5: Building Ultimate Architect Agent"
print_status "Building Rust application..."
cd $APP_DIR

# Create Cargo.toml if not exists
if [ ! -f "Cargo.toml" ]; then
    cat > Cargo.toml << 'EOF'
[package]
name = "ultimate-architect-agent"
version = "2.0.0"
edition = "2021"
authors = ["Ultimate Architect <architect@blacc-diamond.ai>"]
description = "Supreme AI Engineer and System Architect"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
axum = { version = "0.7", features = ["ws", "multipart"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "fs"] }
reqwest = { version = "0.11", features = ["json"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
dotenv = "0.15"
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "sqlite", "chrono", "uuid"] }
futures-util = "0.3"
rand = "0.8"
base64 = "0.21"
sha2 = "0.10"
ed25519-dalek = "2.0"
solana-sdk = "1.17"
solana-client = "1.17"
solana-account-decoder = "1.17"
anchor-lang = "0.29"
spl-token = "4.0"
mpl-token-metadata = "4.0"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true

[profile.dev]
opt-level = 1
debug = true
EOF
fi

# Build the application
print_status "Compiling with optimizations..."
source ~/.cargo/env
cargo build --release

print_success "Build completed successfully!"

print_header "💰 Step 6: Wallet Generation"
print_status "Generating secure wallets..."

# Create wallet generation script
cat > generate_wallets.sh << 'EOF'
#!/bin/bash
WALLET_DIR="/opt/app/wallets"
mkdir -p $WALLET_DIR

echo "🔐 Generating 5 secure wallets..."

for i in {1..5}; do
    echo "Generating wallet $i..."
    
    # Generate keypair
    solana-keygen new --no-bip39-passphrase --silent --outfile "$WALLET_DIR/wallet_${i}.json"
    
    # Get public key
    PUBKEY=$(solana-keygen pubkey "$WALLET_DIR/wallet_${i}.json")
    
    # Create secure wallet info file
    cat > "$WALLET_DIR/wallet_${i}_info.json" << EOL
{
    "wallet_id": $i,
    "public_key": "$PUBKEY",
    "private_key_file": "wallet_${i}.json",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "balance_sol": 0,
    "purpose": "Trading Wallet $i"
}
EOL
    
    echo "✅ Wallet $i: $PUBKEY"
done

echo ""
echo "🎯 FUND THESE WALLETS TO START TRADING:"
echo "========================================"
for i in {1..5}; do
    PUBKEY=$(solana-keygen pubkey "$WALLET_DIR/wallet_${i}.json")
    echo "Wallet $i: $PUBKEY"
done
echo ""
echo "💡 Send SOL to these addresses using Phantom, Solflare, or any Solana wallet"
echo "💡 Recommended: 1-10 SOL per wallet for optimal trading"
EOF

chmod +x generate_wallets.sh
./generate_wallets.sh

print_header "🌐 Step 7: Nginx Configuration"
print_status "Configuring Nginx reverse proxy..."

sudo tee /etc/nginx/sites-available/ultimate-architect << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Dashboard
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
    
    # WebSocket for real-time updates
    location /ws {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Console Interface
    location /console {
        proxy_pass http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # File Upload
    location /upload {
        proxy_pass http://127.0.0.1:8082;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        client_max_body_size 100M;
    }
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/ultimate-architect /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

print_header "🔧 Step 8: Systemd Service Configuration"
print_status "Creating systemd service..."

sudo tee /etc/systemd/system/ultimate-architect.service << EOF
[Unit]
Description=Ultimate Architect Agent - Supreme AI Engineer
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$APP_DIR
Environment=PATH=/home/$USER/.cargo/bin:/home/$USER/.local/share/solana/install/active_release/bin:/usr/local/bin:/usr/bin:/bin
Environment=RUST_LOG=info
ExecStart=$APP_DIR/target/release/ultimate-architect-agent
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ultimate-architect

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

print_header "📊 Step 9: Monitoring Setup"
print_status "Setting up system monitoring..."

# Create monitoring script
cat > $APP_DIR/monitor.sh << 'EOF'
#!/bin/bash

echo "🤖 Ultimate Architect Agent - System Status"
echo "==========================================="
echo ""

# Service status
echo "📊 Service Status:"
systemctl is-active ultimate-architect && echo "✅ Ultimate Architect: RUNNING" || echo "❌ Ultimate Architect: STOPPED"
systemctl is-active nginx && echo "✅ Nginx: RUNNING" || echo "❌ Nginx: STOPPED"
echo ""

# System resources
echo "💻 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s", $5}')"
echo ""

# Network status
echo "🌐 Network Status:"
curl -s -o /dev/null -w "Quicknode RPC: %{http_code} (%{time_total}s)\n" https://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/
curl -s -o /dev/null -w "Alchemy RPC: %{http_code} (%{time_total}s)\n" https://solana-mainnet.g.alchemy.com/v2/PPQbbM4WmrX_82GOP8QR5pJ_JsBvyLWR
echo ""

# Wallet balances
echo "💰 Wallet Balances:"
for i in {1..5}; do
    if [ -f "/opt/app/wallets/wallet_${i}.json" ]; then
        PUBKEY=$(solana-keygen pubkey "/opt/app/wallets/wallet_${i}.json")
        BALANCE=$(solana balance $PUBKEY 2>/dev/null || echo "0")
        echo "Wallet $i ($PUBKEY): $BALANCE"
    fi
done
echo ""

# Recent logs
echo "📝 Recent Logs (last 10 lines):"
journalctl -u ultimate-architect -n 10 --no-pager
EOF

chmod +x $APP_DIR/monitor.sh

print_header "🚀 Step 10: Service Startup"
print_status "Starting Ultimate Architect Agent..."

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable ultimate-architect
sudo systemctl start ultimate-architect

# Wait a moment for startup
sleep 5

# Check service status
if sudo systemctl is-active --quiet ultimate-architect; then
    print_success "Ultimate Architect Agent is running!"
else
    print_error "Service failed to start. Checking logs..."
    sudo journalctl -u ultimate-architect -n 20 --no-pager
fi

print_header "🎯 Step 11: Final Configuration"
print_status "Setting up management commands..."

# Create management script
cat > $APP_DIR/commands.sh << 'EOF'
#!/bin/bash

case "$1" in
    start)
        echo "🚀 Starting Ultimate Architect Agent..."
        sudo systemctl start ultimate-architect
        ;;
    stop)
        echo "🛑 Stopping Ultimate Architect Agent..."
        sudo systemctl stop ultimate-architect
        ;;
    restart)
        echo "🔄 Restarting Ultimate Architect Agent..."
        sudo systemctl restart ultimate-architect
        ;;
    status)
        echo "📊 Ultimate Architect Agent Status:"
        sudo systemctl status ultimate-architect
        ;;
    logs)
        echo "📝 Live logs (Ctrl+C to exit):"
        sudo journalctl -u ultimate-architect -f
        ;;
    monitor)
        ./monitor.sh
        ;;
    wallets)
        echo "💰 Wallet Addresses:"
        for i in {1..5}; do
            if [ -f "wallets/wallet_${i}.json" ]; then
                PUBKEY=$(solana-keygen pubkey "wallets/wallet_${i}.json")
                echo "Wallet $i: $PUBKEY"
            fi
        done
        ;;
    update)
        echo "🔄 Updating Ultimate Architect Agent..."
        git pull
        cargo build --release
        sudo systemctl restart ultimate-architect
        ;;
    backup)
        echo "💾 Creating backup..."
        tar -czf "backup_$(date +%Y%m%d_%H%M%S).tar.gz" wallets/ config/ logs/
        echo "Backup created successfully!"
        ;;
    *)
        echo "🤖 Ultimate Architect Agent - Management Commands"
        echo "================================================"
        echo "Usage: $0 {start|stop|restart|status|logs|monitor|wallets|update|backup}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the service"
        echo "  stop     - Stop the service"
        echo "  restart  - Restart the service"
        echo "  status   - Show service status"
        echo "  logs     - Show live logs"
        echo "  monitor  - Show system status"
        echo "  wallets  - Show wallet addresses"
        echo "  update   - Update and restart"
        echo "  backup   - Create backup"
        ;;
esac
EOF

chmod +x $APP_DIR/commands.sh

print_header "✅ DEPLOYMENT COMPLETE!"
echo ""
print_success "🎉 Ultimate Architect Agent successfully deployed!"
echo ""
echo -e "${CYAN}🌐 Access your system:${NC}"
echo "   Dashboard: http://$(curl -s ifconfig.me)"
echo "   Console:   http://$(curl -s ifconfig.me)/console"
echo "   Upload:    http://$(curl -s ifconfig.me)/upload"
echo ""
echo -e "${CYAN}💰 Fund your wallets:${NC}"
for i in {1..5}; do
    if [ -f "$APP_DIR/wallets/wallet_${i}.json" ]; then
        PUBKEY=$(solana-keygen pubkey "$APP_DIR/wallets/wallet_${i}.json")
        echo "   Wallet $i: $PUBKEY"
    fi
done
echo ""
echo -e "${CYAN}🔧 Management commands:${NC}"
echo "   cd $APP_DIR"
echo "   ./commands.sh status    # Check status"
echo "   ./commands.sh logs      # View logs"
echo "   ./commands.sh monitor   # System monitor"
echo "   ./commands.sh wallets   # Show wallet addresses"
echo ""
echo -e "${YELLOW}🚨 IMPORTANT:${NC}"
echo "1. Update your API keys in $APP_DIR/.env"
echo "2. Add your Helius API key for backup RPC"
echo "3. Fund the wallets with SOL to start trading"
echo "4. Monitor the system with ./commands.sh monitor"
echo ""
print_success "🤖 Your Ultimate Architect Agent is ready to engineer supreme solutions!"
echo ""

