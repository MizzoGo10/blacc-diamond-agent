#!/bin/bash

# 🌊 BLACC DIAMOND AGENT - ONE CLICK DEPLOYMENT
# ==============================================
# This script deploys your entire AI trading system with one command

set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${PURPLE}"
echo "🌊🤖 BLACC DIAMOND AI TRADING SYSTEM 🤖🌊"
echo "=========================================="
echo -e "${NC}"
echo -e "${CYAN}🎯 What you're deploying:${NC}"
echo -e "   • AI Trading Bots that work 24/7"
echo -e "   • Consciousness-enhanced decision making"
echo -e "   • MEV extraction and flash loan arbitrage"
echo -e "   • Real-time dashboard and agent chat"
echo -e "   • Elite strategies: VOIDSTRIKE, Phoenix, Quantum Genesis"
echo ""
echo -e "${YELLOW}💰 Expected results:${NC}"
echo -e "   • Automated trading on Solana"
echo -e "   • 24/7 profit generation"
echo -e "   • Smart risk management"
echo -e "   • Complete control via web dashboard"
echo ""

# Check if user wants to continue
read -p "🚀 Ready to deploy your AI trading system? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Deployment cancelled${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Starting one-click deployment...${NC}"

# Step 1: Check for Digital Ocean token
echo -e "${BLUE}🔑 Digital Ocean Setup${NC}"
if [ -z "$DO_TOKEN" ]; then
    echo -e "${YELLOW}📝 I need your Digital Ocean API token${NC}"
    echo -e "${CYAN}   1. Go to: https://cloud.digitalocean.com/account/api/tokens${NC}"
    echo -e "${CYAN}   2. Click 'Generate New Token'${NC}"
    echo -e "${CYAN}   3. Name it 'blacc-diamond-agent'${NC}"
    echo -e "${CYAN}   4. Check 'Write' scope${NC}"
    echo -e "${CYAN}   5. Copy the token and paste it here${NC}"
    echo ""
    read -p "🔑 Paste your Digital Ocean token: " DO_TOKEN
    
    if [ -z "$DO_TOKEN" ]; then
        echo -e "${RED}❌ Token required for deployment${NC}"
        exit 1
    fi
fi

# Export token for doctl
export DIGITALOCEAN_ACCESS_TOKEN=$DO_TOKEN

echo -e "${GREEN}✅ Token received${NC}"

# Step 2: Install doctl if needed
echo -e "${BLUE}🛠️ Installing Digital Ocean CLI...${NC}"
if ! command -v doctl &> /dev/null; then
    curl -sL https://github.com/digitalocean/doctl/releases/download/v1.104.0/doctl-1.104.0-linux-amd64.tar.gz | tar -xzv
    sudo mv doctl /usr/local/bin
    echo -e "${GREEN}✅ doctl installed${NC}"
else
    echo -e "${GREEN}✅ doctl already installed${NC}"
fi

# Step 3: Authenticate
echo -e "${BLUE}🔐 Authenticating with Digital Ocean...${NC}"
echo $DO_TOKEN | doctl auth init --access-token -

if ! doctl account get &> /dev/null; then
    echo -e "${RED}❌ Authentication failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Authentication successful${NC}"

# Step 4: Create SSH key
echo -e "${BLUE}🔑 Setting up SSH access...${NC}"
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -q
fi

# Upload SSH key to Digital Ocean
SSH_KEY_ID=$(doctl compute ssh-key create blacc-diamond-$(date +%s ) --public-key-file ~/.ssh/id_rsa.pub --format ID --no-header 2>/dev/null || echo "")

if [ -z "$SSH_KEY_ID" ]; then
    # Try to find existing key
    SSH_KEY_ID=$(doctl compute ssh-key list --format ID,Name --no-header | grep blacc-diamond | head -1 | cut -d' ' -f1)
fi

if [ -z "$SSH_KEY_ID" ]; then
    echo -e "${RED}❌ Failed to setup SSH key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH key ready${NC}"

# Step 5: Create droplet
echo -e "${BLUE}🚀 Creating your AI trading server...${NC}"
echo -e "${CYAN}   • Server size: 4 vCPU, 8GB RAM${NC}"
echo -e "${CYAN}   • Location: New York datacenter${NC}"
echo -e "${CYAN}   • Cost: ~$48/month${NC}"
echo -e "${CYAN}   • Features: Docker pre-installed${NC}"

DROPLET_NAME="blacc-diamond-$(date +%s)"
DROPLET_ID=$(doctl compute droplet create $DROPLET_NAME \
    --size s-4vcpu-8gb \
    --region nyc1 \
    --image docker-20-04 \
    --ssh-keys $SSH_KEY_ID \
    --enable-monitoring \
    --format ID \
    --no-header \
    --wait)

if [ -z "$DROPLET_ID" ]; then
    echo -e "${RED}❌ Failed to create server${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Server created: $DROPLET_ID${NC}"

# Step 6: Get IP address
echo -e "${BLUE}🌐 Getting server IP address...${NC}"
DROPLET_IP=$(doctl compute droplet get $DROPLET_ID --format PublicIPv4 --no-header)
echo -e "${GREEN}✅ Server IP: $DROPLET_IP${NC}"

# Step 7: Wait for server to be ready
echo -e "${BLUE}⏳ Waiting for server to boot up...${NC}"
echo -e "${CYAN}   This takes about 2-3 minutes...${NC}"

for i in {1..20}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$DROPLET_IP "echo 'ready'" &> /dev/null; then
        echo -e "${GREEN}✅ Server is ready!${NC}"
        break
    else
        echo -ne "${YELLOW}⏳ Waiting... ($i/20)\r${NC}"
        sleep 15
    fi
    
    if [ $i -eq 20 ]; then
        echo -e "${RED}❌ Server took too long to start${NC}"
        exit 1
    fi
done

# Step 8: Clone and deploy from GitHub
echo -e "${BLUE}📤 Deploying your AI trading system from GitHub...${NC}"
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << 'EOF'
cd /opt

# Clone the repository
git clone https://github.com/MizzoGo10/blacc-diamond-agent.git
cd blacc-diamond-agent

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Install additional dependencies
apt-get update -qq
apt-get install -y build-essential pkg-config libssl-dev

# Build the application
echo "🏗️ Building AI trading system..."
cargo build --release

# Setup environment
if [ -f .env.production ]; then
    cp .env.production .env
fi

# Create necessary directories
mkdir -p logs wallets monitoring/grafana/dashboards monitoring/grafana/datasources

# Setup Docker Compose if available
if [ -f digital_ocean_deployment.yml ]; then
    docker-compose -f digital_ocean_deployment.yml up -d
fi

echo "✅ Deployment complete!"
EOF

# Step 9: Configure firewall
echo -e "${BLUE}🛡️ Setting up security...${NC}"
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << 'EOF'
ufw allow ssh
ufw allow 80
ufw allow 443
ufw allow 8080
ufw allow 8081
ufw allow 8082
ufw --force enable
EOF

# Step 10: Final setup
echo -e "${BLUE}🔧 Final configuration...${NC}"
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP << EOF
cd /opt/blacc-diamond-agent

# Set Quicknode endpoint
export QUICKNODE_RPC_URL="https://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/"
export QUICKNODE_WSS_URL="wss://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/"

# Add to environment file
echo 'QUICKNODE_RPC_URL="https://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/"' >> .env
echo 'QUICKNODE_WSS_URL="wss://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/"' >> .env

# Restart services if Docker Compose is running
if [ -f digital_ocean_deployment.yml ]; then
    docker-compose -f digital_ocean_deployment.yml restart
fi

# Wait for services to start
sleep 30

echo "🎉 System is ready!"
EOF

# Success!
clear
echo -e "${GREEN}"
echo "🎉🎉🎉 DEPLOYMENT SUCCESSFUL! 🎉🎉🎉"
echo "====================================="
echo -e "${NC}"
echo ""
echo -e "${PURPLE}🤖 Your AI Trading System is LIVE! 🤖${NC}"
echo ""
echo -e "${CYAN}🔗 Access Your System:${NC}"
echo -e "   🎯 Main Control Panel: ${YELLOW}http://$DROPLET_IP:8080${NC}"
echo -e "   📊 Trading Dashboard:  ${YELLOW}http://$DROPLET_IP:8081${NC}"
echo -e "   📈 Monitoring:         ${YELLOW}http://$DROPLET_IP:3000${NC}"
echo -e "   🔌 WebSocket:          ${YELLOW}ws://$DROPLET_IP:8082${NC}"
echo ""
echo -e "${CYAN}🔑 Server Access:${NC}"
echo -e "   SSH: ${YELLOW}ssh root@$DROPLET_IP${NC}"
echo ""
echo -e "${GREEN}🚀 Your consciousness-enhanced trading empire is now operational!${NC}"
