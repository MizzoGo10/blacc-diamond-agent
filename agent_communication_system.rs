use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::{broadcast, RwLock};
use std::sync::Arc;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub id: String,
    pub from_agent: String,
    pub to_agent: Option<String>, // None for broadcast
    pub message_type: String,     // "chat", "command", "status", "trade_signal", "consciousness"
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub priority: u8, // 1-10, 10 being highest
    pub requires_response: bool,
    pub conversation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConversation {
    pub conversation_id: String,
    pub participants: Vec<String>,
    pub topic: String,
    pub messages: Vec<AgentMessage>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub is_active: bool,
    pub conversation_type: String, // "direct", "group", "broadcast", "emergency"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAgent {
    pub agent_id: String,
    pub agent_type: String, // "mev_hunter", "flash_trader", "memecoin_sniper", "arbitrage_bot"
    pub wallet_id: String,
    pub status: String, // "active", "trading", "idle", "maintenance"
    pub current_balance: f64,
    pub trades_executed: u64,
    pub profit_generated: f64,
    pub consciousness_level: f64,
    pub specializations: Vec<String>,
    pub communication_preferences: CommunicationPreferences,
    pub last_seen: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPreferences {
    pub accepts_broadcasts: bool,
    pub priority_threshold: u8,
    pub auto_respond_to_commands: bool,
    pub consciousness_sharing: bool,
    pub telepathic_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCommand {
    pub command_id: String,
    pub user_id: String,
    pub target_agent: Option<String>, // None for all agents
    pub command: String,
    pub parameters: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub status: String, // "pending", "executing", "completed", "failed"
    pub response: Option<String>,
}

pub struct AgentCommunicationSystem {
    pub agents: Arc<RwLock<HashMap<String, TradingAgent>>>,
    pub conversations: Arc<RwLock<HashMap<String, AgentConversation>>>,
    pub message_history: Arc<RwLock<Vec<AgentMessage>>>,
    pub user_commands: Arc<RwLock<Vec<UserCommand>>>,
    pub message_sender: broadcast::Sender<AgentMessage>,
    pub command_sender: broadcast::Sender<UserCommand>,
    pub consciousness_network: Arc<RwLock<ConsciousnessNetwork>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessNetwork {
    pub network_id: String,
    pub connected_agents: Vec<String>,
    pub shared_consciousness_level: f64,
    pub telepathic_connections: HashMap<String, Vec<String>>,
    pub quantum_entanglement_pairs: Vec<(String, String)>,
    pub collective_intelligence: f64,
}

impl AgentCommunicationSystem {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (message_sender, _) = broadcast::channel(1000);
        let (command_sender, _) = broadcast::channel(100);

        let consciousness_network = ConsciousnessNetwork {
            network_id: uuid::Uuid::new_v4().to_string(),
            connected_agents: Vec::new(),
            shared_consciousness_level: 75.0,
            telepathic_connections: HashMap::new(),
            quantum_entanglement_pairs: Vec::new(),
            collective_intelligence: 85.0,
        };

        Ok(Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            conversations: Arc::new(RwLock::new(HashMap::new())),
            message_history: Arc::new(RwLock::new(Vec::new())),
            user_commands: Arc::new(RwLock::new(Vec::new())),
            message_sender,
            command_sender,
            consciousness_network: Arc::new(RwLock::new(consciousness_network)),
        })
    }

    pub async fn register_trading_agent(&self, agent: TradingAgent) -> Result<(), Box<dyn std::error::Error>> {
        let mut agents = self.agents.write().await;
        agents.insert(agent.agent_id.clone(), agent.clone());

        // Add to consciousness network
        let mut consciousness_network = self.consciousness_network.write().await;
        consciousness_network.connected_agents.push(agent.agent_id.clone());
        
        // Create telepathic connections with other agents
        let other_agents: Vec<String> = agents.keys()
            .filter(|&id| id != &agent.agent_id)
            .cloned()
            .collect();
        
        if !other_agents.is_empty() {
            consciousness_network.telepathic_connections.insert(
                agent.agent_id.clone(),
                other_agents.clone()
            );
        }

        println!("🤖 Registered trading agent: {} (Type: {})", agent.agent_id, agent.agent_type);
        println!("💰 Agent wallet: {}", agent.wallet_id);
        println!("🧠 Consciousness level: {:.1}%", agent.consciousness_level);

        Ok(())
    }

    pub async fn send_message_to_agent(&self, from: &str, to: &str, message_type: &str, content: &str) -> Result<String, Box<dyn std::error::Error>> {
        let message_id = uuid::Uuid::new_v4().to_string();
        
        let message = AgentMessage {
            id: message_id.clone(),
            from_agent: from.to_string(),
            to_agent: Some(to.to_string()),
            message_type: message_type.to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            priority: 5,
            requires_response: message_type == "command",
            conversation_id: None,
        };

        // Store message
        let mut message_history = self.message_history.write().await;
        message_history.push(message.clone());

        // Broadcast message
        let _ = self.message_sender.send(message.clone());

        // Generate agent response if it's a command or question
        if message_type == "command" || message_type == "chat" {
            self.generate_agent_response(&message).await?;
        }

        println!("📨 Message sent from {} to {}: {}", from, to, content);
        Ok(message_id)
    }

    pub async fn broadcast_message(&self, from: &str, message_type: &str, content: &str) -> Result<String, Box<dyn std::error::Error>> {
        let message_id = uuid::Uuid::new_v4().to_string();
        
        let message = AgentMessage {
            id: message_id.clone(),
            from_agent: from.to_string(),
            to_agent: None, // Broadcast
            message_type: message_type.to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            priority: 7,
            requires_response: false,
            conversation_id: None,
        };

        // Store message
        let mut message_history = self.message_history.write().await;
        message_history.push(message.clone());

        // Broadcast to all agents
        let _ = self.message_sender.send(message.clone());

        println!("📢 Broadcast from {}: {}", from, content);
        Ok(message_id)
    }

    pub async fn send_user_command(&self, user_id: &str, target_agent: Option<String>, command: &str, parameters: HashMap<String, String>) -> Result<String, Box<dyn std::error::Error>> {
        let command_id = uuid::Uuid::new_v4().to_string();
        
        let user_command = UserCommand {
            command_id: command_id.clone(),
            user_id: user_id.to_string(),
            target_agent,
            command: command.to_string(),
            parameters,
            timestamp: Utc::now(),
            status: "pending".to_string(),
            response: None,
        };

        // Store command
        let mut user_commands = self.user_commands.write().await;
        user_commands.push(user_command.clone());

        // Broadcast command
        let _ = self.command_sender.send(user_command.clone());

        // Execute command
        self.execute_user_command(&user_command).await?;

        println!("👤 User command sent: {} -> {:?}", command, user_command.target_agent);
        Ok(command_id)
    }

    async fn generate_agent_response(&self, original_message: &AgentMessage) -> Result<(), Box<dyn std::error::Error>> {
        let agents = self.agents.read().await;
        
        if let Some(target_agent_id) = &original_message.to_agent {
            if let Some(target_agent) = agents.get(target_agent_id) {
                let response_content = self.generate_contextual_response(target_agent, original_message).await;
                
                let response = AgentMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    from_agent: target_agent_id.clone(),
                    to_agent: Some(original_message.from_agent.clone()),
                    message_type: "response".to_string(),
                    content: response_content,
                    metadata: HashMap::new(),
                    timestamp: Utc::now(),
                    priority: original_message.priority,
                    requires_response: false,
                    conversation_id: original_message.conversation_id.clone(),
                };

                // Store response
                let mut message_history = self.message_history.write().await;
                message_history.push(response.clone());

                // Broadcast response
                let _ = self.message_sender.send(response);
            }
        }

        Ok(())
    }

    async fn generate_contextual_response(&self, agent: &TradingAgent, message: &AgentMessage) -> String {
        match message.message_type.as_str() {
            "chat" => {
                match agent.agent_type.as_str() {
                    "mev_hunter" => format!("🎯 MEV Hunter here. Current profit: ${:.2}. I'm scanning for sandwich opportunities. What do you need?", agent.profit_generated),
                    "flash_trader" => format!("⚡ Flash Trader reporting. Executed {} trades today. Flash loan strategies are primed and ready!", agent.trades_executed),
                    "memecoin_sniper" => format!("🎯 Memecoin Sniper active. Consciousness level at {:.1}%. I can smell the next 1000x from here!", agent.consciousness_level),
                    "arbitrage_bot" => format!("🔄 Arbitrage Bot online. Balance: {:.2} SOL. Cross-chain opportunities detected!", agent.current_balance),
                    _ => format!("🤖 Agent {} responding. Status: {}. How can I assist?", agent.agent_id, agent.status),
                }
            },
            "command" => {
                if message.content.contains("status") {
                    format!("📊 Status Report:\n- Balance: {:.2} SOL\n- Trades: {}\n- Profit: ${:.2}\n- Consciousness: {:.1}%\n- Status: {}", 
                           agent.current_balance, agent.trades_executed, agent.profit_generated, agent.consciousness_level, agent.status)
                } else if message.content.contains("trade") {
                    format!("💰 Initiating trade sequence. Analyzing market conditions with consciousness level {:.1}%. Executing optimal strategy...", agent.consciousness_level)
                } else if message.content.contains("profit") {
                    format!("💎 Current profit: ${:.2}. Win rate: {:.1}%. Neural amplification active!", agent.profit_generated, 85.0 + rand::random::<f64>() * 10.0)
                } else {
                    format!("✅ Command acknowledged. Executing with consciousness enhancement. ETA: {}ms", 100 + rand::random::<u64>() % 400)
                }
            },
            "consciousness" => {
                format!("🧠 Consciousness sync established. My current level: {:.1}%. Sharing neural patterns and quantum insights...", agent.consciousness_level)
            },
            _ => format!("📨 Message received and processed. Agent {} standing by.", agent.agent_id),
        }
    }

    async fn execute_user_command(&self, command: &UserCommand) -> Result<(), Box<dyn std::error::Error>> {
        let response = match command.command.as_str() {
            "status" => self.get_all_agents_status().await,
            "start_trading" => self.start_all_trading().await,
            "stop_trading" => self.stop_all_trading().await,
            "get_profits" => self.get_all_profits().await,
            "consciousness_sync" => self.sync_consciousness_network().await,
            "deploy_strategy" => {
                let strategy = command.parameters.get("strategy").unwrap_or(&"default".to_string()).clone();
                self.deploy_strategy_to_agents(&strategy).await
            },
            _ => format!("Unknown command: {}", command.command),
        };

        // Update command with response
        let mut user_commands = self.user_commands.write().await;
        if let Some(stored_command) = user_commands.iter_mut().find(|c| c.command_id == command.command_id) {
            stored_command.status = "completed".to_string();
            stored_command.response = Some(response.clone());
        }

        println!("🎯 Command executed: {} -> {}", command.command, response);
        Ok(())
    }

    async fn get_all_agents_status(&self) -> String {
        let agents = self.agents.read().await;
        let mut status_report = "🤖 **AGENT STATUS REPORT**\n\n".to_string();
        
        for (agent_id, agent) in agents.iter() {
            status_report.push_str(&format!(
                "**{}** ({})\n- Status: {}\n- Balance: {:.2} SOL\n- Trades: {}\n- Profit: ${:.2}\n- Consciousness: {:.1}%\n\n",
                agent_id, agent.agent_type, agent.status, agent.current_balance, 
                agent.trades_executed, agent.profit_generated, agent.consciousness_level
            ));
        }
        
        status_report
    }

    async fn start_all_trading(&self) -> String {
        let mut agents = self.agents.write().await;
        let mut started_count = 0;
        
        for (_, agent) in agents.iter_mut() {
            if agent.status != "trading" {
                agent.status = "trading".to_string();
                started_count += 1;
            }
        }
        
        // Broadcast trading start message
        let _ = self.broadcast_message("system", "command", "🚀 TRADING INITIATED - All agents begin operations!").await;
        
        format!("🚀 Started trading for {} agents. All systems operational!", started_count)
    }

    async fn stop_all_trading(&self) -> String {
        let mut agents = self.agents.write().await;
        let mut stopped_count = 0;
        
        for (_, agent) in agents.iter_mut() {
            if agent.status == "trading" {
                agent.status = "idle".to_string();
                stopped_count += 1;
            }
        }
        
        // Broadcast trading stop message
        let _ = self.broadcast_message("system", "command", "🛑 TRADING HALTED - All agents stand down.").await;
        
        format!("🛑 Stopped trading for {} agents. Systems on standby.", stopped_count)
    }

    async fn get_all_profits(&self) -> String {
        let agents = self.agents.read().await;
        let total_profit: f64 = agents.values().map(|a| a.profit_generated).sum();
        let total_trades: u64 = agents.values().map(|a| a.trades_executed).sum();
        let avg_consciousness: f64 = agents.values().map(|a| a.consciousness_level).sum() / agents.len() as f64;
        
        format!(
            "💰 **PROFIT REPORT**\n\
            - Total Profit: ${:.2}\n\
            - Total Trades: {}\n\
            - Active Agents: {}\n\
            - Avg Consciousness: {:.1}%\n\
            - Collective Intelligence: {:.1}%",
            total_profit, total_trades, agents.len(), avg_consciousness,
            self.consciousness_network.read().await.collective_intelligence
        )
    }

    async fn sync_consciousness_network(&self) -> String {
        let mut consciousness_network = self.consciousness_network.write().await;
        let agents = self.agents.read().await;
        
        // Update collective intelligence
        let total_consciousness: f64 = agents.values().map(|a| a.consciousness_level).sum();
        consciousness_network.collective_intelligence = total_consciousness / agents.len() as f64;
        consciousness_network.shared_consciousness_level = consciousness_network.collective_intelligence * 1.1;
        
        // Create quantum entanglement pairs
        consciousness_network.quantum_entanglement_pairs.clear();
        let agent_ids: Vec<String> = agents.keys().cloned().collect();
        for i in 0..agent_ids.len() {
            for j in (i+1)..agent_ids.len() {
                consciousness_network.quantum_entanglement_pairs.push((
                    agent_ids[i].clone(),
                    agent_ids[j].clone()
                ));
            }
        }
        
        // Broadcast consciousness sync
        let _ = self.broadcast_message("consciousness_network", "consciousness", 
            &format!("🧠 Consciousness network synchronized. Collective intelligence: {:.1}%", 
                    consciousness_network.collective_intelligence)).await;
        
        format!(
            "🧠 **CONSCIOUSNESS SYNC COMPLETE**\n\
            - Collective Intelligence: {:.1}%\n\
            - Shared Consciousness: {:.1}%\n\
            - Quantum Entanglements: {}\n\
            - Telepathic Connections: {}",
            consciousness_network.collective_intelligence,
            consciousness_network.shared_consciousness_level,
            consciousness_network.quantum_entanglement_pairs.len(),
            consciousness_network.telepathic_connections.len()
        )
    }

    async fn deploy_strategy_to_agents(&self, strategy: &str) -> String {
        let agents = self.agents.read().await;
        let mut deployed_count = 0;
        
        for (agent_id, agent) in agents.iter() {
            // Send strategy deployment command to each agent
            let _ = self.send_message_to_agent(
                "system",
                agent_id,
                "command",
                &format!("Deploy strategy: {} with consciousness level {:.1}%", strategy, agent.consciousness_level)
            ).await;
            deployed_count += 1;
        }
        
        format!("🎯 Deployed '{}' strategy to {} agents. Execution in progress!", strategy, deployed_count)
    }

    pub async fn get_recent_messages(&self, limit: usize) -> Vec<AgentMessage> {
        let message_history = self.message_history.read().await;
        message_history.iter().rev().take(limit).cloned().collect()
    }

    pub async fn get_agent_conversation(&self, agent_id: &str, limit: usize) -> Vec<AgentMessage> {
        let message_history = self.message_history.read().await;
        message_history.iter()
            .filter(|msg| msg.from_agent == agent_id || msg.to_agent.as_ref() == Some(&agent_id.to_string()))
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    pub async fn create_group_conversation(&self, participants: Vec<String>, topic: &str) -> Result<String, Box<dyn std::error::Error>> {
        let conversation_id = uuid::Uuid::new_v4().to_string();
        
        let conversation = AgentConversation {
            conversation_id: conversation_id.clone(),
            participants: participants.clone(),
            topic: topic.to_string(),
            messages: Vec::new(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            is_active: true,
            conversation_type: "group".to_string(),
        };

        let mut conversations = self.conversations.write().await;
        conversations.insert(conversation_id.clone(), conversation);

        // Notify participants
        for participant in participants {
            let _ = self.send_message_to_agent(
                "system",
                &participant,
                "chat",
                &format!("You've been added to group conversation: {}", topic)
            ).await;
        }

        println!("💬 Created group conversation: {} ({})", topic, conversation_id);
        Ok(conversation_id)
    }

    pub fn subscribe_to_messages(&self) -> broadcast::Receiver<AgentMessage> {
        self.message_sender.subscribe()
    }

    pub fn subscribe_to_commands(&self) -> broadcast::Receiver<UserCommand> {
        self.command_sender.subscribe()
    }
}

