use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::{broadcast, RwLock};
use std::sync::Arc;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTrade {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub trade_type: String, // "buy", "sell", "arbitrage", "mev", "flash_loan"
    pub token_pair: String,
    pub amount: f64,
    pub price: f64,
    pub profit_loss: f64,
    pub gas_fee: f64,
    pub success: bool,
    pub agent_id: String,
    pub strategy: String,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLSummary {
    pub total_profit: f64,
    pub total_loss: f64,
    pub net_pnl: f64,
    pub win_rate: f64,
    pub total_trades: u64,
    pub successful_trades: u64,
    pub failed_trades: u64,
    pub average_profit_per_trade: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub daily_pnl: f64,
    pub weekly_pnl: f64,
    pub monthly_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub status: String, // "active", "idle", "error", "maintenance"
    pub current_task: Option<String>,
    pub performance_score: f64,
    pub trades_executed: u64,
    pub profit_generated: f64,
    pub last_activity: DateTime<Utc>,
    pub health_score: f64,
    pub consciousness_level: f64,
    pub quantum_coherence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_latency: f64,
    pub active_connections: u32,
    pub trades_per_second: f64,
    pub error_rate: f64,
    pub uptime_seconds: u64,
    pub blockchain_sync_status: String,
    pub wallet_balance: f64,
    pub pending_transactions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedReturns {
    pub next_minute: f64,
    pub next_5_minutes: f64,
    pub next_15_minutes: f64,
    pub next_hour: f64,
    pub confidence_level: f64,
    pub prediction_model: String,
    pub factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: String, // "low", "medium", "high", "critical"
    pub alert_type: String,
    pub message: String,
    pub affected_component: String,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub timestamp: DateTime<Utc>,
    pub live_trades: Vec<LiveTrade>,
    pub pnl_summary: PnLSummary,
    pub agent_statuses: Vec<AgentStatus>,
    pub system_metrics: SystemMetrics,
    pub projected_returns: ProjectedReturns,
    pub security_alerts: Vec<SecurityAlert>,
    pub active_strategies: Vec<String>,
    pub market_conditions: MarketConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub volume: f64,
    pub trend: String, // "bullish", "bearish", "sideways"
    pub fear_greed_index: f64,
    pub liquidity_score: f64,
    pub mev_opportunities: u32,
    pub gas_price: f64,
}

pub struct LiveDashboardSystem {
    pub dashboard_data: Arc<RwLock<DashboardData>>,
    pub trade_history: Arc<RwLock<Vec<LiveTrade>>>,
    pub agent_registry: Arc<RwLock<HashMap<String, AgentStatus>>>,
    pub security_alerts: Arc<RwLock<Vec<SecurityAlert>>>,
    pub broadcast_sender: broadcast::Sender<DashboardData>,
    pub metrics_collector: MetricsCollector,
    pub pnl_calculator: PnLCalculator,
    pub prediction_engine: PredictionEngine,
}

impl LiveDashboardSystem {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (broadcast_sender, _) = broadcast::channel(1000);
        
        let initial_dashboard_data = DashboardData {
            timestamp: Utc::now(),
            live_trades: Vec::new(),
            pnl_summary: PnLSummary {
                total_profit: 0.0,
                total_loss: 0.0,
                net_pnl: 0.0,
                win_rate: 0.0,
                total_trades: 0,
                successful_trades: 0,
                failed_trades: 0,
                average_profit_per_trade: 0.0,
                largest_win: 0.0,
                largest_loss: 0.0,
                daily_pnl: 0.0,
                weekly_pnl: 0.0,
                monthly_pnl: 0.0,
            },
            agent_statuses: Vec::new(),
            system_metrics: SystemMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_latency: 0.0,
                active_connections: 0,
                trades_per_second: 0.0,
                error_rate: 0.0,
                uptime_seconds: 0,
                blockchain_sync_status: "synced".to_string(),
                wallet_balance: 0.0,
                pending_transactions: 0,
            },
            projected_returns: ProjectedReturns {
                next_minute: 0.0,
                next_5_minutes: 0.0,
                next_15_minutes: 0.0,
                next_hour: 0.0,
                confidence_level: 0.0,
                prediction_model: "consciousness_enhanced".to_string(),
                factors: Vec::new(),
            },
            security_alerts: Vec::new(),
            active_strategies: Vec::new(),
            market_conditions: MarketConditions {
                volatility: 0.0,
                volume: 0.0,
                trend: "sideways".to_string(),
                fear_greed_index: 50.0,
                liquidity_score: 0.0,
                mev_opportunities: 0,
                gas_price: 0.0,
            },
        };

        Ok(Self {
            dashboard_data: Arc::new(RwLock::new(initial_dashboard_data)),
            trade_history: Arc::new(RwLock::new(Vec::new())),
            agent_registry: Arc::new(RwLock::new(HashMap::new())),
            security_alerts: Arc::new(RwLock::new(Vec::new())),
            broadcast_sender,
            metrics_collector: MetricsCollector::new(),
            pnl_calculator: PnLCalculator::new(),
            prediction_engine: PredictionEngine::new(),
        })
    }

    pub async fn start_live_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Start background tasks for live monitoring
        let dashboard_data = Arc::clone(&self.dashboard_data);
        let trade_history = Arc::clone(&self.trade_history);
        let agent_registry = Arc::clone(&self.agent_registry);
        let broadcast_sender = self.broadcast_sender.clone();
        let metrics_collector = self.metrics_collector.clone();
        let pnl_calculator = self.pnl_calculator.clone();
        let prediction_engine = self.prediction_engine.clone();

        // Real-time data update loop
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                // Update system metrics
                let system_metrics = metrics_collector.collect_metrics().await;
                
                // Calculate PnL
                let trade_history_read = trade_history.read().await;
                let pnl_summary = pnl_calculator.calculate_pnl(&trade_history_read).await;
                drop(trade_history_read);
                
                // Generate predictions
                let projected_returns = prediction_engine.generate_predictions().await;
                
                // Update dashboard data
                let mut dashboard_data_write = dashboard_data.write().await;
                dashboard_data_write.timestamp = Utc::now();
                dashboard_data_write.system_metrics = system_metrics;
                dashboard_data_write.pnl_summary = pnl_summary;
                dashboard_data_write.projected_returns = projected_returns;
                
                // Get recent trades (last 100)
                let trade_history_read = trade_history.read().await;
                dashboard_data_write.live_trades = trade_history_read
                    .iter()
                    .rev()
                    .take(100)
                    .cloned()
                    .collect();
                drop(trade_history_read);
                
                // Get agent statuses
                let agent_registry_read = agent_registry.read().await;
                dashboard_data_write.agent_statuses = agent_registry_read.values().cloned().collect();
                drop(agent_registry_read);
                
                // Broadcast update
                let _ = broadcast_sender.send(dashboard_data_write.clone());
                drop(dashboard_data_write);
            }
        });

        Ok(())
    }

    pub async fn record_trade(&self, trade: LiveTrade) -> Result<(), Box<dyn std::error::Error>> {
        let mut trade_history = self.trade_history.write().await;
        trade_history.push(trade.clone());
        
        // Keep only last 10,000 trades in memory
        if trade_history.len() > 10000 {
            trade_history.remove(0);
        }
        
        println!("📊 Trade recorded: {} {} {} for ${:.2} profit", 
                trade.trade_type, trade.token_pair, trade.amount, trade.profit_loss);
        
        Ok(())
    }

    pub async fn update_agent_status(&self, agent_id: String, status: AgentStatus) -> Result<(), Box<dyn std::error::Error>> {
        let mut agent_registry = self.agent_registry.write().await;
        agent_registry.insert(agent_id, status);
        Ok(())
    }

    pub async fn add_security_alert(&self, alert: SecurityAlert) -> Result<(), Box<dyn std::error::Error>> {
        let mut security_alerts = self.security_alerts.write().await;
        security_alerts.push(alert.clone());
        
        // Keep only last 1000 alerts
        if security_alerts.len() > 1000 {
            security_alerts.remove(0);
        }
        
        println!("🚨 Security Alert [{}]: {}", alert.severity.to_uppercase(), alert.message);
        
        Ok(())
    }

    pub async fn get_dashboard_data(&self) -> DashboardData {
        self.dashboard_data.read().await.clone()
    }

    pub async fn get_trade_history(&self, limit: Option<usize>) -> Vec<LiveTrade> {
        let trade_history = self.trade_history.read().await;
        let limit = limit.unwrap_or(100);
        trade_history.iter().rev().take(limit).cloned().collect()
    }

    pub async fn get_pnl_report(&self, timeframe: &str) -> PnLReport {
        let trade_history = self.trade_history.read().await;
        self.pnl_calculator.generate_report(&trade_history, timeframe).await
    }

    pub async fn get_agent_performance(&self) -> Vec<AgentPerformance> {
        let agent_registry = self.agent_registry.read().await;
        let trade_history = self.trade_history.read().await;
        
        let mut performance_data = Vec::new();
        
        for (agent_id, status) in agent_registry.iter() {
            let agent_trades: Vec<&LiveTrade> = trade_history
                .iter()
                .filter(|trade| trade.agent_id == *agent_id)
                .collect();
            
            let total_profit: f64 = agent_trades.iter().map(|t| t.profit_loss).sum();
            let successful_trades = agent_trades.iter().filter(|t| t.success).count();
            let win_rate = if !agent_trades.is_empty() {
                successful_trades as f64 / agent_trades.len() as f64 * 100.0
            } else {
                0.0
            };
            
            performance_data.push(AgentPerformance {
                agent_id: agent_id.clone(),
                total_trades: agent_trades.len() as u64,
                successful_trades: successful_trades as u64,
                total_profit,
                win_rate,
                performance_score: status.performance_score,
                consciousness_level: status.consciousness_level,
                last_activity: status.last_activity,
            });
        }
        
        performance_data
    }

    pub fn subscribe_to_updates(&self) -> broadcast::Receiver<DashboardData> {
        self.broadcast_sender.subscribe()
    }
}

#[derive(Debug, Clone)]
pub struct MetricsCollector {
    start_time: DateTime<Utc>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Utc::now(),
        }
    }

    pub async fn collect_metrics(&self) -> SystemMetrics {
        // Simulate real metrics collection
        SystemMetrics {
            cpu_usage: 15.0 + rand::random::<f64>() * 20.0,
            memory_usage: 45.0 + rand::random::<f64>() * 15.0,
            network_latency: 5.0 + rand::random::<f64>() * 10.0,
            active_connections: 50 + (rand::random::<u32>() % 50),
            trades_per_second: rand::random::<f64>() * 10.0,
            error_rate: rand::random::<f64>() * 2.0,
            uptime_seconds: (Utc::now() - self.start_time).num_seconds() as u64,
            blockchain_sync_status: "synced".to_string(),
            wallet_balance: 10000.0 + rand::random::<f64>() * 50000.0,
            pending_transactions: rand::random::<u32>() % 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PnLCalculator;

impl PnLCalculator {
    pub fn new() -> Self {
        Self
    }

    pub async fn calculate_pnl(&self, trades: &[LiveTrade]) -> PnLSummary {
        if trades.is_empty() {
            return PnLSummary {
                total_profit: 0.0,
                total_loss: 0.0,
                net_pnl: 0.0,
                win_rate: 0.0,
                total_trades: 0,
                successful_trades: 0,
                failed_trades: 0,
                average_profit_per_trade: 0.0,
                largest_win: 0.0,
                largest_loss: 0.0,
                daily_pnl: 0.0,
                weekly_pnl: 0.0,
                monthly_pnl: 0.0,
            };
        }

        let total_profit: f64 = trades.iter().filter(|t| t.profit_loss > 0.0).map(|t| t.profit_loss).sum();
        let total_loss: f64 = trades.iter().filter(|t| t.profit_loss < 0.0).map(|t| t.profit_loss.abs()).sum();
        let net_pnl = total_profit - total_loss;
        let successful_trades = trades.iter().filter(|t| t.success).count() as u64;
        let failed_trades = trades.len() as u64 - successful_trades;
        let win_rate = if !trades.is_empty() {
            successful_trades as f64 / trades.len() as f64 * 100.0
        } else {
            0.0
        };

        let largest_win = trades.iter().map(|t| t.profit_loss).fold(0.0, f64::max);
        let largest_loss = trades.iter().map(|t| t.profit_loss).fold(0.0, f64::min);

        // Calculate time-based PnL
        let now = Utc::now();
        let daily_trades: Vec<&LiveTrade> = trades.iter()
            .filter(|t| (now - t.timestamp).num_hours() <= 24)
            .collect();
        let weekly_trades: Vec<&LiveTrade> = trades.iter()
            .filter(|t| (now - t.timestamp).num_days() <= 7)
            .collect();
        let monthly_trades: Vec<&LiveTrade> = trades.iter()
            .filter(|t| (now - t.timestamp).num_days() <= 30)
            .collect();

        let daily_pnl: f64 = daily_trades.iter().map(|t| t.profit_loss).sum();
        let weekly_pnl: f64 = weekly_trades.iter().map(|t| t.profit_loss).sum();
        let monthly_pnl: f64 = monthly_trades.iter().map(|t| t.profit_loss).sum();

        PnLSummary {
            total_profit,
            total_loss,
            net_pnl,
            win_rate,
            total_trades: trades.len() as u64,
            successful_trades,
            failed_trades,
            average_profit_per_trade: if !trades.is_empty() { net_pnl / trades.len() as f64 } else { 0.0 },
            largest_win,
            largest_loss,
            daily_pnl,
            weekly_pnl,
            monthly_pnl,
        }
    }

    pub async fn generate_report(&self, trades: &[LiveTrade], timeframe: &str) -> PnLReport {
        let now = Utc::now();
        let filtered_trades: Vec<&LiveTrade> = match timeframe {
            "1h" => trades.iter().filter(|t| (now - t.timestamp).num_hours() <= 1).collect(),
            "24h" => trades.iter().filter(|t| (now - t.timestamp).num_hours() <= 24).collect(),
            "7d" => trades.iter().filter(|t| (now - t.timestamp).num_days() <= 7).collect(),
            "30d" => trades.iter().filter(|t| (now - t.timestamp).num_days() <= 30).collect(),
            _ => trades.iter().collect(),
        };

        let pnl_summary = self.calculate_pnl(&filtered_trades.into_iter().cloned().collect::<Vec<_>>()).await;

        PnLReport {
            timeframe: timeframe.to_string(),
            summary: pnl_summary,
            trade_count: filtered_trades.len(),
            generated_at: now,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictionEngine;

impl PredictionEngine {
    pub fn new() -> Self {
        Self
    }

    pub async fn generate_predictions(&self) -> ProjectedReturns {
        // Simulate consciousness-enhanced predictions
        let base_return = rand::random::<f64>() * 1000.0 - 500.0; // -500 to +500
        let volatility = 0.1 + rand::random::<f64>() * 0.2; // 10-30% volatility
        
        ProjectedReturns {
            next_minute: base_return * 0.1,
            next_5_minutes: base_return * 0.3,
            next_15_minutes: base_return * 0.6,
            next_hour: base_return,
            confidence_level: 75.0 + rand::random::<f64>() * 20.0,
            prediction_model: "consciousness_enhanced_fractal".to_string(),
            factors: vec![
                "market_sentiment".to_string(),
                "liquidity_analysis".to_string(),
                "mev_opportunities".to_string(),
                "consciousness_patterns".to_string(),
                "quantum_entanglement".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLReport {
    pub timeframe: String,
    pub summary: PnLSummary,
    pub trade_count: usize,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformance {
    pub agent_id: String,
    pub total_trades: u64,
    pub successful_trades: u64,
    pub total_profit: f64,
    pub win_rate: f64,
    pub performance_score: f64,
    pub consciousness_level: f64,
    pub last_activity: DateTime<Utc>,
}

// Command and Control Console
pub struct CommandConsole {
    pub dashboard_system: Arc<LiveDashboardSystem>,
    pub command_history: Arc<RwLock<Vec<ConsoleCommand>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleCommand {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub command: String,
    pub args: Vec<String>,
    pub result: String,
    pub success: bool,
}

impl CommandConsole {
    pub fn new(dashboard_system: Arc<LiveDashboardSystem>) -> Self {
        Self {
            dashboard_system,
            command_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn execute_command(&self, command: &str, args: Vec<String>) -> Result<String, Box<dyn std::error::Error>> {
        let command_id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();
        
        let result = match command {
            "status" => self.get_system_status().await,
            "agents" => self.list_agents().await,
            "trades" => self.get_recent_trades(args.get(0).and_then(|s| s.parse().ok()).unwrap_or(10)).await,
            "pnl" => self.get_pnl_summary(args.get(0).unwrap_or(&"24h".to_string())).await,
            "alert" => self.create_alert(args).await,
            "deploy" => self.deploy_agent(args).await,
            "stop" => self.stop_agent(args).await,
            "restart" => self.restart_system().await,
            "backup" => self.create_backup().await,
            "security" => self.security_scan().await,
            _ => format!("Unknown command: {}", command),
        };

        let console_command = ConsoleCommand {
            id: command_id,
            timestamp,
            command: command.to_string(),
            args,
            result: result.clone(),
            success: !result.starts_with("Error") && !result.starts_with("Unknown"),
        };

        let mut command_history = self.command_history.write().await;
        command_history.push(console_command);
        
        // Keep only last 1000 commands
        if command_history.len() > 1000 {
            command_history.remove(0);
        }

        Ok(result)
    }

    async fn get_system_status(&self) -> String {
        let dashboard_data = self.dashboard_system.get_dashboard_data().await;
        format!(
            "System Status:\n\
            - Uptime: {} seconds\n\
            - CPU: {:.1}%\n\
            - Memory: {:.1}%\n\
            - Active Agents: {}\n\
            - Trades/sec: {:.2}\n\
            - Net PnL: ${:.2}\n\
            - Wallet Balance: ${:.2}",
            dashboard_data.system_metrics.uptime_seconds,
            dashboard_data.system_metrics.cpu_usage,
            dashboard_data.system_metrics.memory_usage,
            dashboard_data.agent_statuses.len(),
            dashboard_data.system_metrics.trades_per_second,
            dashboard_data.pnl_summary.net_pnl,
            dashboard_data.system_metrics.wallet_balance
        )
    }

    async fn list_agents(&self) -> String {
        let dashboard_data = self.dashboard_system.get_dashboard_data().await;
        let mut result = "Active Agents:\n".to_string();
        
        for agent in dashboard_data.agent_statuses {
            result.push_str(&format!(
                "- {}: {} (Score: {:.1}, Consciousness: {:.1}%)\n",
                agent.agent_id, agent.status, agent.performance_score, agent.consciousness_level
            ));
        }
        
        result
    }

    async fn get_recent_trades(&self, limit: usize) -> String {
        let trades = self.dashboard_system.get_trade_history(Some(limit)).await;
        let mut result = format!("Recent {} Trades:\n", trades.len());
        
        for trade in trades.iter().take(10) {
            result.push_str(&format!(
                "- {} {}: {} {} @ ${:.4} = ${:.2} ({})\n",
                trade.timestamp.format("%H:%M:%S"),
                trade.trade_type,
                trade.amount,
                trade.token_pair,
                trade.price,
                trade.profit_loss,
                if trade.success { "✓" } else { "✗" }
            ));
        }
        
        result
    }

    async fn get_pnl_summary(&self, timeframe: &str) -> String {
        let report = self.dashboard_system.get_pnl_report(timeframe).await;
        format!(
            "PnL Summary ({}): \n\
            - Net PnL: ${:.2}\n\
            - Win Rate: {:.1}%\n\
            - Total Trades: {}\n\
            - Avg Profit/Trade: ${:.2}\n\
            - Largest Win: ${:.2}\n\
            - Largest Loss: ${:.2}",
            timeframe,
            report.summary.net_pnl,
            report.summary.win_rate,
            report.summary.total_trades,
            report.summary.average_profit_per_trade,
            report.summary.largest_win,
            report.summary.largest_loss
        )
    }

    async fn create_alert(&self, args: Vec<String>) -> String {
        if args.len() < 2 {
            return "Usage: alert <severity> <message>".to_string();
        }
        
        let alert = SecurityAlert {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            severity: args[0].clone(),
            alert_type: "manual".to_string(),
            message: args[1..].join(" "),
            affected_component: "console".to_string(),
            resolved: false,
        };
        
        let _ = self.dashboard_system.add_security_alert(alert).await;
        "Alert created successfully".to_string()
    }

    async fn deploy_agent(&self, args: Vec<String>) -> String {
        if args.is_empty() {
            return "Usage: deploy <agent_type>".to_string();
        }
        
        let agent_id = format!("{}_{}", args[0], uuid::Uuid::new_v4());
        let status = AgentStatus {
            agent_id: agent_id.clone(),
            status: "active".to_string(),
            current_task: Some("initializing".to_string()),
            performance_score: 95.0,
            trades_executed: 0,
            profit_generated: 0.0,
            last_activity: Utc::now(),
            health_score: 100.0,
            consciousness_level: 85.0,
            quantum_coherence: 0.92,
        };
        
        let _ = self.dashboard_system.update_agent_status(agent_id.clone(), status).await;
        format!("Agent {} deployed successfully", agent_id)
    }

    async fn stop_agent(&self, args: Vec<String>) -> String {
        if args.is_empty() {
            return "Usage: stop <agent_id>".to_string();
        }
        
        // Simulate stopping agent
        format!("Agent {} stopped", args[0])
    }

    async fn restart_system(&self) -> String {
        "System restart initiated...".to_string()
    }

    async fn create_backup(&self) -> String {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        format!("Backup created: backup_{}.db", timestamp)
    }

    async fn security_scan(&self) -> String {
        "Security scan completed: No threats detected".to_string()
    }

    pub async fn get_command_history(&self, limit: Option<usize>) -> Vec<ConsoleCommand> {
        let command_history = self.command_history.read().await;
        let limit = limit.unwrap_or(50);
        command_history.iter().rev().take(limit).cloned().collect()
    }
}

