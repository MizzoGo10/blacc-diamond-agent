use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, Instant};
use anyhow::Result;
use tracing::{info, warn, error};

use crate::atomic_micro_flash::{AtomicMicroFlashEngine, ExecutionResult};

#[derive(Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub id: String,
    pub name: String,
    pub strategy_type: StrategyType,
    pub enabled: bool,
    pub min_capital: f64,
    pub max_capital: f64,
    pub risk_level: u8,
    pub expected_win_rate: f64,
    pub execution_time_ms: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum StrategyType {
    FlashLoan,
    MEVHunting,
    Arbitrage,
    MemecoinSniping,
    LiquidationBot,
    ConsciousnessTrading,
}

pub struct StrategyEngine {
    pub strategies: HashMap<String, Strategy>,
    pub flash_engine: AtomicMicroFlashEngine,
    pub active_executions: Vec<ActiveExecution>,
    pub performance_tracker: PerformanceTracker,
}

#[derive(Debug)]
pub struct ActiveExecution {
    pub strategy_id: String,
    pub start_time: Instant,
    pub capital_used: f64,
    pub status: ExecutionStatus,
}

#[derive(Debug)]
pub enum ExecutionStatus {
    Running,
    Completed(ExecutionResult),
    Failed(String),
}

#[derive(Default)]
pub struct PerformanceTracker {
    pub total_strategies_executed: u64,
    pub successful_strategies: u64,
    pub total_profit: f64,
    pub total_loss: f64,
    pub best_performing_strategy: Option<String>,
    pub worst_performing_strategy: Option<String>,
}

impl StrategyEngine {
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        
        // Initialize core strategies
        strategies.insert("quantum_cascade".to_string(), Strategy {
            id: "quantum_cascade".to_string(),
            name: "Quantum Atomic Cascade".to_string(),
            strategy_type: StrategyType::FlashLoan,
            enabled: true,
            min_capital: 0.0,
            max_capital: 1000.0,
            risk_level: 3,
            expected_win_rate: 97.8,
            execution_time_ms: 1,
        });
        
        strategies.insert("consciousness_arb".to_string(), Strategy {
            id: "consciousness_arb".to_string(),
            name: "Consciousness Arbitrage".to_string(),
            strategy_type: StrategyType::ConsciousnessTrading,
            enabled: true,
            min_capital: 0.0,
            max_capital: 500.0,
            risk_level: 2,
            expected_win_rate: 98.2,
            execution_time_ms: 1,
        });
        
        strategies.insert("memecoin_hunter".to_string(), Strategy {
            id: "memecoin_hunter".to_string(),
            name: "Atomic Memecoin Hunter".to_string(),
            strategy_type: StrategyType::MemecoinSniping,
            enabled: true,
            min_capital: 0.0,
            max_capital: 100.0,
            risk_level: 7,
            expected_win_rate: 92.7,
            execution_time_ms: 1,
        });
        
        strategies.insert("lightning_matrix".to_string(), Strategy {
            id: "lightning_matrix".to_string(),
            name: "Lightning Arbitrage Matrix".to_string(),
            strategy_type: StrategyType::Arbitrage,
            enabled: true,
            min_capital: 0.0,
            max_capital: 750.0,
            risk_level: 4,
            expected_win_rate: 96.4,
            execution_time_ms: 1,
        });
        
        strategies.insert("fractal_cascade".to_string(), Strategy {
            id: "fractal_cascade".to_string(),
            name: "Fractal Memecoin Cascade".to_string(),
            strategy_type: StrategyType::MemecoinSniping,
            enabled: true,
            min_capital: 0.0,
            max_capital: 200.0,
            risk_level: 8,
            expected_win_rate: 85.9,
            execution_time_ms: 1,
        });
        
        Self {
            strategies,
            flash_engine: AtomicMicroFlashEngine::new(),
            active_executions: Vec::new(),
            performance_tracker: PerformanceTracker::default(),
        }
    }
    
    pub async fn execute_strategy(&mut self, strategy_id: &str, capital: f64) -> Result<ExecutionResult> {
        let strategy = self.strategies.get(strategy_id)
            .ok_or_else(|| anyhow::anyhow!("Strategy not found: {}", strategy_id))?;
        
        if !strategy.enabled {
            return Err(anyhow::anyhow!("Strategy is disabled: {}", strategy_id));
        }
        
        if capital < strategy.min_capital || capital > strategy.max_capital {
            return Err(anyhow::anyhow!("Capital {} outside strategy limits [{}, {}]", 
                                     capital, strategy.min_capital, strategy.max_capital));
        }
        
        info!("🚀 Executing strategy: {} with {} SOL", strategy.name, capital);
        
        // Add to active executions
        let execution = ActiveExecution {
            strategy_id: strategy_id.to_string(),
            start_time: Instant::now(),
            capital_used: capital,
            status: ExecutionStatus::Running,
        };
        self.active_executions.push(execution);
        
        // Execute via flash engine
        let result = match strategy.strategy_type {
            StrategyType::FlashLoan => {
                self.flash_engine.execute_strategy(&strategy_id, capital).await?
            },
            StrategyType::Arbitrage => {
                self.flash_engine.execute_strategy("lightning_arbitrage_matrix", capital).await?
            },
            StrategyType::MemecoinSniping => {
                self.flash_engine.execute_strategy("atomic_memecoin_hunter", capital).await?
            },
            StrategyType::ConsciousnessTrading => {
                self.flash_engine.execute_strategy("consciousness_arbitrage", capital).await?
            },
            _ => {
                self.flash_engine.execute_strategy(&strategy_id, capital).await?
            }
        };
        
        // Update performance tracking
        self.performance_tracker.total_strategies_executed += 1;
        if result.success {
            self.performance_tracker.successful_strategies += 1;
            self.performance_tracker.total_profit += result.profit;
        } else {
            self.performance_tracker.total_loss += result.profit.abs();
        }
        
        // Update active execution status
        if let Some(execution) = self.active_executions.last_mut() {
            execution.status = ExecutionStatus::Completed(result.clone());
        }
        
        info!("✅ Strategy {} completed: {} SOL profit in {}μs", 
              strategy.name, result.profit, result.execution_time_micros);
        
        Ok(result)
    }
    
    pub async fn run_autonomous_trading(&mut self) -> Result<()> {
        info!("🤖 Starting autonomous trading with all strategies");
        
        let enabled_strategies: Vec<_> = self.strategies.iter()
            .filter(|(_, strategy)| strategy.enabled)
            .map(|(id, _)| id.clone())
            .collect();
        
        loop {
            for strategy_id in &enabled_strategies {
                // Execute with zero capital (flash borrowed)
                match self.execute_strategy(strategy_id, 0.0).await {
                    Ok(result) => {
                        if result.success {
                            info!("💰 {} profit: +{:.4} SOL", strategy_id, result.profit);
                        }
                    },
                    Err(e) => warn!("⚠️ Strategy {} failed: {}", strategy_id, e),
                }
                
                // Brief pause between strategies
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            
            // Longer pause between cycles
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    
    pub fn get_strategy_performance(&self) -> StrategyPerformanceReport {
        let win_rate = if self.performance_tracker.total_strategies_executed > 0 {
            (self.performance_tracker.successful_strategies as f64 / 
             self.performance_tracker.total_strategies_executed as f64) * 100.0
        } else {
            0.0
        };
        
        let net_profit = self.performance_tracker.total_profit - self.performance_tracker.total_loss;
        
        StrategyPerformanceReport {
            total_executions: self.performance_tracker.total_strategies_executed,
            successful_executions: self.performance_tracker.successful_strategies,
            win_rate,
            total_profit: self.performance_tracker.total_profit,
            total_loss: self.performance_tracker.total_loss,
            net_profit,
            active_strategies: self.strategies.len(),
            enabled_strategies: self.strategies.values().filter(|s| s.enabled).count(),
        }
    }
    
    pub fn enable_strategy(&mut self, strategy_id: &str) -> Result<()> {
        if let Some(strategy) = self.strategies.get_mut(strategy_id) {
            strategy.enabled = true;
            info!("✅ Enabled strategy: {}", strategy.name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Strategy not found: {}", strategy_id))
        }
    }
    
    pub fn disable_strategy(&mut self, strategy_id: &str) -> Result<()> {
        if let Some(strategy) = self.strategies.get_mut(strategy_id) {
            strategy.enabled = false;
            warn!("🚫 Disabled strategy: {}", strategy.name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Strategy not found: {}", strategy_id))
        }
    }
    
    pub fn list_strategies(&self) -> Vec<&Strategy> {
        self.strategies.values().collect()
    }
    
    pub fn get_active_executions(&self) -> &Vec<ActiveExecution> {
        &self.active_executions
    }
}

#[derive(Debug)]
pub struct StrategyPerformanceReport {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub win_rate: f64,
    pub total_profit: f64,
    pub total_loss: f64,
    pub net_profit: f64,
    pub active_strategies: usize,
    pub enabled_strategies: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_strategy_execution() {
        let mut engine = StrategyEngine::new();
        let result = engine.execute_strategy("quantum_cascade", 0.0).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_strategy_management() {
        let mut engine = StrategyEngine::new();
        assert!(engine.disable_strategy("quantum_cascade").is_ok());
        assert!(engine.enable_strategy("quantum_cascade").is_ok());
    }
}

