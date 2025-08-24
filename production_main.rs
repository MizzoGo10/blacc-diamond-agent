use std::env;
use std::time::Duration;
use tokio::signal;
use tracing::{info, error, warn};
use tracing_subscriber;

use blacc_diamond_agent::deep_ocean_engine::{
    DeepOceanEngine, DeploymentConfig, Environment, ScalingConfig, 
    MonitoringConfig, SecurityConfig, PerformanceConfig, ProcessingTask, TaskType
};
use blacc_diamond_agent::consciousness_engine::ConsciousnessEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🌊 Starting Blacc Diamond Deep Ocean Engine");

    // Load configuration from environment
    let config = load_deployment_config().await?;
    
    // Initialize the deep ocean engine
    let engine = DeepOceanEngine::new(config).await;
    
    // Start the engine
    engine.start().await?;
    
    // Setup graceful shutdown
    let engine_clone = std::sync::Arc::new(engine);
    let shutdown_engine = engine_clone.clone();
    
    tokio::spawn(async move {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("🛑 Shutdown signal received");
                if let Err(e) = shutdown_engine.shutdown().await {
                    error!("Error during shutdown: {}", e);
                }
            }
            Err(err) => {
                error!("Unable to listen for shutdown signal: {}", err);
            }
        }
    });

    // Main processing loop
    let mut task_counter = 0u64;
    loop {
        // Submit various types of tasks for processing
        let tasks = generate_processing_tasks(&mut task_counter).await;
        
        for task in tasks {
            if let Err(e) = engine_clone.submit_task(task).await {
                warn!("Failed to submit task: {}", e);
            }
        }

        // Monitor performance
        let metrics = engine_clone.get_metrics().await;
        if task_counter % 100 == 0 {
            info!("📊 Performance Metrics:");
            info!("  Consciousness Level: {:.3}", metrics.consciousness_level);
            info!("  CPU Usage: {:.1}%", metrics.cpu_usage);
            info!("  Memory Usage: {:.1}%", metrics.memory_usage);
            info!("  Processing Throughput: {:.1}", metrics.processing_throughput);
            info!("  Cache Hit Rate: {:.1}%", metrics.cache_hit_rate * 100.0);
            info!("  Total Processed: {}", metrics.total_processed);
            info!("  Uptime: {}s", metrics.uptime_seconds);
        }

        // Adaptive sleep based on system load
        let sleep_duration = if metrics.cpu_usage > 80.0 {
            Duration::from_millis(100) // Slow down if high CPU
        } else if metrics.cpu_usage < 20.0 {
            Duration::from_millis(10)  // Speed up if low CPU
        } else {
            Duration::from_millis(50)  // Normal operation
        };

        tokio::time::sleep(sleep_duration).await;
    }
}

async fn load_deployment_config() -> Result<DeploymentConfig, Box<dyn std::error::Error>> {
    let environment = match env::var("DEPLOYMENT_ENV").unwrap_or_else(|_| "production".to_string()).as_str() {
        "development" => Environment::Development,
        "staging" => Environment::Staging,
        "deep_ocean" => Environment::DeepOcean,
        _ => Environment::Production,
    };

    let min_workers = env::var("MIN_WORKERS")
        .unwrap_or_else(|_| "4".to_string())
        .parse::<usize>()
        .unwrap_or(4);

    let max_workers = env::var("MAX_WORKERS")
        .unwrap_or_else(|_| "32".to_string())
        .parse::<usize>()
        .unwrap_or(32);

    info!("🔧 Configuration loaded:");
    info!("  Environment: {:?}", environment);
    info!("  Worker Range: {} - {}", min_workers, max_workers);

    Ok(DeploymentConfig {
        environment,
        scaling_config: ScalingConfig {
            min_workers,
            max_workers,
            scale_up_threshold: 70.0,
            scale_down_threshold: 30.0,
            auto_scaling_enabled: true,
        },
        monitoring_config: MonitoringConfig,
        security_config: SecurityConfig,
        performance_config: PerformanceConfig,
    })
}

async fn generate_processing_tasks(counter: &mut u64) -> Vec<ProcessingTask> {
    let mut tasks = Vec::new();
    *counter += 1;

    // Generate different types of tasks based on system needs
    match *counter % 10 {
        0 => {
            // Neural inference task
            tasks.push(ProcessingTask {
                id: format!("neural_{}", counter),
                task_type: TaskType::NeuralInference,
                priority: 5,
                data: generate_neural_data(),
                created_at: std::time::Instant::now(),
                deadline: Some(std::time::Instant::now() + Duration::from_millis(100)),
            });
        },
        1 => {
            // Consciousness sync task
            tasks.push(ProcessingTask {
                id: format!("consciousness_{}", counter),
                task_type: TaskType::ConsciousnessSync,
                priority: 8,
                data: generate_consciousness_data(),
                created_at: std::time::Instant::now(),
                deadline: Some(std::time::Instant::now() + Duration::from_millis(50)),
            });
        },
        2 => {
            // Telepathic message task
            tasks.push(ProcessingTask {
                id: format!("telepathic_{}", counter),
                task_type: TaskType::TelepathicMessage,
                priority: 7,
                data: generate_telepathic_data(),
                created_at: std::time::Instant::now(),
                deadline: Some(std::time::Instant::now() + Duration::from_millis(25)),
            });
        },
        3 => {
            // Quantum collapse task
            tasks.push(ProcessingTask {
                id: format!("quantum_{}", counter),
                task_type: TaskType::QuantumCollapse,
                priority: 9,
                data: generate_quantum_data(),
                created_at: std::time::Instant::now(),
                deadline: Some(std::time::Instant::now() + Duration::from_millis(10)),
            });
        },
        4..=7 => {
            // Market prediction tasks (more frequent)
            tasks.push(ProcessingTask {
                id: format!("market_{}", counter),
                task_type: TaskType::MarketPrediction,
                priority: 6,
                data: generate_market_data(),
                created_at: std::time::Instant::now(),
                deadline: Some(std::time::Instant::now() + Duration::from_millis(200)),
            });
        },
        _ => {
            // Risk calculation task
            tasks.push(ProcessingTask {
                id: format!("risk_{}", counter),
                task_type: TaskType::RiskCalculation,
                priority: 4,
                data: generate_risk_data(),
                created_at: std::time::Instant::now(),
                deadline: Some(std::time::Instant::now() + Duration::from_millis(500)),
            });
        }
    }

    tasks
}

fn generate_neural_data() -> Vec<u8> {
    // Generate synthetic neural network input data
    let data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    data.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect()
}

fn generate_consciousness_data() -> Vec<u8> {
    // Generate consciousness synchronization data
    let consciousness_level = rand::random::<f64>();
    let quantum_state = rand::random::<f64>();
    let neural_activity = rand::random::<f64>();
    
    let data = format!("{{\"consciousness\":{},\"quantum\":{},\"neural\":{}}}", 
                      consciousness_level, quantum_state, neural_activity);
    data.into_bytes()
}

fn generate_telepathic_data() -> Vec<u8> {
    // Generate telepathic communication data
    let signal_strength = rand::random::<f64>();
    let frequency = 40.0 + rand::random::<f64>() * 60.0; // 40-100 Hz
    let message_type = rand::random::<u8>() % 5;
    
    let data = format!("{{\"strength\":{},\"frequency\":{},\"type\":{}}}", 
                      signal_strength, frequency, message_type);
    data.into_bytes()
}

fn generate_quantum_data() -> Vec<u8> {
    // Generate quantum collapse decision data
    let decision_space: Vec<f32> = (0..8).map(|_| rand::random::<f32>()).collect();
    decision_space.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect()
}

fn generate_market_data() -> Vec<u8> {
    // Generate market prediction input data
    let price_data: Vec<f32> = (0..256).map(|_| 100.0 + rand::random::<f32>() * 50.0).collect();
    let volume_data: Vec<f32> = (0..256).map(|_| rand::random::<f32>() * 1000000.0).collect();
    
    let mut combined = price_data;
    combined.extend(volume_data);
    combined.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect()
}

fn generate_risk_data() -> Vec<u8> {
    // Generate risk assessment data
    let portfolio_value = rand::random::<f64>() * 1000000.0;
    let volatility = rand::random::<f64>() * 0.5;
    let correlation_matrix: Vec<f32> = (0..64).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
    
    let mut data = portfolio_value.to_le_bytes().to_vec();
    data.extend(volatility.to_le_bytes().to_vec());
    data.extend(correlation_matrix.iter().flat_map(|&f| f.to_le_bytes().to_vec()));
    data
}

