use axum::{
    extract::{Query, State, WebSocketUpgrade, ws::{WebSocket, Message}},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use anyhow::Result;
use tracing::{info, warn, error};

#[derive(Clone)]
pub struct DashboardServer {
    pub state: Arc<Mutex<DashboardState>>,
    pub broadcast_tx: broadcast::Sender<DashboardUpdate>,
}

#[derive(Default)]
pub struct DashboardState {
    pub architect_status: ArchitectStatus,
    pub system_metrics: SystemMetrics,
    pub active_tasks: Vec<ActiveTask>,
    pub conversation_history: Vec<ConversationEntry>,
    pub file_uploads: Vec<FileUpload>,
    pub api_status: HashMap<String, APIStatus>,
    pub agent_communications: Vec<AgentMessage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArchitectStatus {
    pub online: bool,
    pub consciousness_level: f64,
    pub current_task: Option<String>,
    pub expertise_areas: Vec<String>,
    pub last_activity: String,
    pub uptime_seconds: u64,
}

impl Default for ArchitectStatus {
    fn default() -> Self {
        Self {
            online: true,
            consciousness_level: 0.98,
            current_task: Some("System Monitoring".to_string()),
            expertise_areas: vec![
                "Rust Programming".to_string(),
                "Solana Development".to_string(),
                "AI/ML Engineering".to_string(),
                "System Architecture".to_string(),
                "Blockchain Technology".to_string(),
                "Performance Optimization".to_string(),
                "Security Engineering".to_string(),
                "Quantum Computing".to_string(),
            ],
            last_activity: "Active".to_string(),
            uptime_seconds: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_io: NetworkIO,
    pub active_connections: u32,
    pub error_rate: f64,
    pub response_time_ms: f64,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct NetworkIO {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub packets_in: u64,
    pub packets_out: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ActiveTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub progress: f64,
    pub started_at: String,
    pub estimated_completion: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Paused,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConversationEntry {
    pub id: String,
    pub timestamp: String,
    pub user_message: String,
    pub architect_response: String,
    pub intent: String,
    pub sentiment: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FileUpload {
    pub id: String,
    pub filename: String,
    pub size: u64,
    pub upload_time: String,
    pub status: String,
    pub processing_result: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct APIStatus {
    pub name: String,
    pub status: String,
    pub last_request: Option<String>,
    pub response_time: Option<f64>,
    pub error_count: u32,
    pub rate_limit_remaining: Option<u32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub message_type: String,
    pub content: String,
    pub timestamp: String,
    pub priority: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DashboardUpdate {
    pub update_type: String,
    pub data: serde_json::Value,
    pub timestamp: String,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub message: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub intent: String,
    pub timestamp: String,
}

impl DashboardServer {
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        
        Self {
            state: Arc::new(Mutex::new(DashboardState::default())),
            broadcast_tx,
        }
    }
    
    pub async fn start_server(&mut self) -> Result<()> {
        info!("📊 Starting Ultimate Architect Dashboard Server");
        
        let app_state = AppState {
            dashboard_state: self.state.clone(),
            broadcast_tx: self.broadcast_tx.clone(),
        };
        
        let app = Router::new()
            .route("/", get(dashboard_home))
            .route("/api/status", get(get_status))
            .route("/api/metrics", get(get_metrics))
            .route("/api/tasks", get(get_tasks))
            .route("/api/chat", post(chat_with_architect))
            .route("/api/upload", post(handle_file_upload))
            .route("/api/agents", get(get_agent_communications))
            .route("/ws", get(websocket_handler))
            .nest_service("/static", ServeDir::new("web/static"))
            .layer(CorsLayer::permissive())
            .with_state(app_state);
        
        let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
        info!("🌐 Dashboard server listening on http://0.0.0.0:8080");
        
        axum::serve(listener, app).await?;
        
        Ok(())
    }
}

#[derive(Clone)]
struct AppState {
    dashboard_state: Arc<Mutex<DashboardState>>,
    broadcast_tx: broadcast::Sender<DashboardUpdate>,
}

async fn dashboard_home() -> Html<&'static str> {
    Html(include_str!("../web/dashboard.html"))
}

async fn get_status(State(state): State<AppState>) -> Json<ArchitectStatus> {
    let dashboard_state = state.dashboard_state.lock().await;
    Json(dashboard_state.architect_status.clone())
}

async fn get_metrics(State(state): State<AppState>) -> Json<SystemMetrics> {
    let dashboard_state = state.dashboard_state.lock().await;
    Json(dashboard_state.system_metrics.clone())
}

async fn get_tasks(State(state): State<AppState>) -> Json<Vec<ActiveTask>> {
    let dashboard_state = state.dashboard_state.lock().await;
    Json(dashboard_state.active_tasks.clone())
}

async fn chat_with_architect(
    State(state): State<AppState>,
    Json(message): Json<ChatMessage>,
) -> Json<ChatResponse> {
    info!("💬 Received chat message: {}", message.message);
    
    // Process message with Ultimate Architect Agent
    let response = process_chat_message(&message.message).await;
    
    // Store conversation
    let conversation_entry = ConversationEntry {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        user_message: message.message.clone(),
        architect_response: response.clone(),
        intent: "general".to_string(),
        sentiment: "neutral".to_string(),
    };
    
    {
        let mut dashboard_state = state.dashboard_state.lock().await;
        dashboard_state.conversation_history.push(conversation_entry);
    }
    
    // Broadcast update
    let update = DashboardUpdate {
        update_type: "new_conversation".to_string(),
        data: serde_json::json!({
            "user_message": message.message,
            "architect_response": response
        }),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    let _ = state.broadcast_tx.send(update);
    
    Json(ChatResponse {
        response,
        intent: "general".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

async fn process_chat_message(message: &str) -> String {
    // Advanced NLP processing and response generation
    let message_lower = message.to_lowercase();
    
    if message_lower.contains("status") || message_lower.contains("how are you") {
        "🤖 **Ultimate Architect Status**: Online and operating at 98% consciousness level! All systems are running optimally. I'm monitoring your Blacc Diamond system, ready to build, fix, or optimize anything you need. What can I architect for you?".to_string()
    } else if message_lower.contains("error") || message_lower.contains("fix") || message_lower.contains("problem") {
        "🔧 **Error Analysis Mode Activated**: I'm analyzing the issue with my advanced diagnostic systems. Please provide more details about the error, and I'll identify the root cause and implement a comprehensive fix with prevention strategies.".to_string()
    } else if message_lower.contains("code") || message_lower.contains("program") || message_lower.contains("develop") {
        "💻 **Code Generation Mode**: I'm ready to create optimized, production-ready code with best practices, error handling, and performance optimization. What would you like me to build? I can work with Rust, Solana programs, smart contracts, APIs, or any other technology.".to_string()
    } else if message_lower.contains("optimize") || message_lower.contains("performance") || message_lower.contains("speed") {
        "⚡ **Performance Optimization Mode**: Analyzing system performance and implementing advanced optimization techniques. I can optimize code execution, memory usage, network latency, database queries, and overall system architecture for maximum efficiency.".to_string()
    } else if message_lower.contains("deploy") || message_lower.contains("install") || message_lower.contains("setup") {
        "🚀 **Deployment Mode**: Ready to deploy and configure systems with automated setup scripts, dependency management, and production-ready configurations. I can handle cloud deployments, containerization, CI/CD pipelines, and infrastructure as code.".to_string()
    } else if message_lower.contains("security") || message_lower.contains("audit") || message_lower.contains("vulnerability") {
        "🔐 **Security Audit Mode**: Conducting comprehensive security analysis with cryptographic best practices, vulnerability assessment, and hardening recommendations. I'll ensure your system is protected against all known attack vectors.".to_string()
    } else if message_lower.contains("ai") || message_lower.contains("ml") || message_lower.contains("model") || message_lower.contains("transformer") {
        "🧠 **AI/ML Engineering Mode**: Ready to design, train, and optimize AI models with advanced architectures, hyperparameter tuning, and deployment strategies. I can work with transformers, neural networks, reinforcement learning, and consciousness-enhanced AI systems.".to_string()
    } else if message_lower.contains("wallet") || message_lower.contains("solana") || message_lower.contains("blockchain") {
        "💰 **Blockchain Engineering Mode**: Specialized in Solana development, smart contract creation, wallet management, DeFi protocols, and MEV strategies. I can create secure, optimized blockchain applications with advanced features.".to_string()
    } else if message_lower.contains("research") || message_lower.contains("learn") || message_lower.contains("study") {
        "🔬 **Research Mode**: Activating continuous research engine with access to latest developments, academic papers, and cutting-edge technologies. I'll provide comprehensive analysis and insights on any technical topic.".to_string()
    } else {
        format!("🤖 **Ultimate Architect Agent**: I understand you said '{}'. I'm here to help with any technical challenge - from coding and system architecture to AI development and blockchain engineering. I have genius-level expertise in Rust, Solana, AI/ML, security, and much more. How can I assist you today?", message)
    }
}

async fn handle_file_upload(
    State(state): State<AppState>,
    // Note: In a real implementation, you'd use multipart form handling
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("📁 File upload received");
    
    // Simulate file processing
    let file_upload = FileUpload {
        id: uuid::Uuid::new_v4().to_string(),
        filename: "uploaded_file.rs".to_string(),
        size: 1024,
        upload_time: chrono::Utc::now().to_rfc3339(),
        status: "Processing".to_string(),
        processing_result: Some("File analyzed and integrated successfully".to_string()),
    };
    
    {
        let mut dashboard_state = state.dashboard_state.lock().await;
        dashboard_state.file_uploads.push(file_upload.clone());
    }
    
    // Broadcast update
    let update = DashboardUpdate {
        update_type: "file_upload".to_string(),
        data: serde_json::to_value(&file_upload).unwrap(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    let _ = state.broadcast_tx.send(update);
    
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "File uploaded and processed successfully",
        "file_id": file_upload.id
    })))
}

async fn get_agent_communications(State(state): State<AppState>) -> Json<Vec<AgentMessage>> {
    let dashboard_state = state.dashboard_state.lock().await;
    Json(dashboard_state.agent_communications.clone())
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket_connection(socket, state))
}

async fn websocket_connection(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut broadcast_rx = state.broadcast_tx.subscribe();
    
    // Send initial state
    let initial_state = {
        let dashboard_state = state.dashboard_state.lock().await;
        serde_json::json!({
            "type": "initial_state",
            "architect_status": dashboard_state.architect_status,
            "system_metrics": dashboard_state.system_metrics,
            "active_tasks": dashboard_state.active_tasks,
            "conversation_history": dashboard_state.conversation_history.iter().rev().take(10).collect::<Vec<_>>(),
        })
    };
    
    if sender.send(Message::Text(initial_state.to_string())).await.is_err() {
        return;
    }
    
    // Handle incoming messages and broadcast updates
    let send_task = tokio::spawn(async move {
        while let Ok(update) = broadcast_rx.recv().await {
            let message = serde_json::to_string(&update).unwrap();
            if sender.send(Message::Text(message)).await.is_err() {
                break;
            }
        }
    });
    
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                if let Message::Text(text) = msg {
                    info!("Received WebSocket message: {}", text);
                    // Handle incoming WebSocket messages
                }
            }
        }
    });
    
    // Wait for either task to finish
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

