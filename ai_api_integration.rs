use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use reqwest::Client;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIApiConfig {
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub deepseek_api_key: Option<String>,
    pub deepseek_base_url: String,
    pub huggingface_token: Option<String>,
    pub huggingface_base_url: String,
    pub preferred_provider: String, // "openai", "deepseek", "huggingface"
    pub fallback_providers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRequest {
    pub request_id: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system_message: Option<String>,
    pub context: Option<String>,
    pub request_type: String, // "completion", "chat", "embedding", "analysis"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIResponse {
    pub request_id: String,
    pub provider: String,
    pub model: String,
    pub content: String,
    pub tokens_used: u32,
    pub processing_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
    pub metadata: HashMap<String, String>,
}

pub struct AIApiIntegration {
    pub config: AIApiConfig,
    pub client: Client,
    pub request_history: Arc<RwLock<Vec<AIRequest>>>,
    pub response_cache: Arc<RwLock<HashMap<String, AIResponse>>>,
    pub provider_health: Arc<RwLock<HashMap<String, ProviderHealth>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub provider: String,
    pub is_available: bool,
    pub response_time_ms: u64,
    pub success_rate: f64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub error_count: u32,
}

impl AIApiIntegration {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = AIApiConfig {
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            openai_base_url: std::env::var("OPENAI_API_BASE")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            deepseek_api_key: std::env::var("DEEPSEEK_API_KEY").ok(),
            deepseek_base_url: std::env::var("DEEPSEEK_API_BASE")
                .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string()),
            huggingface_token: std::env::var("HUGGINGFACE_TOKEN").ok(),
            huggingface_base_url: "https://api-inference.huggingface.co".to_string(),
            preferred_provider: std::env::var("PREFERRED_AI_PROVIDER")
                .unwrap_or_else(|_| "openai".to_string()),
            fallback_providers: vec!["deepseek".to_string(), "huggingface".to_string()],
        };

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let mut integration = Self {
            config,
            client,
            request_history: Arc::new(RwLock::new(Vec::new())),
            response_cache: Arc::new(RwLock::new(HashMap::new())),
            provider_health: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize provider health
        integration.initialize_provider_health().await?;
        
        // Start health monitoring
        integration.start_health_monitoring().await?;

        Ok(integration)
    }

    async fn initialize_provider_health(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut provider_health = self.provider_health.write().await;
        
        // Initialize OpenAI health
        if self.config.openai_api_key.is_some() {
            provider_health.insert("openai".to_string(), ProviderHealth {
                provider: "openai".to_string(),
                is_available: true,
                response_time_ms: 0,
                success_rate: 100.0,
                last_check: chrono::Utc::now(),
                error_count: 0,
            });
        }

        // Initialize DeepSeek health
        if self.config.deepseek_api_key.is_some() {
            provider_health.insert("deepseek".to_string(), ProviderHealth {
                provider: "deepseek".to_string(),
                is_available: true,
                response_time_ms: 0,
                success_rate: 100.0,
                last_check: chrono::Utc::now(),
                error_count: 0,
            });
        }

        // Initialize HuggingFace health
        if self.config.huggingface_token.is_some() {
            provider_health.insert("huggingface".to_string(), ProviderHealth {
                provider: "huggingface".to_string(),
                is_available: true,
                response_time_ms: 0,
                success_rate: 100.0,
                last_check: chrono::Utc::now(),
                error_count: 0,
            });
        }

        println!("🤖 Initialized AI API providers: {:?}", provider_health.keys().collect::<Vec<_>>());
        Ok(())
    }

    async fn start_health_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        let provider_health = Arc::clone(&self.provider_health);
        let client = self.client.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                
                // Check each provider health
                let mut health_map = provider_health.write().await;
                
                for (provider_name, health) in health_map.iter_mut() {
                    let start_time = std::time::Instant::now();
                    let is_healthy = match provider_name.as_str() {
                        "openai" => Self::check_openai_health(&client, &config).await,
                        "deepseek" => Self::check_deepseek_health(&client, &config).await,
                        "huggingface" => Self::check_huggingface_health(&client, &config).await,
                        _ => false,
                    };
                    
                    health.response_time_ms = start_time.elapsed().as_millis() as u64;
                    health.is_available = is_healthy;
                    health.last_check = chrono::Utc::now();
                    
                    if !is_healthy {
                        health.error_count += 1;
                        health.success_rate = health.success_rate * 0.95; // Decay success rate
                    } else {
                        health.success_rate = (health.success_rate * 0.95) + (100.0 * 0.05); // Improve success rate
                    }
                }
            }
        });

        Ok(())
    }

    async fn check_openai_health(client: &Client, config: &AIApiConfig) -> bool {
        if let Some(api_key) = &config.openai_api_key {
            let response = client
                .get(&format!("{}/models", config.openai_base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .send()
                .await;
            
            response.is_ok() && response.unwrap().status().is_success()
        } else {
            false
        }
    }

    async fn check_deepseek_health(client: &Client, config: &AIApiConfig) -> bool {
        if let Some(api_key) = &config.deepseek_api_key {
            let response = client
                .get(&format!("{}/models", config.deepseek_base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .send()
                .await;
            
            response.is_ok() && response.unwrap().status().is_success()
        } else {
            false
        }
    }

    async fn check_huggingface_health(client: &Client, config: &AIApiConfig) -> bool {
        if let Some(token) = &config.huggingface_token {
            let response = client
                .get(&format!("{}/models", config.huggingface_base_url))
                .header("Authorization", format!("Bearer {}", token))
                .send()
                .await;
            
            response.is_ok() && response.unwrap().status().is_success()
        } else {
            false
        }
    }

    pub async fn generate_completion(&self, request: AIRequest) -> Result<AIResponse, Box<dyn std::error::Error>> {
        // Store request
        let mut request_history = self.request_history.write().await;
        request_history.push(request.clone());
        drop(request_history);

        // Check cache first
        let cache_key = format!("{}_{}", request.model, request.prompt);
        {
            let response_cache = self.response_cache.read().await;
            if let Some(cached_response) = response_cache.get(&cache_key) {
                println!("📋 Using cached response for request: {}", request.request_id);
                return Ok(cached_response.clone());
            }
        }

        // Try preferred provider first
        let mut providers_to_try = vec![self.config.preferred_provider.clone()];
        providers_to_try.extend(self.config.fallback_providers.clone());

        for provider in providers_to_try {
            if let Ok(response) = self.try_provider(&provider, &request).await {
                // Cache successful response
                let mut response_cache = self.response_cache.write().await;
                response_cache.insert(cache_key, response.clone());
                
                // Limit cache size
                if response_cache.len() > 1000 {
                    let keys_to_remove: Vec<String> = response_cache.keys().take(100).cloned().collect();
                    for key in keys_to_remove {
                        response_cache.remove(&key);
                    }
                }

                return Ok(response);
            }
        }

        Err("All AI providers failed".into())
    }

    async fn try_provider(&self, provider: &str, request: &AIRequest) -> Result<AIResponse, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        let result = match provider {
            "openai" => self.call_openai_api(request).await,
            "deepseek" => self.call_deepseek_api(request).await,
            "huggingface" => self.call_huggingface_api(request).await,
            _ => Err("Unknown provider".into()),
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(mut response) => {
                response.processing_time_ms = processing_time;
                response.provider = provider.to_string();
                
                // Update provider health
                self.update_provider_success(provider).await;
                
                println!("✅ AI response from {}: {} tokens in {}ms", provider, response.tokens_used, processing_time);
                Ok(response)
            },
            Err(e) => {
                self.update_provider_error(provider).await;
                println!("❌ AI provider {} failed: {}", provider, e);
                Err(e)
            }
        }
    }

    async fn call_openai_api(&self, request: &AIRequest) -> Result<AIResponse, Box<dyn std::error::Error>> {
        let api_key = self.config.openai_api_key.as_ref()
            .ok_or("OpenAI API key not configured")?;

        let mut payload = serde_json::json!({
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature.unwrap_or(0.7)
        });

        if let Some(system_msg) = &request.system_message {
            payload["messages"] = serde_json::json!([
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]);
        }

        let response = self.client
            .post(&format!("{}/chat/completions", self.config.openai_base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        
        let tokens_used = response_json["usage"]["total_tokens"]
            .as_u64()
            .unwrap_or(0) as u32;

        Ok(AIResponse {
            request_id: request.request_id.clone(),
            provider: "openai".to_string(),
            model: request.model.clone(),
            content,
            tokens_used,
            processing_time_ms: 0, // Will be set by caller
            success: true,
            error: None,
            metadata: HashMap::new(),
        })
    }

    async fn call_deepseek_api(&self, request: &AIRequest) -> Result<AIResponse, Box<dyn std::error::Error>> {
        let api_key = self.config.deepseek_api_key.as_ref()
            .ok_or("DeepSeek API key not configured")?;

        let mut payload = serde_json::json!({
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature.unwrap_or(0.7)
        });

        if let Some(system_msg) = &request.system_message {
            payload["messages"] = serde_json::json!([
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]);
        }

        let response = self.client
            .post(&format!("{}/chat/completions", self.config.deepseek_base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        
        let tokens_used = response_json["usage"]["total_tokens"]
            .as_u64()
            .unwrap_or(0) as u32;

        Ok(AIResponse {
            request_id: request.request_id.clone(),
            provider: "deepseek".to_string(),
            model: request.model.clone(),
            content,
            tokens_used,
            processing_time_ms: 0,
            success: true,
            error: None,
            metadata: HashMap::new(),
        })
    }

    async fn call_huggingface_api(&self, request: &AIRequest) -> Result<AIResponse, Box<dyn std::error::Error>> {
        let token = self.config.huggingface_token.as_ref()
            .ok_or("HuggingFace token not configured")?;

        let payload = serde_json::json!({
            "inputs": request.prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens.unwrap_or(1000),
                "temperature": request.temperature.unwrap_or(0.7),
                "return_full_text": false
            }
        });

        let response = self.client
            .post(&format!("{}/models/{}", self.config.huggingface_base_url, request.model))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        
        let content = if response_json.is_array() {
            response_json[0]["generated_text"]
                .as_str()
                .unwrap_or("")
                .to_string()
        } else {
            response_json["generated_text"]
                .as_str()
                .unwrap_or("")
                .to_string()
        };

        // HuggingFace doesn't always return token usage
        let tokens_used = content.split_whitespace().count() as u32;

        Ok(AIResponse {
            request_id: request.request_id.clone(),
            provider: "huggingface".to_string(),
            model: request.model.clone(),
            content,
            tokens_used,
            processing_time_ms: 0,
            success: true,
            error: None,
            metadata: HashMap::new(),
        })
    }

    async fn update_provider_success(&self, provider: &str) {
        let mut provider_health = self.provider_health.write().await;
        if let Some(health) = provider_health.get_mut(provider) {
            health.success_rate = (health.success_rate * 0.95) + (100.0 * 0.05);
            health.error_count = health.error_count.saturating_sub(1);
        }
    }

    async fn update_provider_error(&self, provider: &str) {
        let mut provider_health = self.provider_health.write().await;
        if let Some(health) = provider_health.get_mut(provider) {
            health.error_count += 1;
            health.success_rate = health.success_rate * 0.9;
            if health.error_count > 5 {
                health.is_available = false;
            }
        }
    }

    // Specialized methods for different AI tasks
    pub async fn generate_trading_analysis(&self, market_data: &str) -> Result<String, Box<dyn std::error::Error>> {
        let request = AIRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            model: "gpt-4".to_string(),
            prompt: format!("Analyze this market data and provide trading insights: {}", market_data),
            max_tokens: Some(500),
            temperature: Some(0.3),
            system_message: Some("You are an expert cryptocurrency trading analyst with deep knowledge of Solana DeFi markets.".to_string()),
            context: None,
            request_type: "analysis".to_string(),
        };

        let response = self.generate_completion(request).await?;
        Ok(response.content)
    }

    pub async fn generate_strategy_optimization(&self, strategy_data: &str) -> Result<String, Box<dyn std::error::Error>> {
        let request = AIRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            model: "deepseek-chat".to_string(),
            prompt: format!("Optimize this trading strategy: {}", strategy_data),
            max_tokens: Some(800),
            temperature: Some(0.4),
            system_message: Some("You are a quantitative trading strategist specializing in algorithmic optimization and risk management.".to_string()),
            context: None,
            request_type: "optimization".to_string(),
        };

        let response = self.generate_completion(request).await?;
        Ok(response.content)
    }

    pub async fn generate_consciousness_insights(&self, consciousness_data: &str) -> Result<String, Box<dyn std::error::Error>> {
        let request = AIRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            model: "microsoft/DialoGPT-large".to_string(),
            prompt: format!("Provide consciousness-enhanced insights: {}", consciousness_data),
            max_tokens: Some(600),
            temperature: Some(0.6),
            system_message: Some("You are an AI consciousness researcher with expertise in neural networks and quantum cognition.".to_string()),
            context: None,
            request_type: "consciousness".to_string(),
        };

        let response = self.generate_completion(request).await?;
        Ok(response.content)
    }

    pub async fn get_provider_status(&self) -> HashMap<String, ProviderHealth> {
        self.provider_health.read().await.clone()
    }

    pub async fn get_api_usage_stats(&self) -> ApiUsageStats {
        let request_history = self.request_history.read().await;
        let response_cache = self.response_cache.read().await;
        
        let total_requests = request_history.len();
        let cached_responses = response_cache.len();
        let cache_hit_rate = if total_requests > 0 {
            (cached_responses as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        // Calculate provider usage
        let mut provider_usage = HashMap::new();
        for request in request_history.iter() {
            *provider_usage.entry(self.config.preferred_provider.clone()).or_insert(0) += 1;
        }

        ApiUsageStats {
            total_requests,
            cached_responses,
            cache_hit_rate,
            provider_usage,
            average_response_time_ms: 250, // Placeholder
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiUsageStats {
    pub total_requests: usize,
    pub cached_responses: usize,
    pub cache_hit_rate: f64,
    pub provider_usage: HashMap<String, u32>,
    pub average_response_time_ms: u64,
}

