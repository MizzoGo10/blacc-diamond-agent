use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, Instant};
use anyhow::Result;
use tracing::{info, warn, error, debug};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use reqwest::Client;
use serde_json::{Value, json};

#[derive(Clone)]
pub struct RPCManager {
    pub providers: Arc<RwLock<Vec<RPCProvider>>>,
    pub cache: Arc<RwLock<RPCCache>>,
    pub rate_limiter: Arc<Mutex<RateLimiter>>,
    pub usage_tracker: Arc<Mutex<UsageTracker>>,
    pub client: Client,
    pub current_provider_index: Arc<Mutex<usize>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RPCProvider {
    pub name: String,
    pub url: String,
    pub websocket_url: Option<String>,
    pub api_key: Option<String>,
    pub priority: u8, // 1 = highest priority
    pub rate_limit: RateLimit,
    pub status: ProviderStatus,
    pub last_error: Option<String>,
    pub error_count: u32,
    pub success_count: u32,
    pub average_response_time: f64,
    pub monthly_usage: u64,
    pub monthly_limit: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_second: u32,
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
    pub requests_per_month: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ProviderStatus {
    Active,
    Degraded,
    Failed,
    RateLimited,
    MonthlyLimitReached,
}

pub struct RPCCache {
    pub account_cache: HashMap<String, CachedAccountData>,
    pub transaction_cache: HashMap<String, CachedTransactionData>,
    pub block_cache: HashMap<u64, CachedBlockData>,
    pub token_cache: HashMap<String, CachedTokenData>,
    pub program_cache: HashMap<String, CachedProgramData>,
    pub price_cache: HashMap<String, CachedPriceData>,
    pub cache_stats: CacheStats,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedAccountData {
    pub data: Value,
    pub cached_at: Instant,
    pub ttl: Duration,
    pub access_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedTransactionData {
    pub data: Value,
    pub cached_at: Instant,
    pub ttl: Duration,
    pub access_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedBlockData {
    pub data: Value,
    pub cached_at: Instant,
    pub ttl: Duration,
    pub access_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedTokenData {
    pub data: Value,
    pub cached_at: Instant,
    pub ttl: Duration,
    pub access_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedProgramData {
    pub data: Value,
    pub cached_at: Instant,
    pub ttl: Duration,
    pub access_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedPriceData {
    pub data: Value,
    pub cached_at: Instant,
    pub ttl: Duration,
    pub access_count: u32,
}

#[derive(Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_ratio: f64,
    pub total_saved_requests: u64,
    pub cache_size_bytes: u64,
}

pub struct RateLimiter {
    pub provider_limits: HashMap<String, ProviderRateLimit>,
    pub global_limit: GlobalRateLimit,
}

pub struct ProviderRateLimit {
    pub requests_this_second: u32,
    pub requests_this_minute: u32,
    pub requests_this_hour: u32,
    pub requests_this_day: u32,
    pub requests_this_month: u32,
    pub last_reset_second: Instant,
    pub last_reset_minute: Instant,
    pub last_reset_hour: Instant,
    pub last_reset_day: Instant,
    pub last_reset_month: Instant,
}

pub struct GlobalRateLimit {
    pub max_concurrent_requests: u32,
    pub current_concurrent_requests: u32,
    pub request_queue: Vec<QueuedRequest>,
}

pub struct QueuedRequest {
    pub id: String,
    pub method: String,
    pub params: Value,
    pub queued_at: Instant,
    pub priority: RequestPriority,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

pub struct UsageTracker {
    pub daily_usage: HashMap<String, u64>,
    pub monthly_usage: HashMap<String, u64>,
    pub cost_tracking: HashMap<String, f64>,
    pub performance_metrics: HashMap<String, PerformanceMetrics>,
}

#[derive(Default)]
pub struct PerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: f64,
    pub min_response_time: f64,
    pub max_response_time: f64,
    pub last_24h_requests: u64,
}

impl RPCManager {
    pub fn new() -> Self {
        let providers = vec![
            // Primary: Quicknode (your existing)
            RPCProvider {
                name: "Quicknode".to_string(),
                url: "https://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/".to_string(),
                websocket_url: Some("wss://indulgent-greatest-wish.solana-mainnet.quiknode.pro/e39bfe4c67a145d55fe4c4e50e68c275fb45f99c/".to_string()),
                api_key: None,
                priority: 1,
                rate_limit: RateLimit {
                    requests_per_second: 50,
                    requests_per_minute: 2000,
                    requests_per_hour: 50000,
                    requests_per_day: 1000000,
                    requests_per_month: 25000000, // Adjust based on your plan
                },
                status: ProviderStatus::Active,
                last_error: None,
                error_count: 0,
                success_count: 0,
                average_response_time: 0.0,
                monthly_usage: 0,
                monthly_limit: 25000000,
            },
            
            // Backup 1: Helius (add your Helius RPC)
            RPCProvider {
                name: "Helius".to_string(),
                url: "https://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_API_KEY".to_string(),
                websocket_url: Some("wss://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_API_KEY".to_string()),
                api_key: Some("YOUR_HELIUS_API_KEY".to_string()),
                priority: 2,
                rate_limit: RateLimit {
                    requests_per_second: 100,
                    requests_per_minute: 3000,
                    requests_per_hour: 100000,
                    requests_per_day: 2000000,
                    requests_per_month: 50000000,
                },
                status: ProviderStatus::Active,
                last_error: None,
                error_count: 0,
                success_count: 0,
                average_response_time: 0.0,
                monthly_usage: 0,
                monthly_limit: 50000000,
            },
            
            // Backup 2: Alchemy (your existing)
            RPCProvider {
                name: "Alchemy".to_string(),
                url: "https://solana-mainnet.g.alchemy.com/v2/PPQbbM4WmrX_82GOP8QR5pJ_JsBvyLWR".to_string(),
                websocket_url: Some("wss://solana-mainnet.g.alchemy.com/v2/PPQbbM4WmrX_82GOP8QR5pJ_JsBvyLWR".to_string()),
                api_key: Some("PPQbbM4WmrX_82GOP8QR5pJ_JsBvyLWR".to_string()),
                priority: 3,
                rate_limit: RateLimit {
                    requests_per_second: 25,
                    requests_per_minute: 1000,
                    requests_per_hour: 30000,
                    requests_per_day: 500000,
                    requests_per_month: 10000000,
                },
                status: ProviderStatus::Active,
                last_error: None,
                error_count: 0,
                success_count: 0,
                average_response_time: 0.0,
                monthly_usage: 0,
                monthly_limit: 10000000,
            },
            
            // Backup 3: Public RPC (free, rate limited)
            RPCProvider {
                name: "Public".to_string(),
                url: "https://api.mainnet-beta.solana.com".to_string(),
                websocket_url: Some("wss://api.mainnet-beta.solana.com".to_string()),
                api_key: None,
                priority: 4,
                rate_limit: RateLimit {
                    requests_per_second: 5,
                    requests_per_minute: 100,
                    requests_per_hour: 1000,
                    requests_per_day: 10000,
                    requests_per_month: 100000,
                },
                status: ProviderStatus::Active,
                last_error: None,
                error_count: 0,
                success_count: 0,
                average_response_time: 0.0,
                monthly_usage: 0,
                monthly_limit: 100000,
            },
        ];
        
        Self {
            providers: Arc::new(RwLock::new(providers)),
            cache: Arc::new(RwLock::new(RPCCache {
                account_cache: HashMap::new(),
                transaction_cache: HashMap::new(),
                block_cache: HashMap::new(),
                token_cache: HashMap::new(),
                program_cache: HashMap::new(),
                price_cache: HashMap::new(),
                cache_stats: CacheStats::default(),
            })),
            rate_limiter: Arc::new(Mutex::new(RateLimiter {
                provider_limits: HashMap::new(),
                global_limit: GlobalRateLimit {
                    max_concurrent_requests: 50,
                    current_concurrent_requests: 0,
                    request_queue: Vec::new(),
                },
            })),
            usage_tracker: Arc::new(Mutex::new(UsageTracker {
                daily_usage: HashMap::new(),
                monthly_usage: HashMap::new(),
                cost_tracking: HashMap::new(),
                performance_metrics: HashMap::new(),
            })),
            client: Client::new(),
            current_provider_index: Arc::new(Mutex::new(0)),
        }
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("🔄 Initializing Intelligent RPC Manager");
        
        // Initialize rate limiters for each provider
        {
            let mut rate_limiter = self.rate_limiter.lock().await;
            let providers = self.providers.read().await;
            
            for provider in providers.iter() {
                rate_limiter.provider_limits.insert(
                    provider.name.clone(),
                    ProviderRateLimit {
                        requests_this_second: 0,
                        requests_this_minute: 0,
                        requests_this_hour: 0,
                        requests_this_day: 0,
                        requests_this_month: 0,
                        last_reset_second: Instant::now(),
                        last_reset_minute: Instant::now(),
                        last_reset_hour: Instant::now(),
                        last_reset_day: Instant::now(),
                        last_reset_month: Instant::now(),
                    }
                );
            }
        }
        
        // Test all providers
        self.test_all_providers().await?;
        
        // Start background tasks
        self.start_cache_cleanup_task().await;
        self.start_rate_limit_reset_task().await;
        self.start_provider_health_check().await;
        
        info!("✅ RPC Manager initialized with intelligent caching and rate limiting");
        info!("📊 {} providers configured with failover support", self.providers.read().await.len());
        
        Ok(())
    }
    
    pub async fn make_request(&self, method: &str, params: Value, priority: RequestPriority) -> Result<Value> {
        // Check cache first
        if let Some(cached_result) = self.check_cache(method, &params).await? {
            debug!("🎯 Cache hit for method: {}", method);
            return Ok(cached_result);
        }
        
        // Find best available provider
        let provider = self.select_best_provider().await?;
        
        // Check rate limits
        if !self.check_rate_limits(&provider.name).await? {
            warn!("⏱️ Rate limit exceeded for {}, trying next provider", provider.name);
            return self.make_request_with_fallback(method, params, priority).await;
        }
        
        // Make the request
        let start_time = Instant::now();
        let result = self.execute_request(&provider, method, params.clone()).await;
        let response_time = start_time.elapsed();
        
        match result {
            Ok(response) => {
                // Update provider metrics
                self.update_provider_success(&provider.name, response_time).await?;
                
                // Cache the result
                self.cache_result(method, &params, &response).await?;
                
                // Update usage tracking
                self.track_usage(&provider.name).await?;
                
                debug!("✅ Request successful via {} in {:?}", provider.name, response_time);
                Ok(response)
            },
            Err(e) => {
                warn!("❌ Request failed via {}: {}", provider.name, e);
                self.update_provider_error(&provider.name, &e.to_string()).await?;
                
                // Try fallback providers
                self.make_request_with_fallback(method, params, priority).await
            }
        }
    }
    
    async fn check_cache(&self, method: &str, params: &Value) -> Result<Option<Value>> {
        let cache = self.cache.read().await;
        let cache_key = self.generate_cache_key(method, params);
        
        let cached_data = match method {
            "getAccountInfo" => {
                cache.account_cache.get(&cache_key).map(|data| &data.data)
            },
            "getTransaction" => {
                cache.transaction_cache.get(&cache_key).map(|data| &data.data)
            },
            "getBlock" => {
                if let Some(slot) = params.get("slot").and_then(|s| s.as_u64()) {
                    cache.block_cache.get(&slot).map(|data| &data.data)
                } else {
                    None
                }
            },
            "getTokenAccountsByOwner" => {
                cache.token_cache.get(&cache_key).map(|data| &data.data)
            },
            "getProgramAccounts" => {
                cache.program_cache.get(&cache_key).map(|data| &data.data)
            },
            _ => None,
        };
        
        if let Some(data) = cached_data {
            // Check if cache is still valid
            if data.cached_at.elapsed() < data.ttl {
                // Update cache stats
                drop(cache);
                let mut cache_mut = self.cache.write().await;
                cache_mut.cache_stats.cache_hits += 1;
                cache_mut.cache_stats.total_requests += 1;
                cache_mut.cache_stats.cache_hit_ratio = 
                    cache_mut.cache_stats.cache_hits as f64 / cache_mut.cache_stats.total_requests as f64;
                
                return Ok(Some(data.data.clone()));
            }
        }
        
        // Cache miss
        let mut cache_mut = self.cache.write().await;
        cache_mut.cache_stats.cache_misses += 1;
        cache_mut.cache_stats.total_requests += 1;
        cache_mut.cache_stats.cache_hit_ratio = 
            cache_mut.cache_stats.cache_hits as f64 / cache_mut.cache_stats.total_requests as f64;
        
        Ok(None)
    }
    
    async fn cache_result(&self, method: &str, params: &Value, result: &Value) -> Result<()> {
        let mut cache = self.cache.write().await;
        let cache_key = self.generate_cache_key(method, params);
        
        let ttl = self.get_cache_ttl(method);
        let cached_data = match method {
            "getAccountInfo" => {
                let cached = CachedAccountData {
                    data: result.clone(),
                    cached_at: Instant::now(),
                    ttl,
                    access_count: 1,
                };
                cache.account_cache.insert(cache_key, cached);
            },
            "getTransaction" => {
                let cached = CachedTransactionData {
                    data: result.clone(),
                    cached_at: Instant::now(),
                    ttl,
                    access_count: 1,
                };
                cache.transaction_cache.insert(cache_key, cached);
            },
            "getBlock" => {
                if let Some(slot) = params.get("slot").and_then(|s| s.as_u64()) {
                    let cached = CachedBlockData {
                        data: result.clone(),
                        cached_at: Instant::now(),
                        ttl,
                        access_count: 1,
                    };
                    cache.block_cache.insert(slot, cached);
                }
            },
            "getTokenAccountsByOwner" => {
                let cached = CachedTokenData {
                    data: result.clone(),
                    cached_at: Instant::now(),
                    ttl,
                    access_count: 1,
                };
                cache.token_cache.insert(cache_key, cached);
            },
            "getProgramAccounts" => {
                let cached = CachedProgramData {
                    data: result.clone(),
                    cached_at: Instant::now(),
                    ttl,
                    access_count: 1,
                };
                cache.program_cache.insert(cache_key, cached);
            },
            _ => {}
        }
        
        Ok(())
    }
    
    fn get_cache_ttl(&self, method: &str) -> Duration {
        match method {
            "getAccountInfo" => Duration::from_secs(5), // Account data changes frequently
            "getTransaction" => Duration::from_secs(3600), // Transactions are immutable
            "getBlock" => Duration::from_secs(3600), // Blocks are immutable
            "getTokenAccountsByOwner" => Duration::from_secs(10), // Token balances change
            "getProgramAccounts" => Duration::from_secs(30), // Program accounts change moderately
            "getSlot" => Duration::from_secs(1), // Slot changes every 400ms
            "getBlockHeight" => Duration::from_secs(1), // Block height changes frequently
            "getBalance" => Duration::from_secs(5), // Balances change frequently
            _ => Duration::from_secs(60), // Default cache time
        }
    }
    
    fn generate_cache_key(&self, method: &str, params: &Value) -> String {
        format!("{}:{}", method, serde_json::to_string(params).unwrap_or_default())
    }
    
    async fn select_best_provider(&self) -> Result<RPCProvider> {
        let providers = self.providers.read().await;
        
        // Find the best available provider based on:
        // 1. Status (Active > Degraded > Failed)
        // 2. Priority (lower number = higher priority)
        // 3. Rate limit availability
        // 4. Response time
        
        let mut best_provider = None;
        let mut best_score = f64::MIN;
        
        for provider in providers.iter() {
            if matches!(provider.status, ProviderStatus::Failed | ProviderStatus::MonthlyLimitReached) {
                continue;
            }
            
            let status_score = match provider.status {
                ProviderStatus::Active => 100.0,
                ProviderStatus::Degraded => 50.0,
                ProviderStatus::RateLimited => 10.0,
                _ => 0.0,
            };
            
            let priority_score = 100.0 - (provider.priority as f64 * 10.0);
            let response_time_score = if provider.average_response_time > 0.0 {
                100.0 / provider.average_response_time
            } else {
                50.0
            };
            
            let usage_score = if provider.monthly_limit > 0 {
                100.0 * (1.0 - (provider.monthly_usage as f64 / provider.monthly_limit as f64))
            } else {
                100.0
            };
            
            let total_score = status_score + priority_score + response_time_score + usage_score;
            
            if total_score > best_score {
                best_score = total_score;
                best_provider = Some(provider.clone());
            }
        }
        
        best_provider.ok_or_else(|| anyhow::anyhow!("No available RPC providers"))
    }
    
    async fn check_rate_limits(&self, provider_name: &str) -> Result<bool> {
        let mut rate_limiter = self.rate_limiter.lock().await;
        
        if let Some(limits) = rate_limiter.provider_limits.get_mut(provider_name) {
            let now = Instant::now();
            
            // Reset counters if time periods have elapsed
            if now.duration_since(limits.last_reset_second) >= Duration::from_secs(1) {
                limits.requests_this_second = 0;
                limits.last_reset_second = now;
            }
            
            if now.duration_since(limits.last_reset_minute) >= Duration::from_secs(60) {
                limits.requests_this_minute = 0;
                limits.last_reset_minute = now;
            }
            
            if now.duration_since(limits.last_reset_hour) >= Duration::from_secs(3600) {
                limits.requests_this_hour = 0;
                limits.last_reset_hour = now;
            }
            
            if now.duration_since(limits.last_reset_day) >= Duration::from_secs(86400) {
                limits.requests_this_day = 0;
                limits.last_reset_day = now;
            }
            
            if now.duration_since(limits.last_reset_month) >= Duration::from_secs(2592000) {
                limits.requests_this_month = 0;
                limits.last_reset_month = now;
            }
            
            // Check if we can make another request
            let providers = self.providers.read().await;
            if let Some(provider) = providers.iter().find(|p| p.name == provider_name) {
                if limits.requests_this_second >= provider.rate_limit.requests_per_second ||
                   limits.requests_this_minute >= provider.rate_limit.requests_per_minute ||
                   limits.requests_this_hour >= provider.rate_limit.requests_per_hour ||
                   limits.requests_this_day >= provider.rate_limit.requests_per_day ||
                   limits.requests_this_month >= provider.rate_limit.requests_per_month {
                    return Ok(false);
                }
                
                // Increment counters
                limits.requests_this_second += 1;
                limits.requests_this_minute += 1;
                limits.requests_this_hour += 1;
                limits.requests_this_day += 1;
                limits.requests_this_month += 1;
            }
        }
        
        Ok(true)
    }
    
    async fn execute_request(&self, provider: &RPCProvider, method: &str, params: Value) -> Result<Value> {
        let request_body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });
        
        let mut request_builder = self.client
            .post(&provider.url)
            .header("Content-Type", "application/json")
            .json(&request_body);
        
        // Add API key if required
        if let Some(api_key) = &provider.api_key {
            if provider.name == "Helius" {
                // Helius uses query parameter
                request_builder = self.client
                    .post(&format!("{}?api-key={}", provider.url.split('?').next().unwrap_or(&provider.url), api_key))
                    .header("Content-Type", "application/json")
                    .json(&request_body);
            } else {
                // Other providers might use headers
                request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
            }
        }
        
        let response = request_builder
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        
        if response.status().is_success() {
            let json_response: Value = response.json().await?;
            
            if let Some(error) = json_response.get("error") {
                return Err(anyhow::anyhow!("RPC Error: {}", error));
            }
            
            if let Some(result) = json_response.get("result") {
                Ok(result.clone())
            } else {
                Err(anyhow::anyhow!("No result in RPC response"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP Error: {}", response.status()))
        }
    }
    
    async fn make_request_with_fallback(&self, method: &str, params: Value, priority: RequestPriority) -> Result<Value> {
        let providers = self.providers.read().await;
        let mut last_error = None;
        
        // Try all providers in order of priority
        for provider in providers.iter() {
            if matches!(provider.status, ProviderStatus::Failed | ProviderStatus::MonthlyLimitReached) {
                continue;
            }
            
            if !self.check_rate_limits(&provider.name).await.unwrap_or(false) {
                continue;
            }
            
            match self.execute_request(provider, method, params.clone()).await {
                Ok(result) => {
                    self.update_provider_success(&provider.name, Duration::from_millis(100)).await.ok();
                    self.cache_result(method, &params, &result).await.ok();
                    self.track_usage(&provider.name).await.ok();
                    return Ok(result);
                },
                Err(e) => {
                    self.update_provider_error(&provider.name, &e.to_string()).await.ok();
                    last_error = Some(e);
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All RPC providers failed")))
    }
    
    async fn update_provider_success(&self, provider_name: &str, response_time: Duration) -> Result<()> {
        let mut providers = self.providers.write().await;
        
        if let Some(provider) = providers.iter_mut().find(|p| p.name == provider_name) {
            provider.success_count += 1;
            provider.error_count = provider.error_count.saturating_sub(1); // Reduce error count on success
            
            // Update average response time
            let response_time_ms = response_time.as_millis() as f64;
            if provider.average_response_time == 0.0 {
                provider.average_response_time = response_time_ms;
            } else {
                provider.average_response_time = (provider.average_response_time * 0.9) + (response_time_ms * 0.1);
            }
            
            // Update status based on performance
            if provider.error_count == 0 {
                provider.status = ProviderStatus::Active;
            } else if provider.error_count < 5 {
                provider.status = ProviderStatus::Degraded;
            }
            
            provider.last_error = None;
        }
        
        Ok(())
    }
    
    async fn update_provider_error(&self, provider_name: &str, error: &str) -> Result<()> {
        let mut providers = self.providers.write().await;
        
        if let Some(provider) = providers.iter_mut().find(|p| p.name == provider_name) {
            provider.error_count += 1;
            provider.last_error = Some(error.to_string());
            
            // Update status based on error count
            if provider.error_count >= 10 {
                provider.status = ProviderStatus::Failed;
            } else if provider.error_count >= 5 {
                provider.status = ProviderStatus::Degraded;
            }
            
            // Check for rate limiting errors
            if error.contains("rate limit") || error.contains("429") {
                provider.status = ProviderStatus::RateLimited;
            }
        }
        
        Ok(())
    }
    
    async fn track_usage(&self, provider_name: &str) -> Result<()> {
        let mut usage_tracker = self.usage_tracker.lock().await;
        
        // Update daily usage
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        *usage_tracker.daily_usage.entry(format!("{}:{}", provider_name, today)).or_insert(0) += 1;
        
        // Update monthly usage
        let this_month = chrono::Utc::now().format("%Y-%m").to_string();
        *usage_tracker.monthly_usage.entry(format!("{}:{}", provider_name, this_month)).or_insert(0) += 1;
        
        // Update provider monthly usage
        let mut providers = self.providers.write().await;
        if let Some(provider) = providers.iter_mut().find(|p| p.name == provider_name) {
            provider.monthly_usage += 1;
            
            // Check if monthly limit is reached
            if provider.monthly_usage >= provider.monthly_limit {
                provider.status = ProviderStatus::MonthlyLimitReached;
                warn!("🚨 Monthly limit reached for provider: {}", provider_name);
            }
        }
        
        Ok(())
    }
    
    async fn test_all_providers(&self) -> Result<()> {
        info!("🧪 Testing all RPC providers");
        
        let test_request = json!([]);
        
        for provider in self.providers.read().await.iter() {
            match self.execute_request(provider, "getHealth", test_request.clone()).await {
                Ok(_) => {
                    info!("✅ Provider {} is healthy", provider.name);
                },
                Err(e) => {
                    warn!("❌ Provider {} failed health check: {}", provider.name, e);
                    self.update_provider_error(&provider.name, &e.to_string()).await.ok();
                }
            }
        }
        
        Ok(())
    }
    
    async fn start_cache_cleanup_task(&self) {
        let cache = self.cache.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
            
            loop {
                interval.tick().await;
                
                let mut cache_guard = cache.write().await;
                let now = Instant::now();
                
                // Clean expired account cache
                cache_guard.account_cache.retain(|_, data| now.duration_since(data.cached_at) < data.ttl);
                
                // Clean expired transaction cache
                cache_guard.transaction_cache.retain(|_, data| now.duration_since(data.cached_at) < data.ttl);
                
                // Clean expired block cache
                cache_guard.block_cache.retain(|_, data| now.duration_since(data.cached_at) < data.ttl);
                
                // Clean expired token cache
                cache_guard.token_cache.retain(|_, data| now.duration_since(data.cached_at) < data.ttl);
                
                // Clean expired program cache
                cache_guard.program_cache.retain(|_, data| now.duration_since(data.cached_at) < data.ttl);
                
                // Clean expired price cache
                cache_guard.price_cache.retain(|_, data| now.duration_since(data.cached_at) < data.ttl);
                
                debug!("🧹 Cache cleanup completed");
            }
        });
    }
    
    async fn start_rate_limit_reset_task(&self) {
        let rate_limiter = self.rate_limiter.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                // Rate limit resets are handled in check_rate_limits
                // This task could be used for additional rate limit management
            }
        });
    }
    
    async fn start_provider_health_check(&self) {
        let providers = self.providers.clone();
        let rpc_manager = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Every minute
            
            loop {
                interval.tick().await;
                
                if let Err(e) = rpc_manager.test_all_providers().await {
                    error!("Provider health check failed: {}", e);
                }
            }
        });
    }
    
    pub async fn get_usage_report(&self) -> UsageReport {
        let providers = self.providers.read().await;
        let cache = self.cache.read().await;
        let usage_tracker = self.usage_tracker.lock().await;
        
        let mut provider_usage = Vec::new();
        for provider in providers.iter() {
            provider_usage.push(ProviderUsageReport {
                name: provider.name.clone(),
                monthly_usage: provider.monthly_usage,
                monthly_limit: provider.monthly_limit,
                usage_percentage: (provider.monthly_usage as f64 / provider.monthly_limit as f64) * 100.0,
                status: provider.status.clone(),
                average_response_time: provider.average_response_time,
                success_rate: if provider.success_count + provider.error_count > 0 {
                    (provider.success_count as f64 / (provider.success_count + provider.error_count) as f64) * 100.0
                } else {
                    0.0
                },
            });
        }
        
        UsageReport {
            cache_stats: cache.cache_stats.clone(),
            provider_usage,
            total_requests_saved: cache.cache_stats.total_saved_requests,
            estimated_cost_saved: cache.cache_stats.total_saved_requests as f64 * 0.0001, // Estimate
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct UsageReport {
    pub cache_stats: CacheStats,
    pub provider_usage: Vec<ProviderUsageReport>,
    pub total_requests_saved: u64,
    pub estimated_cost_saved: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ProviderUsageReport {
    pub name: String,
    pub monthly_usage: u64,
    pub monthly_limit: u64,
    pub usage_percentage: f64,
    pub status: ProviderStatus,
    pub average_response_time: f64,
    pub success_rate: f64,
}

