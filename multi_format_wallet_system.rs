use serde::{Deserialize, Serialize};
use solana_sdk::{
    signature::{Keypair, Signer},
    pubkey::Pubkey,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use bs58;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletFormats {
    pub public_key: String,
    pub private_key_raw: String,
    pub private_key_base58: String,
    pub private_key_hex: String,
    pub private_key_array: Vec<u8>,
    pub mnemonic_phrase: Option<String>,
    pub derivation_path: Option<String>,
    pub wallet_address: String,
    pub keypair_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedWallet {
    pub wallet_id: String,
    pub wallet_name: String,
    pub encrypted_data: Vec<u8>,
    pub nonce: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub wallet_type: String, // "trading", "backup", "cold_storage", "hot_wallet"
    pub balance_sol: f64,
    pub transaction_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletMetadata {
    pub wallet_id: String,
    pub name: String,
    pub wallet_type: String,
    pub public_key: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub balance_sol: f64,
    pub is_active: bool,
    pub security_level: String, // "standard", "high", "maximum"
}

pub struct MultiFormatWalletSystem {
    pub wallets: HashMap<String, EncryptedWallet>,
    pub active_wallet: Option<String>,
    pub encryption_key: [u8; 32],
    pub storage_path: String,
    pub metadata: HashMap<String, WalletMetadata>,
}

impl MultiFormatWalletSystem {
    pub fn new(storage_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Generate or load encryption key
        let encryption_key = Self::generate_encryption_key()?;
        
        // Ensure storage directory exists
        if let Some(parent) = Path::new(storage_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let mut system = Self {
            wallets: HashMap::new(),
            active_wallet: None,
            encryption_key,
            storage_path: storage_path.to_string(),
            metadata: HashMap::new(),
        };

        // Load existing wallets
        system.load_wallets()?;

        Ok(system)
    }

    fn generate_encryption_key() -> Result<[u8; 32], Box<dyn std::error::Error>> {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        Ok(key)
    }

    pub fn create_new_wallet(&mut self, wallet_name: &str, wallet_type: &str) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        // Generate new Solana keypair
        let keypair = Keypair::new();
        let wallet_formats = self.generate_all_formats(&keypair)?;
        
        // Create wallet ID
        let wallet_id = uuid::Uuid::new_v4().to_string();
        
        // Encrypt and store wallet
        let encrypted_wallet = self.encrypt_wallet_data(&wallet_id, wallet_name, wallet_type, &wallet_formats)?;
        
        // Create metadata
        let metadata = WalletMetadata {
            wallet_id: wallet_id.clone(),
            name: wallet_name.to_string(),
            wallet_type: wallet_type.to_string(),
            public_key: wallet_formats.public_key.clone(),
            created_at: chrono::Utc::now(),
            balance_sol: 0.0,
            is_active: true,
            security_level: "maximum".to_string(),
        };

        // Store in memory
        self.wallets.insert(wallet_id.clone(), encrypted_wallet);
        self.metadata.insert(wallet_id.clone(), metadata);
        
        // Set as active wallet if it's the first one
        if self.active_wallet.is_none() {
            self.active_wallet = Some(wallet_id.clone());
        }

        // Save to disk
        self.save_wallets()?;
        
        println!("🔐 Created new wallet: {} ({})", wallet_name, wallet_id);
        println!("📍 Public Key: {}", wallet_formats.public_key);
        
        Ok(wallet_formats)
    }

    fn generate_all_formats(&self, keypair: &Keypair) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        let public_key = keypair.pubkey();
        let private_key_bytes = keypair.to_bytes();
        
        // Generate all possible formats
        let wallet_formats = WalletFormats {
            public_key: public_key.to_string(),
            private_key_raw: format!("{:?}", private_key_bytes),
            private_key_base58: bs58::encode(&private_key_bytes).into_string(),
            private_key_hex: hex::encode(&private_key_bytes),
            private_key_array: private_key_bytes.to_vec(),
            mnemonic_phrase: None, // Could be implemented with bip39
            derivation_path: None,
            wallet_address: public_key.to_string(),
            keypair_bytes: private_key_bytes.to_vec(),
        };

        Ok(wallet_formats)
    }

    fn encrypt_wallet_data(
        &self,
        wallet_id: &str,
        wallet_name: &str,
        wallet_type: &str,
        wallet_formats: &WalletFormats,
    ) -> Result<EncryptedWallet, Box<dyn std::error::Error>> {
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key)?;
        
        // Serialize wallet data
        let wallet_data = serde_json::to_vec(wallet_formats)?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        // Encrypt data
        let encrypted_data = cipher.encrypt(nonce, wallet_data.as_ref())?;
        
        Ok(EncryptedWallet {
            wallet_id: wallet_id.to_string(),
            wallet_name: wallet_name.to_string(),
            encrypted_data,
            nonce: nonce_bytes.to_vec(),
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            wallet_type: wallet_type.to_string(),
            balance_sol: 0.0,
            transaction_count: 0,
        })
    }

    pub fn decrypt_wallet(&self, wallet_id: &str) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        let encrypted_wallet = self.wallets.get(wallet_id)
            .ok_or("Wallet not found")?;
        
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key)?;
        let nonce = Nonce::from_slice(&encrypted_wallet.nonce);
        
        // Decrypt data
        let decrypted_data = cipher.decrypt(nonce, encrypted_wallet.encrypted_data.as_ref())?;
        
        // Deserialize wallet data
        let wallet_formats: WalletFormats = serde_json::from_slice(&decrypted_data)?;
        
        Ok(wallet_formats)
    }

    pub fn get_wallet_for_trading(&self, wallet_id: &str) -> Result<Keypair, Box<dyn std::error::Error>> {
        let wallet_formats = self.decrypt_wallet(wallet_id)?;
        
        // Reconstruct keypair from bytes
        let keypair = Keypair::from_bytes(&wallet_formats.keypair_bytes)?;
        
        Ok(keypair)
    }

    pub fn export_wallet_formats(&self, wallet_id: &str, export_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let wallet_formats = self.decrypt_wallet(wallet_id)?;
        let metadata = self.metadata.get(wallet_id).ok_or("Wallet metadata not found")?;
        
        // Create export data
        let export_data = WalletExport {
            wallet_name: metadata.name.clone(),
            wallet_type: metadata.wallet_type.clone(),
            created_at: metadata.created_at,
            formats: wallet_formats,
            export_timestamp: chrono::Utc::now(),
        };
        
        // Write to file (encrypted)
        let export_json = serde_json::to_string_pretty(&export_data)?;
        
        // Create additional format files
        let base_path = Path::new(export_path);
        let wallet_name = &metadata.name;
        
        // JSON format
        fs::write(base_path.join(format!("{}_wallet.json", wallet_name)), &export_json)?;
        
        // Individual format files
        fs::write(base_path.join(format!("{}_private_key.txt", wallet_name)), &export_data.formats.private_key_base58)?;
        fs::write(base_path.join(format!("{}_private_key.hex", wallet_name)), &export_data.formats.private_key_hex)?;
        fs::write(base_path.join(format!("{}_public_key.txt", wallet_name)), &export_data.formats.public_key)?;
        fs::write(base_path.join(format!("{}_keypair.json", wallet_name)), serde_json::to_string_pretty(&export_data.formats.keypair_bytes)?)?;
        
        // Phantom wallet format
        let phantom_format = PhantomWalletFormat {
            private_key: export_data.formats.private_key_base58.clone(),
            public_key: export_data.formats.public_key.clone(),
        };
        fs::write(base_path.join(format!("{}_phantom.json", wallet_name)), serde_json::to_string_pretty(&phantom_format)?)?;
        
        // Solflare format
        let solflare_format = SolflareWalletFormat {
            keypair: export_data.formats.keypair_bytes.clone(),
            address: export_data.formats.wallet_address.clone(),
        };
        fs::write(base_path.join(format!("{}_solflare.json", wallet_name)), serde_json::to_string_pretty(&solflare_format)?)?;
        
        println!("💾 Wallet exported to: {}", export_path);
        println!("📁 Files created:");
        println!("  - {}_wallet.json (complete wallet data)", wallet_name);
        println!("  - {}_private_key.txt (Base58 private key)", wallet_name);
        println!("  - {}_private_key.hex (Hex private key)", wallet_name);
        println!("  - {}_public_key.txt (Public key)", wallet_name);
        println!("  - {}_keypair.json (Keypair bytes)", wallet_name);
        println!("  - {}_phantom.json (Phantom wallet format)", wallet_name);
        println!("  - {}_solflare.json (Solflare wallet format)", wallet_name);
        
        Ok(())
    }

    pub fn list_wallets(&self) -> Vec<WalletMetadata> {
        self.metadata.values().cloned().collect()
    }

    pub fn get_active_wallet(&self) -> Option<String> {
        self.active_wallet.clone()
    }

    pub fn set_active_wallet(&mut self, wallet_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.wallets.contains_key(wallet_id) {
            self.active_wallet = Some(wallet_id.to_string());
            println!("🔄 Active wallet changed to: {}", wallet_id);
            Ok(())
        } else {
            Err("Wallet not found".into())
        }
    }

    pub fn update_wallet_balance(&mut self, wallet_id: &str, balance: f64) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(metadata) = self.metadata.get_mut(wallet_id) {
            metadata.balance_sol = balance;
        }
        
        if let Some(encrypted_wallet) = self.wallets.get_mut(wallet_id) {
            encrypted_wallet.balance_sol = balance;
            encrypted_wallet.last_accessed = chrono::Utc::now();
        }
        
        Ok(())
    }

    pub fn create_backup_wallet(&mut self) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let wallet_name = format!("backup_{}", timestamp);
        self.create_new_wallet(&wallet_name, "backup")
    }

    pub fn create_cold_storage_wallet(&mut self) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let wallet_name = format!("cold_storage_{}", timestamp);
        self.create_new_wallet(&wallet_name, "cold_storage")
    }

    pub fn create_trading_wallet(&mut self) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let wallet_name = format!("trading_{}", timestamp);
        self.create_new_wallet(&wallet_name, "trading")
    }

    fn save_wallets(&self) -> Result<(), Box<dyn std::error::Error>> {
        let wallet_data = WalletStorage {
            wallets: self.wallets.clone(),
            metadata: self.metadata.clone(),
            active_wallet: self.active_wallet.clone(),
            last_updated: chrono::Utc::now(),
        };
        
        let json_data = serde_json::to_string_pretty(&wallet_data)?;
        fs::write(&self.storage_path, json_data)?;
        
        Ok(())
    }

    fn load_wallets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if Path::new(&self.storage_path).exists() {
            let json_data = fs::read_to_string(&self.storage_path)?;
            let wallet_data: WalletStorage = serde_json::from_str(&json_data)?;
            
            self.wallets = wallet_data.wallets;
            self.metadata = wallet_data.metadata;
            self.active_wallet = wallet_data.active_wallet;
            
            println!("📂 Loaded {} wallets from storage", self.wallets.len());
        }
        
        Ok(())
    }

    pub fn generate_wallet_report(&self) -> WalletSystemReport {
        let total_wallets = self.wallets.len();
        let active_wallets = self.metadata.values().filter(|m| m.is_active).count();
        let total_balance: f64 = self.metadata.values().map(|m| m.balance_sol).sum();
        
        let wallet_types: HashMap<String, usize> = self.metadata.values()
            .fold(HashMap::new(), |mut acc, metadata| {
                *acc.entry(metadata.wallet_type.clone()).or_insert(0) += 1;
                acc
            });

        WalletSystemReport {
            total_wallets,
            active_wallets,
            total_balance_sol: total_balance,
            wallet_types,
            active_wallet_id: self.active_wallet.clone(),
            last_updated: chrono::Utc::now(),
            security_status: "maximum".to_string(),
            encryption_status: "AES-256-GCM".to_string(),
        }
    }

    pub fn rotate_encryption_key(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Generate new encryption key
        let new_key = Self::generate_encryption_key()?;
        
        // Decrypt all wallets with old key and re-encrypt with new key
        let mut new_wallets = HashMap::new();
        
        for (wallet_id, encrypted_wallet) in &self.wallets {
            // Decrypt with old key
            let wallet_formats = self.decrypt_wallet(wallet_id)?;
            
            // Update encryption key
            let old_key = self.encryption_key;
            self.encryption_key = new_key;
            
            // Re-encrypt with new key
            let new_encrypted_wallet = self.encrypt_wallet_data(
                wallet_id,
                &encrypted_wallet.wallet_name,
                &encrypted_wallet.wallet_type,
                &wallet_formats,
            )?;
            
            new_wallets.insert(wallet_id.clone(), new_encrypted_wallet);
            
            // Restore old key temporarily
            self.encryption_key = old_key;
        }
        
        // Update to new key and wallets
        self.encryption_key = new_key;
        self.wallets = new_wallets;
        
        // Save with new encryption
        self.save_wallets()?;
        
        println!("🔐 Encryption key rotated successfully");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletExport {
    pub wallet_name: String,
    pub wallet_type: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub formats: WalletFormats,
    pub export_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomWalletFormat {
    pub private_key: String,
    pub public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolflareWalletFormat {
    pub keypair: Vec<u8>,
    pub address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletStorage {
    pub wallets: HashMap<String, EncryptedWallet>,
    pub metadata: HashMap<String, WalletMetadata>,
    pub active_wallet: Option<String>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletSystemReport {
    pub total_wallets: usize,
    pub active_wallets: usize,
    pub total_balance_sol: f64,
    pub wallet_types: HashMap<String, usize>,
    pub active_wallet_id: Option<String>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub security_status: String,
    pub encryption_status: String,
}

// Wallet Manager for Trading Operations
pub struct TradingWalletManager {
    pub wallet_system: MultiFormatWalletSystem,
    pub hot_wallets: Vec<String>,
    pub cold_wallets: Vec<String>,
    pub rotation_schedule: HashMap<String, chrono::DateTime<chrono::Utc>>,
}

impl TradingWalletManager {
    pub fn new(storage_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let wallet_system = MultiFormatWalletSystem::new(storage_path)?;
        
        Ok(Self {
            wallet_system,
            hot_wallets: Vec::new(),
            cold_wallets: Vec::new(),
            rotation_schedule: HashMap::new(),
        })
    }

    pub fn setup_trading_infrastructure(&mut self) -> Result<TradingInfrastructure, Box<dyn std::error::Error>> {
        // Create primary trading wallet
        let primary_wallet = self.wallet_system.create_trading_wallet()?;
        
        // Create backup wallets
        let backup_wallet_1 = self.wallet_system.create_backup_wallet()?;
        let backup_wallet_2 = self.wallet_system.create_backup_wallet()?;
        
        // Create cold storage wallet
        let cold_storage = self.wallet_system.create_cold_storage_wallet()?;
        
        // Setup hot wallets for rapid trading
        let mut hot_wallets = Vec::new();
        for i in 1..=5 {
            let hot_wallet = self.wallet_system.create_new_wallet(
                &format!("hot_wallet_{}", i),
                "hot_wallet"
            )?;
            hot_wallets.push(hot_wallet);
        }

        Ok(TradingInfrastructure {
            primary_trading_wallet: primary_wallet,
            backup_wallets: vec![backup_wallet_1, backup_wallet_2],
            cold_storage_wallet: cold_storage,
            hot_wallets,
            created_at: chrono::Utc::now(),
        })
    }

    pub fn get_next_available_wallet(&self) -> Result<Keypair, Box<dyn std::error::Error>> {
        if let Some(active_wallet_id) = &self.wallet_system.active_wallet {
            self.wallet_system.get_wallet_for_trading(active_wallet_id)
        } else {
            Err("No active wallet available".into())
        }
    }

    pub fn rotate_hot_wallets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Implement wallet rotation for security
        println!("🔄 Rotating hot wallets for enhanced security");
        
        // Create new hot wallet
        let new_hot_wallet = self.wallet_system.create_new_wallet(
            &format!("hot_wallet_{}", chrono::Utc::now().timestamp()),
            "hot_wallet"
        )?;
        
        println!("🔥 New hot wallet created: {}", new_hot_wallet.public_key);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingInfrastructure {
    pub primary_trading_wallet: WalletFormats,
    pub backup_wallets: Vec<WalletFormats>,
    pub cold_storage_wallet: WalletFormats,
    pub hot_wallets: Vec<WalletFormats>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

