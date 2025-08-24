use serde::{Deserialize, Serialize};
use solana_sdk::{
    signature::{Keypair, Signer},
    pubkey::Pubkey,
};
use std::collections::HashMap;
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, NewAead};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletFormats {
    pub wallet_id: String,
    pub base58_private_key: String,
    pub base58_public_key: String,
    pub hex_private_key: String,
    pub hex_public_key: String,
    pub ed25519_binary_private: Vec<u8>, // Raw Ed25519 binary format
    pub ed25519_binary_public: Vec<u8>,
    pub phantom_format: String,
    pub solflare_format: String,
    pub backpack_format: String,
    pub glow_format: String,
    pub array_format_private: Vec<u8>,
    pub array_format_public: Vec<u8>,
    pub keypair_bytes: Vec<u8>, // Full 64-byte keypair
    pub seed_phrase: Option<String>,
    pub derivation_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedWalletStorage {
    pub wallet_id: String,
    pub encrypted_data: Vec<u8>,
    pub nonce: Vec<u8>,
    pub salt: Vec<u8>,
    pub encryption_method: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub access_count: u64,
}

pub struct AdvancedWalletManager {
    pub wallets: HashMap<String, WalletFormats>,
    pub encrypted_storage: HashMap<String, EncryptedWalletStorage>,
    pub master_key: Vec<u8>,
}

impl AdvancedWalletManager {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Generate master encryption key
        let master_key = Self::generate_master_key()?;
        
        Ok(Self {
            wallets: HashMap::new(),
            encrypted_storage: HashMap::new(),
            master_key,
        })
    }

    fn generate_master_key() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use rand::RngCore;
        let mut key = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        Ok(key)
    }

    pub fn generate_wallet_all_formats(&mut self) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        // Generate new Solana keypair
        let keypair = Keypair::new();
        let wallet_id = uuid::Uuid::new_v4().to_string();

        // Extract private and public key bytes
        let private_key_bytes = keypair.secret().to_bytes();
        let public_key_bytes = keypair.pubkey().to_bytes();
        let full_keypair_bytes = keypair.to_bytes();

        // Create all wallet formats
        let wallet_formats = WalletFormats {
            wallet_id: wallet_id.clone(),
            
            // Base58 formats (standard Solana)
            base58_private_key: bs58::encode(&private_key_bytes).into_string(),
            base58_public_key: keypair.pubkey().to_string(),
            
            // Hex formats
            hex_private_key: hex::encode(&private_key_bytes),
            hex_public_key: hex::encode(&public_key_bytes),
            
            // Ed25519 binary formats (raw bytes)
            ed25519_binary_private: private_key_bytes.to_vec(),
            ed25519_binary_public: public_key_bytes.to_vec(),
            
            // Wallet-specific formats
            phantom_format: self.create_phantom_format(&keypair)?,
            solflare_format: self.create_solflare_format(&keypair)?,
            backpack_format: self.create_backpack_format(&keypair)?,
            glow_format: self.create_glow_format(&keypair)?,
            
            // Array formats
            array_format_private: private_key_bytes.to_vec(),
            array_format_public: public_key_bytes.to_vec(),
            
            // Full keypair bytes (64 bytes total)
            keypair_bytes: full_keypair_bytes.to_vec(),
            
            // Optional seed phrase and derivation
            seed_phrase: None, // Could implement BIP39 if needed
            derivation_path: None,
        };

        // Store wallet
        self.wallets.insert(wallet_id.clone(), wallet_formats.clone());
        
        // Encrypt and store securely
        self.encrypt_and_store_wallet(&wallet_formats)?;

        println!("🔐 Generated wallet with all formats: {}", wallet_id);
        println!("📋 Ed25519 Binary Private Key Length: {} bytes", wallet_formats.ed25519_binary_private.len());
        println!("📋 Full Keypair Length: {} bytes", wallet_formats.keypair_bytes.len());

        Ok(wallet_formats)
    }

    fn create_phantom_format(&self, keypair: &Keypair) -> Result<String, Box<dyn std::error::Error>> {
        // Phantom uses base58 encoded private key with specific prefix
        let private_bytes = keypair.secret().to_bytes();
        Ok(format!("phantom:{}", bs58::encode(&private_bytes).into_string()))
    }

    fn create_solflare_format(&self, keypair: &Keypair) -> Result<String, Box<dyn std::error::Error>> {
        // Solflare uses JSON format with array of bytes
        let private_bytes = keypair.secret().to_bytes();
        let array_string: String = private_bytes.iter()
            .map(|b| b.to_string())
            .collect::<Vec<String>>()
            .join(",");
        Ok(format!("[{}]", array_string))
    }

    fn create_backpack_format(&self, keypair: &Keypair) -> Result<String, Box<dyn std::error::Error>> {
        // Backpack uses hex format with 0x prefix
        let private_bytes = keypair.secret().to_bytes();
        Ok(format!("0x{}", hex::encode(&private_bytes)))
    }

    fn create_glow_format(&self, keypair: &Keypair) -> Result<String, Box<dyn std::error::Error>> {
        // Glow uses base64 encoded private key
        let private_bytes = keypair.secret().to_bytes();
        Ok(base64::encode(&private_bytes))
    }

    fn encrypt_and_store_wallet(&mut self, wallet: &WalletFormats) -> Result<(), Box<dyn std::error::Error>> {
        // Serialize wallet data
        let wallet_data = serde_json::to_vec(wallet)?;
        
        // Generate random nonce and salt
        use rand::RngCore;
        let mut nonce_bytes = vec![0u8; 12];
        let mut salt = vec![0u8; 16];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        rand::thread_rng().fill_bytes(&mut salt);

        // Create cipher
        let key = Key::from_slice(&self.master_key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt data
        let encrypted_data = cipher.encrypt(nonce, wallet_data.as_ref())
            .map_err(|e| format!("Encryption failed: {}", e))?;

        // Store encrypted wallet
        let encrypted_storage = EncryptedWalletStorage {
            wallet_id: wallet.wallet_id.clone(),
            encrypted_data,
            nonce: nonce_bytes,
            salt,
            encryption_method: "AES-256-GCM".to_string(),
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            access_count: 0,
        };

        self.encrypted_storage.insert(wallet.wallet_id.clone(), encrypted_storage);
        Ok(())
    }

    pub fn get_wallet_for_signing(&mut self, wallet_id: &str) -> Result<Keypair, Box<dyn std::error::Error>> {
        // Decrypt and retrieve wallet
        let wallet = self.decrypt_wallet(wallet_id)?;
        
        // Create keypair from Ed25519 binary format
        let keypair = Keypair::from_bytes(&wallet.keypair_bytes)?;
        
        // Update access tracking
        if let Some(encrypted_storage) = self.encrypted_storage.get_mut(wallet_id) {
            encrypted_storage.last_accessed = chrono::Utc::now();
            encrypted_storage.access_count += 1;
        }

        Ok(keypair)
    }

    fn decrypt_wallet(&self, wallet_id: &str) -> Result<WalletFormats, Box<dyn std::error::Error>> {
        let encrypted_storage = self.encrypted_storage.get(wallet_id)
            .ok_or("Wallet not found in encrypted storage")?;

        // Create cipher
        let key = Key::from_slice(&self.master_key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(&encrypted_storage.nonce);

        // Decrypt data
        let decrypted_data = cipher.decrypt(nonce, encrypted_storage.encrypted_data.as_ref())
            .map_err(|e| format!("Decryption failed: {}", e))?;

        // Deserialize wallet
        let wallet: WalletFormats = serde_json::from_slice(&decrypted_data)?;
        Ok(wallet)
    }

    pub fn export_wallet_format(&mut self, wallet_id: &str, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        let wallet = self.decrypt_wallet(wallet_id)?;
        
        match format.to_lowercase().as_str() {
            "base58" | "base58_private" => Ok(wallet.base58_private_key),
            "base58_public" => Ok(wallet.base58_public_key),
            "hex" | "hex_private" => Ok(wallet.hex_private_key),
            "hex_public" => Ok(wallet.hex_public_key),
            "phantom" => Ok(wallet.phantom_format),
            "solflare" => Ok(wallet.solflare_format),
            "backpack" => Ok(wallet.backpack_format),
            "glow" => Ok(wallet.glow_format),
            "ed25519_binary" => Ok(hex::encode(&wallet.ed25519_binary_private)),
            "array" => Ok(format!("{:?}", wallet.array_format_private)),
            "keypair_bytes" => Ok(hex::encode(&wallet.keypair_bytes)),
            _ => Err(format!("Unknown wallet format: {}", format).into()),
        }
    }

    pub fn import_wallet_from_format(&mut self, private_key: &str, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        let keypair = match format.to_lowercase().as_str() {
            "base58" => {
                let bytes = bs58::decode(private_key).into_vec()?;
                Keypair::from_bytes(&bytes)?
            },
            "hex" => {
                let bytes = hex::decode(private_key)?;
                Keypair::from_bytes(&bytes)?
            },
            "array" => {
                // Parse array format like "[1,2,3,...]"
                let cleaned = private_key.trim_start_matches('[').trim_end_matches(']');
                let bytes: Result<Vec<u8>, _> = cleaned.split(',')
                    .map(|s| s.trim().parse::<u8>())
                    .collect();
                Keypair::from_bytes(&bytes?)?
            },
            "ed25519_binary" => {
                let bytes = hex::decode(private_key)?;
                Keypair::from_bytes(&bytes)?
            },
            _ => return Err(format!("Unsupported import format: {}", format).into()),
        };

        // Generate wallet ID and create all formats
        let wallet_id = uuid::Uuid::new_v4().to_string();
        let private_key_bytes = keypair.secret().to_bytes();
        let public_key_bytes = keypair.pubkey().to_bytes();
        let full_keypair_bytes = keypair.to_bytes();

        let wallet_formats = WalletFormats {
            wallet_id: wallet_id.clone(),
            base58_private_key: bs58::encode(&private_key_bytes).into_string(),
            base58_public_key: keypair.pubkey().to_string(),
            hex_private_key: hex::encode(&private_key_bytes),
            hex_public_key: hex::encode(&public_key_bytes),
            ed25519_binary_private: private_key_bytes.to_vec(),
            ed25519_binary_public: public_key_bytes.to_vec(),
            phantom_format: self.create_phantom_format(&keypair)?,
            solflare_format: self.create_solflare_format(&keypair)?,
            backpack_format: self.create_backpack_format(&keypair)?,
            glow_format: self.create_glow_format(&keypair)?,
            array_format_private: private_key_bytes.to_vec(),
            array_format_public: public_key_bytes.to_vec(),
            keypair_bytes: full_keypair_bytes.to_vec(),
            seed_phrase: None,
            derivation_path: None,
        };

        // Store wallet
        self.wallets.insert(wallet_id.clone(), wallet_formats.clone());
        self.encrypt_and_store_wallet(&wallet_formats)?;

        println!("📥 Imported wallet from {} format: {}", format, wallet_id);
        Ok(wallet_id)
    }

    pub fn create_signing_keypair_from_binary(&self, ed25519_binary: &[u8]) -> Result<Keypair, Box<dyn std::error::Error>> {
        // Create keypair directly from Ed25519 binary format for transaction signing
        if ed25519_binary.len() == 32 {
            // If only private key (32 bytes), create full keypair
            let keypair = Keypair::from_bytes(&[ed25519_binary, &[0u8; 32]].concat())?;
            Ok(keypair)
        } else if ed25519_binary.len() == 64 {
            // Full keypair (64 bytes)
            let keypair = Keypair::from_bytes(ed25519_binary)?;
            Ok(keypair)
        } else {
            Err(format!("Invalid Ed25519 binary length: {} bytes", ed25519_binary.len()).into())
        }
    }

    pub fn get_wallet_stats(&self) -> WalletStats {
        WalletStats {
            total_wallets: self.wallets.len(),
            encrypted_wallets: self.encrypted_storage.len(),
            formats_supported: vec![
                "Base58".to_string(),
                "Hex".to_string(),
                "Ed25519 Binary".to_string(),
                "Phantom".to_string(),
                "Solflare".to_string(),
                "Backpack".to_string(),
                "Glow".to_string(),
                "Array".to_string(),
            ],
            encryption_method: "AES-256-GCM".to_string(),
        }
    }

    pub fn list_wallets(&self) -> Vec<WalletSummary> {
        self.wallets.iter().map(|(id, wallet)| {
            WalletSummary {
                wallet_id: id.clone(),
                public_key: wallet.base58_public_key.clone(),
                formats_available: 8,
                created_at: chrono::Utc::now(), // Would be stored in real implementation
            }
        }).collect()
    }

    pub fn backup_wallet_to_file(&mut self, wallet_id: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let wallet = self.decrypt_wallet(wallet_id)?;
        let backup_data = serde_json::to_string_pretty(&wallet)?;
        std::fs::write(file_path, backup_data)?;
        println!("💾 Wallet backed up to: {}", file_path);
        Ok(())
    }

    pub fn restore_wallet_from_file(&mut self, file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
        let backup_data = std::fs::read_to_string(file_path)?;
        let wallet: WalletFormats = serde_json::from_str(&backup_data)?;
        let wallet_id = wallet.wallet_id.clone();
        
        self.wallets.insert(wallet_id.clone(), wallet.clone());
        self.encrypt_and_store_wallet(&wallet)?;
        
        println!("📥 Wallet restored from: {}", file_path);
        Ok(wallet_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletStats {
    pub total_wallets: usize,
    pub encrypted_wallets: usize,
    pub formats_supported: Vec<String>,
    pub encryption_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletSummary {
    pub wallet_id: String,
    pub public_key: String,
    pub formats_available: u8,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

// Transaction signing utilities
impl AdvancedWalletManager {
    pub fn sign_transaction_with_wallet(&mut self, wallet_id: &str, transaction_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let keypair = self.get_wallet_for_signing(wallet_id)?;
        let signature = keypair.sign_message(transaction_data);
        Ok(signature.as_ref().to_vec())
    }

    pub fn sign_transaction_with_binary_key(&self, ed25519_binary: &[u8], transaction_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let keypair = self.create_signing_keypair_from_binary(ed25519_binary)?;
        let signature = keypair.sign_message(transaction_data);
        Ok(signature.as_ref().to_vec())
    }

    pub fn verify_signature(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> bool {
        use ed25519_dalek::{PublicKey, Signature, Verifier};
        
        if let (Ok(public_key), Ok(signature)) = (
            PublicKey::from_bytes(public_key),
            Signature::from_bytes(signature)
        ) {
            public_key.verify(message, &signature).is_ok()
        } else {
            false
        }
    }
}

