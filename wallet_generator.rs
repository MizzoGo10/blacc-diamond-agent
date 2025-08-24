use std::env;
use std::fs;
use std::path::Path;
use anyhow::Result;
use serde_json::json;
use solana_sdk::signer::keypair::Keypair;
use solana_sdk::signer::Signer;
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, NewAead};
use rand::RngCore;
use bs58;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    // Parse command line arguments
    let mut wallet_count = 5;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--count" | "-c" => {
                if i + 1 < args.len() {
                    wallet_count = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            },
            "--help" | "-h" => {
                print_help();
                return Ok(());
            },
            _ => {}
        }
        i += 1;
    }
    
    println!("🔑 Blacc Diamond Wallet Generator");
    println!("💰 Generating {} encrypted wallets...", wallet_count);
    
    // Create wallets directory
    fs::create_dir_all("wallets")?;
    
    // Generate encryption key
    let mut encryption_key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut encryption_key);
    
    for i in 1..=wallet_count {
        generate_wallet(i, &encryption_key)?;
        println!("✅ Generated wallet {}/{}", i, wallet_count);
    }
    
    // Save encryption key securely
    let key_file = format!("wallets/master_key.enc");
    fs::write(&key_file, &encryption_key)?;
    println!("🔐 Master encryption key saved to {}", key_file);
    
    println!();
    println!("🎉 All wallets generated successfully!");
    println!("📁 Wallets saved in ./wallets/ directory");
    println!("⚠️  Keep your master key safe - it's needed to decrypt wallets!");
    
    Ok(())
}

fn generate_wallet(wallet_id: usize, encryption_key: &[u8; 32]) -> Result<()> {
    // Generate new keypair
    let keypair = Keypair::new();
    let public_key = keypair.pubkey();
    let private_key_bytes = keypair.to_bytes();
    
    // Convert to different formats
    let private_key_base58 = bs58::encode(&private_key_bytes).into_string();
    let private_key_hex = hex::encode(&private_key_bytes);
    let private_key_array = format!("[{}]", private_key_bytes.iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(","));
    
    // Create wallet data
    let wallet_data = json!({
        "wallet_id": wallet_id,
        "public_key": public_key.to_string(),
        "private_key_base58": private_key_base58,
        "private_key_hex": private_key_hex,
        "private_key_array": private_key_array,
        "formats": {
            "phantom": private_key_base58.clone(),
            "solflare": private_key_base58.clone(),
            "sollet": private_key_hex.clone(),
            "raw_bytes": private_key_array
        },
        "created_at": chrono::Utc::now().to_rfc3339(),
        "purpose": match wallet_id {
            1 => "Blacc Diamond Transaction Engine",
            2 => "Transformer System Operations",
            3 => "MEV Hunter Agent",
            4 => "Flash Trader Agent",
            5 => "Consciousness Trader Agent",
            _ => "General Trading Operations"
        }
    });
    
    // Encrypt the wallet data
    let cipher = Aes256Gcm::new(Key::from_slice(encryption_key));
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    
    let wallet_json = serde_json::to_string_pretty(&wallet_data)?;
    let encrypted_data = cipher.encrypt(nonce, wallet_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;
    
    // Create encrypted wallet file
    let encrypted_wallet = json!({
        "wallet_id": wallet_id,
        "public_key": public_key.to_string(),
        "encrypted_data": base64::encode(&encrypted_data),
        "nonce": base64::encode(&nonce_bytes),
        "encryption": "AES-256-GCM",
        "created_at": chrono::Utc::now().to_rfc3339()
    });
    
    // Save encrypted wallet
    let wallet_file = format!("wallets/wallet_{}_secure.json", wallet_id);
    fs::write(&wallet_file, serde_json::to_string_pretty(&encrypted_wallet)?)?;
    
    // Also save unencrypted version for immediate use (development only)
    let dev_wallet_file = format!("wallets/wallet_{}_dev.json", wallet_id);
    fs::write(&dev_wallet_file, serde_json::to_string_pretty(&wallet_data)?)?;
    
    Ok(())
}

fn print_help() {
    println!("Blacc Diamond Wallet Generator");
    println!();
    println!("USAGE:");
    println!("    wallet-generator [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -c, --count <NUMBER>    Number of wallets to generate (default: 5)");
    println!("    -h, --help              Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    wallet-generator --count 10");
    println!("    wallet-generator -c 3");
}

