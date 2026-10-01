# src/config_loader.py
import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config():
    # Tentukan nama file config Anda (ubah jika namanya bukan config.yaml)
    config_file = "config.yaml" 
    
    # Cek apakah file config benar-benar ada di folder proyek
    if not os.path.exists(config_file):
        logger.error(f"Kritis: File konfigurasi '{config_file}' TIDAK DITEMUKAN di root direktori!")
        # Kita kembalikan struktur default agar aplikasi tidak langsung crash total
        return {"paths": {"faiss_index_dir": "data/faiss_index"}}
        
    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            logger.error(f"Kritis: File '{config_file}' ada tetapi ISI FILENYA KOSONG!")
            return {"paths": {"faiss_index_dir": "data/faiss_index"}}
            
        return config_data
        
    except Exception as e:
        logger.error(f"Gagal membaca file konfigurasi: {e}")
        return {"paths": {"faiss_index_dir": "data/faiss_index"}}