"""
🔄 Migration Script - CLIP to BLIP-2 System Migration
Script tự động để migrate từ CLIP sang BLIP-2 và cleanup duplicates
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemMigration:
    """🔄 System Migration Manager"""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.backup_path = self.root_path / "backup_migration"
        self.config_path = self.root_path / "config"
        
        # Files to delete (from analysis)
        self.files_to_delete = [
            "fix_search.py",
            "fix_search_complete.py", 
            "simple_fix.py",
            "debug_search.py",
            "ai_search_lite.py",
            "backend_ai_lite.py"
        ]
        
        # Files to preserve
        self.core_files = [
            "web_interface.py",
            "enhanced_hybrid_manager.py",
            "tensorflow_model_manager.py",
            "ai_search_engine.py",
            "scripts/build_faiss.py"
        ]
        
        # New BLIP-2 files
        self.new_files = [
            "blip2_core_manager.py",
            "ai_search_engine_v2.py", 
            "web_interface_v2.py",
            "config/requirements_blip2.txt"
        ]
    
    def create_backup(self) -> bool:
        """📦 Create backup of current system"""
        try:
            logger.info("📦 Creating system backup...")
            
            if self.backup_path.exists():
                logger.warning("⚠️  Backup already exists, removing old backup...")
                shutil.rmtree(self.backup_path)
            
            self.backup_path.mkdir(exist_ok=True)
            
            # Backup important files
            backup_files = self.files_to_delete + self.core_files + [
                "web_interface.py",
                "ai_search_engine.py",
                "requirements.txt"
            ]
            
            backed_up = 0
            for file_path in backup_files:
                src = self.root_path / file_path
                if src.exists():
                    dst = self.backup_path / file_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    if src.is_file():
                        shutil.copy2(src, dst)
                        backed_up += 1
                    elif src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                        backed_up += 1
            
            # Backup current requirements
            for req_file in ["requirements.txt", "config/requirements.txt"]:
                src = self.root_path / req_file
                if src.exists():
                    dst = self.backup_path / req_file
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            
            logger.info(f"✅ Backup created: {backed_up} items backed up to {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def cleanup_duplicates(self) -> bool:
        """🧹 Remove duplicate and obsolete files"""
        try:
            logger.info("🧹 Cleaning up duplicate files...")
            
            removed = 0
            for file_path in self.files_to_delete:
                target = self.root_path / file_path
                if target.exists():
                    if target.is_file():
                        target.unlink()
                        logger.info(f"🗑️  Removed: {file_path}")
                        removed += 1
                    elif target.is_dir():
                        shutil.rmtree(target)
                        logger.info(f"🗑️  Removed directory: {file_path}")
                        removed += 1
                else:
                    logger.info(f"⏭️  Already removed: {file_path}")
            
            logger.info(f"✅ Cleanup completed: {removed} items removed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    def install_blip2_requirements(self) -> bool:
        """📦 Install BLIP-2 requirements"""
        try:
            logger.info("📦 Installing BLIP-2 requirements...")
            
            # Check if virtual environment is active
            venv_path = self.root_path / ".venv"
            if venv_path.exists():
                if os.name == 'nt':  # Windows
                    python_exe = venv_path / "Scripts" / "python.exe"
                    pip_exe = venv_path / "Scripts" / "pip.exe"
                else:  # Linux/Mac
                    python_exe = venv_path / "bin" / "python"
                    pip_exe = venv_path / "bin" / "pip"
                
                if not python_exe.exists():
                    logger.error("❌ Virtual environment python not found")
                    return False
            else:
                python_exe = "python"
                pip_exe = "pip"
            
            # Install requirements
            req_file = self.config_path / "requirements_blip2.txt"
            if not req_file.exists():
                logger.error(f"❌ Requirements file not found: {req_file}")
                return False
            
            cmd = [str(pip_exe), "install", "-r", str(req_file), "--upgrade"]
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ BLIP-2 requirements installed successfully")
                return True
            else:
                logger.error(f"❌ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Requirements installation failed: {e}")
            return False
    
    def verify_new_system(self) -> bool:
        """✅ Verify new BLIP-2 system"""
        try:
            logger.info("✅ Verifying new BLIP-2 system...")
            
            # Check if new files exist
            missing_files = []
            for file_path in self.new_files:
                target = self.root_path / file_path
                if not target.exists():
                    missing_files.append(file_path)
            
            if missing_files:
                logger.error(f"❌ Missing new files: {missing_files}")
                return False
            
            # Try importing new modules
            try:
                sys.path.insert(0, str(self.root_path))
                
                logger.info("🧠 Testing BLIP-2 core manager import...")
                from blip2_core_manager import BLIP2CoreManager, ComplexQueryProcessor
                
                logger.info("🔍 Testing new search engine import...")
                from ai_search_engine_v2 import EnhancedBLIP2SearchEngine
                
                logger.info("🌐 Testing new web interface import...")
                from web_interface_v2 import app
                
                logger.info("✅ All imports successful")
                return True
                
            except ImportError as e:
                logger.error(f"❌ Import failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def create_migration_summary(self) -> bool:
        """📊 Create migration summary"""
        try:
            summary = {
                "migration_info": {
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "migration_type": "CLIP to BLIP-2",
                    "backup_location": str(self.backup_path),
                    "status": "completed"
                },
                "files_removed": self.files_to_delete,
                "files_preserved": self.core_files,
                "new_files_added": self.new_files,
                "next_steps": [
                    "1. Test new BLIP-2 system: python -m uvicorn web_interface_v2:app --host 0.0.0.0 --port 8080",
                    "2. Build new index: Access /api/index/rebuild endpoint",
                    "3. Test search functionality with complex queries",
                    "4. Monitor performance vs old system",
                    "5. If satisfied, remove backup folder"
                ],
                "rollback_instructions": [
                    "1. Stop current server",
                    "2. Copy files from backup folder back to root",
                    "3. Reinstall original requirements",
                    "4. Start original web_interface.py"
                ]
            }
            
            summary_file = self.root_path / "MIGRATION_SUMMARY.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Migration summary created: {summary_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create summary: {e}")
            return False
    
    def run_migration(self) -> bool:
        """🚀 Run complete migration process"""
        logger.info("🚀 Starting CLIP to BLIP-2 migration...")
        
        steps = [
            ("📦 Creating backup", self.create_backup),
            ("🧹 Cleaning up duplicates", self.cleanup_duplicates),
            ("📦 Installing BLIP-2 requirements", self.install_blip2_requirements),
            ("✅ Verifying new system", self.verify_new_system),
            ("📊 Creating migration summary", self.create_migration_summary)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            if not step_func():
                logger.error(f"❌ Migration failed at: {step_name}")
                logger.info(f"🔄 Restore from backup: {self.backup_path}")
                return False
        
        logger.info("\n🎉 Migration completed successfully!")
        logger.info("📋 Next steps:")
        logger.info("1. Start new server: python -m uvicorn web_interface_v2:app --host 0.0.0.0 --port 8080")
        logger.info("2. Test BLIP-2 search functionality")
        logger.info("3. Build new index if needed")
        logger.info("4. Check MIGRATION_SUMMARY.json for details")
        
        return True


def main():
    """Main migration script"""
    print("🔄 CLIP to BLIP-2 System Migration")
    print("=" * 50)
    
    migrator = SystemMigration()
    
    # Confirm migration
    while True:
        confirm = input("\n⚠️  This will modify your system. Continue? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no', '']:
            print("❌ Migration cancelled")
            return
        else:
            print("Please enter 'y' or 'n'")
    
    # Run migration
    success = migrator.run_migration()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("🚀 You can now start the new BLIP-2 system")
    else:
        print("\n❌ Migration failed")
        print(f"🔄 Restore from backup: {migrator.backup_path}")


if __name__ == "__main__":
    main()
