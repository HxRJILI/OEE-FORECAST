#!/usr/bin/env python3
"""
Setup script for OEE Advisory System
Automates installation and initialization of the RAG-based advisory system
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import requests
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvisorySystemSetup:
    """Setup and configuration manager for the OEE Advisory System"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.required_files = [
            "document_processor.py",
            "rag_system.py", 
            "advisory_integration.py",
            "requirements_rag.txt"
        ]
        self.optional_files = [
            "The Complete_Guide_to_Simple_OEE.pdf"
        ]
        
        # Dependencies to install
        self.core_dependencies = [
            "google-generativeai>=0.3.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "PyPDF2>=3.0.0",
            "pdfplumber>=0.9.0",
            "spacy>=3.7.0",
            "nltk>=3.8.0",
            "scikit-learn>=1.3.0"
        ]
        
        self.spacy_models = ["en_core_web_sm"]
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_required_files(self) -> Dict[str, bool]:
        """Check if required files exist"""
        results = {}
        
        logger.info("📁 Checking required files...")
        for file_name in self.required_files:
            file_path = self.project_dir / file_name
            exists = file_path.exists()
            results[file_name] = exists
            
            if exists:
                logger.info(f"✅ Found: {file_name}")
            else:
                logger.warning(f"❌ Missing: {file_name}")
        
        # Check optional files
        logger.info("📂 Checking optional files...")
        for file_name in self.optional_files:
            file_path = self.project_dir / file_name
            exists = file_path.exists()
            results[file_name] = exists
            
            if exists:
                logger.info(f"✅ Found: {file_name}")
            else:
                logger.info(f"ℹ️  Optional file not found: {file_name}")
        
        return results
    
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Check if requirements file exists
            requirements_file = self.project_dir / "requirements_rag.txt"
            
            if requirements_file.exists():
                logger.info("Installing from requirements_rag.txt...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
            else:
                logger.info("Installing core dependencies individually...")
                for package in self.core_dependencies:
                    logger.info(f"Installing {package}...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def download_spacy_models(self) -> bool:
        """Download required spaCy models"""
        logger.info("🧠 Downloading spaCy language models...")
        
        try:
            for model in self.spacy_models:
                logger.info(f"Downloading {model}...")
                subprocess.run([
                    sys.executable, "-m", "spacy", "download", model
                ], check=True)
            
            logger.info("✅ spaCy models downloaded successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to download spaCy models: {e}")
            return False
    
    def download_nltk_data(self) -> bool:
        """Download required NLTK data"""
        logger.info("📚 Downloading NLTK data...")
        
        try:
            import nltk
            
            # Download required NLTK data
            nltk_downloads = [
                'punkt', 'stopwords', 'averaged_perceptron_tagger',
                'maxent_ne_chunker', 'words', 'wordnet'
            ]
            
            for data_name in nltk_downloads:
                logger.info(f"Downloading NLTK {data_name}...")
                nltk.download(data_name, quiet=True)
            
            logger.info("✅ NLTK data downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download NLTK data: {e}")
            return False
    
    def test_imports(self) -> Dict[str, bool]:
        """Test if all required modules can be imported"""
        logger.info("🔍 Testing imports...")
        
        test_modules = [
            ("google.generativeai", "Gemini API"),
            ("sentence_transformers", "Sentence Transformers"),
            ("faiss", "FAISS"),
            ("PyPDF2", "PyPDF2"),
            ("pdfplumber", "PDFPlumber"),
            ("spacy", "spaCy"),
            ("nltk", "NLTK"),
            ("sklearn", "Scikit-learn")
        ]
        
        results = {}
        
        for module_name, display_name in test_modules:
            try:
                __import__(module_name)
                logger.info(f"✅ {display_name}: OK")
                results[module_name] = True
            except ImportError as e:
                logger.error(f"❌ {display_name}: Failed - {e}")
                results[module_name] = False
        
        return results
    
    def test_advisory_system(self) -> bool:
        """Test if the advisory system can be initialized"""
        logger.info("🤖 Testing advisory system initialization...")
        
        try:
            # Test document processor
            from document_processor import create_document_processor
            processor = create_document_processor()
            logger.info("✅ Document processor: OK")
            
            # Test RAG system (with dummy API key for structure test)
            from rag_system import create_oee_advisor
            # Don't actually initialize with API key in test
            logger.info("✅ RAG system structure: OK")
            
            # Test Streamlit integration
            from advisory_integration import check_advisory_system_status
            status = check_advisory_system_status()
            logger.info(f"✅ Advisory integration: {status}")
            
            logger.info("✅ Advisory system test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Advisory system test failed: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        logger.info("📁 Creating directories...")
        
        directories = [
            "processed_documents",
            "embeddings", 
            "document_metadata",
            "vector_db",
            "models"
        ]
        
        try:
            for dir_name in directories:
                dir_path = self.project_dir / dir_name
                dir_path.mkdir(exist_ok=True)
                logger.info(f"✅ Created/verified: {dir_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def setup_gitignore(self) -> bool:
        """Add advisory system files to .gitignore"""
        logger.info("📝 Updating .gitignore...")
        
        gitignore_entries = [
            "# OEE Advisory System",
            "processed_documents/",
            "embeddings/",
            "document_metadata/", 
            "vector_db/",
            "models/",
            "*.pkl",
            "*.faiss",
            "__pycache__/",
            "*.log"
        ]
        
        try:
            gitignore_path = self.project_dir / ".gitignore"
            
            # Read existing content
            existing_content = ""
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    existing_content = f.read()
            
            # Add new entries if not present
            new_entries = []
            for entry in gitignore_entries:
                if entry not in existing_content:
                    new_entries.append(entry)
            
            if new_entries:
                with open(gitignore_path, 'a') as f:
                    f.write("\n\n" + "\n".join(new_entries))
                logger.info("✅ Updated .gitignore")
            else:
                logger.info("ℹ️  .gitignore already up to date")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update .gitignore: {e}")
            return False
    
    def run_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("🚀 Starting OEE Advisory System setup...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check required files
        file_status = self.check_required_files()
        missing_required = [f for f, exists in file_status.items() 
                          if f in self.required_files and not exists]
        
        if missing_required:
            logger.error(f"❌ Missing required files: {missing_required}")
            logger.error("Please ensure all required files are in the project directory")
            return False
        
        # Create directories
        if not self.create_directories():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Download models
        if not self.download_spacy_models():
            logger.warning("⚠️  spaCy model download failed, but continuing...")
        
        if not self.download_nltk_data():
            logger.warning("⚠️  NLTK data download failed, but continuing...")
        
        # Test imports
        import_results = self.test_imports()
        failed_imports = [module for module, success in import_results.items() if not success]
        
        if failed_imports:
            logger.warning(f"⚠️  Some imports failed: {failed_imports}")
        
        # Test advisory system
        if not self.test_advisory_system():
            logger.warning("⚠️  Advisory system test failed, but installation may still work")
        
        # Update .gitignore
        self.setup_gitignore()
        
        logger.info("✅ Setup completed successfully!")
        logger.info("\n" + "="*60)
        logger.info("🎉 OEE Advisory System is ready!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Restart your Streamlit application")
        logger.info("2. Navigate to the '🤖 OEE Advisory' page")
        logger.info("3. Upload PDF documents to build your knowledge base")
        logger.info("4. Start asking questions about OEE and manufacturing!")
        
        if file_status.get("The Complete_Guide_to_Simple_OEE.pdf", False):
            logger.info("\n📖 Default OEE guide found - it will be automatically processed")
        else:
            logger.info("\n📖 Consider adding 'The Complete_Guide_to_Simple_OEE.pdf' for enhanced knowledge base")
        
        return True


def main():
    """Main setup function"""
    print("🏭 OEE Advisory System Setup")
    print("="*40)
    
    setup = AdvisorySystemSetup()
    
    try:
        success = setup.run_setup()
        
        if success:
            print("\n✅ Setup completed successfully!")
            print("You can now run: streamlit run app.py")
        else:
            print("\n❌ Setup failed. Please check the logs above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    