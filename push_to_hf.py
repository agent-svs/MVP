import os
from huggingface_hub import HfApi, create_repo

# ==============================================================================
# ⚠️  CONFIGURATION REQUIRED
# ==============================================================================
# Replace 'your-org-name' with the EXACT name of the organization you just created
ORG_NAME = "agents-svs"

# Choose a name for your model repository
MODEL_NAME = "voice-receptionist-joint-nlu"

# Local path to where the model is saved
LOCAL_MODEL_DIR = "models/joint_nlu/"
# ==============================================================================

def push_model():
    repo_id = f"{ORG_NAME}/{MODEL_NAME}"
    api = HfApi()

    print(f"🚀 Preparing to push to Hugging Face Hub: {repo_id}")
    
    # 1. Check if the repo exists. If not, create it.
    try:
        api.repo_info(repo_id=repo_id)
        print(f"✅ Repository {repo_id} already exists.")
    except Exception:
        print(f"✨ Creating new repository: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", private=False) # Change to True if you want it private
        
    # 2. Upload the local folder
    print(f"⬆️  Uploading files from '{LOCAL_MODEL_DIR}'...")
    try:
        api.upload_folder(
            folder_path=LOCAL_MODEL_DIR,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial push of fine-tuned Joint NLU (DeBERTa-v3) model",
            ignore_patterns=["checkpoint-*", "runs/*"] # Ignore intermediate 2GB training checkpoints!
        )
        print(f"🎉 Success! Your model is now live at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        print("\nMake sure you are logged in. Run 'huggingface-cli login' in your terminal and paste your Write-access token.")

if __name__ == "__main__":
    if ORG_NAME == "your-org-name":
        print("🛑 Please open push_to_hf.py and change ORG_NAME to your actual Hugging Face organization name!")
    else:
        push_model()
