Here we will find scripts and tracking of model modifications.

**Use MLFLOW in your code:**
   
   **Option 1: Using environment file (Recommended)**
   ```python
   # 1. Save mlflow.env file from email into the directory
   
   from dotenv import load_dotenv
   import os
   
   load_dotenv('mlflow.env')  # Loads credentials automatically
   
   import mlflow
   mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
   
   # MLFlow automatically uses MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
   # from environment variables
   ```

   **For Nav production deployment**:
   - Deploy MLFlow to Nav's internal infrastructure
   - Update `MLFLOW_TRACKING_URI` environment variable to point to Nav's server
   - All code remains the same - only the connection string changes