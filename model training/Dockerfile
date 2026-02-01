FROM ollama/ollama:latest

# Install jq for JSON parsing
USER root
RUN apt-get update && apt-get install -y jq && rm -rf /var/lib/apt/lists/*

# Copy run_models_config.json
COPY run_models_config.json /app/run_models_config.json

# Copy all model directories
COPY Models/ /app/models/

# Create startup script to load models from config
RUN cat > /app/load_models.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to be ready..."
sleep 5

# Parse config and create enabled models
CONFIG_FILE="/app/run_models_config.json"

echo "Loading models from configuration..."
jq -c '.models[] | select(.enabled == true)' "$CONFIG_FILE" | while read -r model; do
    MODEL_NAME=$(echo "$model" | jq -r '.name')
    MODEL_PATH=$(echo "$model" | jq -r '.path')
    MODELFILE=$(echo "$model" | jq -r '.modelfile')
    DESCRIPTION=$(echo "$model" | jq -r '.description')
    
    echo "Creating model: $MODEL_NAME ($DESCRIPTION)"
    cd "/app/models/$MODEL_PATH"
    
    if [ -f "$MODELFILE" ]; then
        ollama create "$MODEL_NAME" -f "$MODELFILE"
        echo "✓ Model $MODEL_NAME created successfully"
    else
        echo "✗ Modelfile not found: /app/models/$MODEL_PATH/$MODELFILE"
    fi
done

echo "All models loaded. Ollama is ready."

# Keep Ollama running in foreground
wait $OLLAMA_PID
EOF

RUN chmod +x /app/load_models.sh

# Switch back to ollama user
USER ollama

# Use the startup script as entrypoint
CMD ["/bin/bash", "/app/load_models.sh"]
