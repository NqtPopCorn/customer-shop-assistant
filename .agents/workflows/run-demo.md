# Workflow: /run-demo

Workflow to quickly prepare, start the application, and test health and conversation capability.

## Steps

1. **Synchronize Environment**
   ```powershell
   uv sync
   ```

2. **Verify Configuration**
   Check that `.env` is present and contains required endpoints for the active provider:
   - `LLM_PROVIDER`: `ollama` or `lmstudio`

3. **Start the Development Server**
   Start Uvicorn server:
   ```powershell
   uv run uvicorn customer_shop_assistant.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Verify Health Endpoint**
   Execute an HTTP GET request to verify the server is running:
   ```powershell
   Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
   ```

5. **Test Chat Interaction (When Chat Endpoint Exists)**
   Send a sample inquiry:
   ```powershell
   $body = @{ message = "Xin chào, shop có bán áo khoác không?" } | ConvertTo-Json
   Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/chat -Method Post -ContentType "application/json" -Body $body
   ```
