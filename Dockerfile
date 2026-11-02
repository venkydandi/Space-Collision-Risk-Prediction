FROM python:3.11-slim

# Prevent Python from writing .pyc files, enable unbuffered output, and suppress unpickle warning noise
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies (includes pyyaml, plotly, openpyxl, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source files and configuration
COPY app.py .
COPY config/ config/
COPY core/ core/
COPY services/ services/
COPY components/ components/
COPY views/ views/
COPY assets/ assets/
COPY data/ data/
COPY metadata.json .
COPY .streamlit/ .streamlit/

# Explicitly copy both trained model artifacts and their metadata
# This is declared separately to guarantee model files are always present in the image
# regardless of .dockerignore changes.
COPY models/metadata.json models/metadata.json
COPY models/rf_unweighted.joblib models/rf_unweighted.joblib
COPY models/rf_smotenc_025.joblib models/rf_smotenc_025.joblib

# Verify both model artifacts are present at build time (fails build if missing)
RUN test -f models/rf_unweighted.joblib || (echo "ERROR: rf_unweighted.joblib missing from image" && exit 1)
RUN test -f models/rf_smotenc_025.joblib || (echo "ERROR: rf_smotenc_025.joblib missing from image" && exit 1)

# Expose default Streamlit port
EXPOSE 8501

# Healthcheck endpoint provided by Streamlit runtime
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start Streamlit with dynamic Render PORT expansion (shell form required for ${PORT:-8501})
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
