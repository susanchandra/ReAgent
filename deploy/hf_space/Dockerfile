FROM python:3.12-slim

# Create non-root user (Hugging Face Spaces convention UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    REAGENT_PUBLIC_MODE=1

WORKDIR $HOME/app

# Install minimal dependencies
COPY --chown=user:user deploy/hf_space/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application packages and verified benchmark reports
COPY --chown=user:user reagent/ reagent/
COPY --chown=user:user reports/ reports/
COPY --chown=user:user deploy/hf_space/entrypoint.py entrypoint.py

# Create writable temp directory for isolated disposable SQLite stores
RUN mkdir -p /tmp/reagent_run

EXPOSE 7860

CMD ["python", "entrypoint.py", "--host", "0.0.0.0", "--port", "7860"]
