FROM vishwa123/cuda-13.0-pytorch:latest

# ── System deps ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# ── Install uv (fast Python package manager) ──
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Set working directory ──
WORKDIR /workspace

# ── Install Python dependencies ──
# Uses uv for fast installs; falls back to pip if needed.
# The --system flag installs into the system Python (no venv).
RUN uv pip install --system \
    peft \
    "trl[vllm]" \
    pandas \
    datasets \
    sympy \
    tqdm 

# ── Copy project ──
COPY . /workspace/Step_Based_Updates

WORKDIR /workspace/Step_Based_Updates

# ── Default entrypoint ──
CMD ["bash"]
