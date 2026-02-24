# ============================================================================
# roshambo2 – GPU-accelerated molecular shape overlay
# Multi-stage build: pixi env + CUDA build → slim runtime image
# ============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIXI_HOME=/root/.pixi

# System essentials (git needed for pip editable installs, curl for pixi)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="${PIXI_HOME}/bin:${PATH}"

# Copy only dependency manifests first (layer cache for deps)
WORKDIR /opt/roshambo2
COPY pixi.toml pixi.lock* ./

# Install pixi environment (downloads conda packages)
RUN pixi install

# Now copy full source and build the C++/CUDA extensions
COPY . .
RUN pixi run build

# Smoke-test: import should work
RUN pixi run test

# ============================================================================
# Runtime stage – drop the build tools, keep the pixi env + built extensions
# ============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIXI_HOME=/root/.pixi
ENV PATH="${PIXI_HOME}/bin:${PATH}"
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pixi binary
COPY --from=builder /root/.pixi /root/.pixi

# Copy the full project with built extensions + pixi env
WORKDIR /opt/roshambo2
COPY --from=builder /opt/roshambo2 /opt/roshambo2

# Default entrypoint: run the screening script via pixi
ENTRYPOINT ["pixi", "run", "python", "example/smiles_to_roshambo2.py"]
