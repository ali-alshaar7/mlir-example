# Base image
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    clang \
    lld \
    libglib2.0-dev \
    wget \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install initial Python packages
RUN pip install --upgrade pip
RUN pip install numpy wheel

# Clone repositories
WORKDIR /workspace
RUN git clone https://github.com/llvm/torch-mlir.git --branch=main
RUN git clone https://github.com/llvm/llvm-project.git /workspace/torch-mlir/externals/llvm-project && \
    cd /workspace/torch-mlir/externals/llvm-project



WORKDIR /workspace/torch-mlir
# Install Python requirements
RUN python -m pip install -r requirements.txt -r torchvision-requirements.txt

# Build LLVM + MLIR + Torch-MLIR together
RUN cmake -GNinja -Sexternals/llvm-project/llvm -Bbuild \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD"
RUN cmake --build build

RUN . /opt/venv/bin/activate && \
    export PYTHONPATH="$(pwd)/build/tools/torch-mlir/python_packages/torch_mlir:$(pwd)/test/python/fx_importer"


# Default to bash
CMD ["/bin/bash"]
