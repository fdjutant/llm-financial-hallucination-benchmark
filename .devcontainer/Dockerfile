# Base: official devcontainer Miniconda image (nice for VS Code) :contentReference[oaicite:4]{index=4}
FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Optional: avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set work dir
WORKDIR /workspace

# Copy environment file
COPY ./environment.yml /tmp/environment.yml

# Create the Conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Make sure the environment is default
SHELL ["bash", "-c"]
RUN echo "conda activate mlenv" >> ~/.bashrc

# Set PATH so the env is active in non-interactive shells
ENV CONDA_DEFAULT_ENV=mlenv
ENV PATH /opt/conda/envs/mlenv/bin:$PATH

# (Optional) Install any OS-level packages you want
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Remove problematic Yarn repo before running apt-get update
RUN rm -f /etc/apt/sources.list.d/yarn.list \
 && apt-get update \
 && apt-get install -y curl gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Default command
CMD [ "bash" ]
