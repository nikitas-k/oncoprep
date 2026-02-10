# =============================================================================
# OncoPrep: Neuro-Oncology MRI Preprocessing Pipeline
# Multi-stage Dockerfile for the full pipeline (preprocessing, segmentation,
# fusion, surface processing, and DICOM conversion).
# =============================================================================
# Build args — pin versions for reproducibility
ARG PYTHON_VERSION=3.11
ARG ANTS_VERSION=2.5.3
ARG FSL_VERSION=6.0.7.16
ARG WORKBENCH_VERSION=2.0.1

# ---------------------------------------------------------------------------
# Stage 1 — System dependencies & neuroimaging tools
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

LABEL maintainer="OncoPrep Contributors"
LABEL org.opencontainers.image.title="OncoPrep"
LABEL org.opencontainers.image.description="Neuro-oncology MRI preprocessing pipeline"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/neuronets/oncoprep"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Core build & runtime libraries (includes WeasyPrint deps: pango, cairo, gdk-pixbuf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    dc \
    file \
    git \
    gnupg \
    libfontconfig1 \
    libfreetype6 \
    libgdk-pixbuf-2.0-0 \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    liblapack3 \
    libopenblas0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libpng16-16 \
    libsm6 \
    libxext6 \
    libxml2 \
    libxrender1 \
    libxslt1.1 \
    libzstd1 \
    unzip \
    wget \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# ---- dcm2niix ----
RUN apt-get update && apt-get install -y --no-install-recommends dcm2niix \
    && rm -rf /var/lib/apt/lists/*

# ---- ANTs (pre-built binaries) ----
ARG ANTS_VERSION
ENV ANTSPATH="/opt/ants/bin"
ENV PATH="${ANTSPATH}:${PATH}"
RUN mkdir -p /opt/ants/bin && \
    curl -fsSL https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-ubuntu-22.04-X64-gcc.zip \
        -o /tmp/ants.zip && \
    unzip -q /tmp/ants.zip -d /tmp/ants && \
    ANTS_BIN=$(find /tmp/ants -type f -name "antsRegistration" -print -quit | xargs dirname) && \
    cp -a "${ANTS_BIN}"/* /opt/ants/bin/ && \
    ANTS_SCRIPTS=$(find /tmp/ants -type f -name "antsRegistrationSyN.sh" -print -quit | xargs dirname) && \
    if [ -n "${ANTS_SCRIPTS}" ] && [ "${ANTS_SCRIPTS}" != "${ANTS_BIN}" ]; then \
        cp -a "${ANTS_SCRIPTS}"/* /opt/ants/bin/ ; \
    fi && \
    rm -rf /tmp/ants /tmp/ants.zip

# ---- FSL (minimal — FAST only) ----
# Install FSL via NeuroDebian-style minimal approach
ARG FSL_VERSION
ENV FSLDIR="/opt/fsl" \
    FSLOUTPUTTYPE="NIFTI_GZ"
ENV PATH="${FSLDIR}/bin:${PATH}"
RUN mkdir -p ${FSLDIR}/bin && \
    ( curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py \
        -o /tmp/fslinstaller.py && \
      python /tmp/fslinstaller.py -d ${FSLDIR} --miniconda /opt/fsl/miniconda -V ${FSL_VERSION} --skip_registration \
    ) || echo "FSL full install skipped — falling back to minimal" && \
    rm -f /tmp/fslinstaller.py

# ---- Docker CLI (for tumor segmentation containers) ----
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# ---- Connectome Workbench (optional surface processing) ----
ARG WORKBENCH_VERSION
ENV PATH="/opt/workbench/bin_linux64:${PATH}"
RUN ( mkdir -p /opt/workbench && \
      curl -fsSL https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v${WORKBENCH_VERSION}.zip \
          -o /tmp/wb.zip && \
      unzip -q /tmp/wb.zip -d /opt && \
      rm -f /tmp/wb.zip \
    ) || echo "Workbench install skipped — surface processing will not be available"

# ---------------------------------------------------------------------------
# Stage 3 — Python environment & OncoPrep
# ---------------------------------------------------------------------------
WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY pyproject.toml README.md LICENSE ./
COPY setup.py* ./
COPY src/ ./src/

# Upgrade pip first (base image ships old pip that may fail to find wheels)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install OncoPrep and all core dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Pre-fetch ALL TemplateFlow templates used by OncoPrep
# (ensures offline operation on HPC / air-gapped systems)
ENV TEMPLATEFLOW_HOME="/opt/templateflow"
RUN python -c "\
import templateflow.api as tflow; \
tflow.get('MNI152NLin2009cAsym', resolution=1, desc=None, suffix='T1w'); \
tflow.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='mask'); \
tflow.get('MNI152NLin2009cAsym', resolution=1, desc=None, suffix='T2w'); \
tflow.get('MNI152NLin2009cAsym', resolution=2, desc=None, suffix='T1w'); \
tflow.get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'); \
tflow.get('MNI152NLin2009cAsym', resolution=2, desc=None, suffix='T2w'); \
tflow.get('OASIS30ANTs', resolution=1, suffix='T1w'); \
tflow.get('OASIS30ANTs', resolution=1, desc='brain', suffix='mask'); \
tflow.get('OASIS30ANTs', resolution=1, label='brain', suffix='mask'); \
tflow.get('OASIS30ANTs', resolution=1, desc='BrainCerebellumRegistration', suffix='mask'); \
tflow.get('OASIS30ANTs', resolution=1, desc='4', suffix='dseg'); \
[tflow.get('fsLR', density=d, hemi=h, suffix=s) for d in ('32k','59k') for h in ('L','R') for s in ('midthickness','sphere')]; \
[tflow.get('fsLR', density=d, hemi=h, desc='nomedialwall', suffix='dparc') for d in ('32k','59k') for h in ('L','R')]; \
[tflow.get('fsaverage', density='164k', hemi=h, suffix=s) for h in ('L','R') for s in ('sphere','sulc')]; \
[tflow.get('fsaverage', density='10k', hemi=h, suffix='sphere') for h in ('L','R')]; \
print('TemplateFlow pre-fetch complete:', tflow.TF_LAYOUT.root); \
"

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Neuroimaging environment defaults
ENV OMP_NUM_THREADS=1 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
    IS_DOCKER_8395080871=1 \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

# Create standard data directories
RUN mkdir -p /data/bids /data/output /data/work /data/dicom

# Healthcheck — verify core tools are reachable
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import oncoprep; print(oncoprep.__version__)" && \
        which dcm2niix && which N4BiasFieldCorrection

# Default entrypoint is the main pipeline; override for DICOM conversion
ENTRYPOINT ["oncoprep"]
CMD ["--help"]
