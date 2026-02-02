FROM python:3.11-slim

LABEL maintainer="OncoPrep Contributors"
LABEL description="OncoPrep DICOM to BIDS Conversion Container"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    dcm2niix \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy OncoPrep
COPY . /app/

# Install OncoPrep
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /data/dicom /data/bids /data/work

# Set entrypoint
ENTRYPOINT ["python", "-m", "oncoprep.dicom_cli"]

# Default help message
CMD ["--help"]
