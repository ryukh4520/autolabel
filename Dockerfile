FROM yolov11-seg:latest

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM2 checkpoint (base model for VRAM efficiency)
RUN mkdir -p /workspace/models && \
    cd /workspace/models && \
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Download YOLO-Pose model
RUN cd /workspace/models && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt

# Copy project files
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
