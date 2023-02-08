FROM nvcr.io/nvidia/pytorch:22.09-py3


ARG USERNAME=$USERNAME
ARG USER_UID=$USER_UID
ARG USER_GID=$USER_GID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get -y install --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# libaio-dev is required for deepspeed it seems
RUN apt-get install libaio-dev -y

USER $USERNAME

RUN pip3 install --upgrade pip

# Remove existing torch and torchvision from the base image
# RUN pip3 uninstall -y torch torchvision

# Install nightly torch and torchvision
# RUN pip install --force-reinstall --pre "torch==2.*" torchvision  --index-url https://download.pytorch.org/whl/nightly/cu117

COPY ../ ./src

RUN pip install -r ./src/requirements.txt