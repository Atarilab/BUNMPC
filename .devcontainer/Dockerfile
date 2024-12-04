# Base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# remove yes no questions
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# install system related packages
RUN apt-get update && \
    apt-get install -y \
    cmake \
    make \
    nano \
    vim \
    curl \
    git \
    pip \
    xauth \
    x11-xserver-utils \
    x11-apps \
    software-properties-common \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    libx11-dev \
    python3-tk \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
# Linux default main user with UID 1000
# USERNAME is not important. files created will still be accessable 
ARG USERNAME=atari
ARG USER_UID=1000 
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

# biconmp dependencies
RUN sudo apt-get update \
  && sudo apt-get install -y \
  coinor-libipopt-dev \
  libeigen3-dev \
  && rm -rf /var/lib/apt/lists/*

# Pybind11 global for biconmp
RUN pip install pybind11[global]

# python dependencies
RUN python3 -m pip install pytest pybullet matplotlib pandas scipy jupyter meshcat PyQt5 opencv-python
RUN python3 -m pip install --upgrade wheel pyzmp xacro

# Neural Network Specific Dependencies
RUN pip3 install torch torchvision torchaudio
RUN python3 -m pip install tqdm hydra-core tensorboard torch-tb-profiler h5py wandb moviepy imageio
RUN pip install hydra-submitit-launcher --upgrade

# Debug Dependencies
RUN pip3 install memray memory-profiler Pympler

# install robotpkg
RUN sudo mkdir -p /etc/apt/keyrings
RUN curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc
RUN sudo tee /etc/apt/sources.list.d/robotpkg.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub jammy robotpkg
EOF
RUN sudo apt-get update


# manually install crocoddyl 1.9.0 and its dependencies
RUN sudo apt-get update \
  && sudo apt-get install -y \
  robotpkg-py310-eigenpy=2.7.11 \
  robotpkg-hpp-fcl=1.8.1 robotpkg-py310-hpp-fcl=1.8.1 \
  robotpkg-pinocchio=2.6.9 robotpkg-py310-pinocchio=2.6.9 \
  robotpkg-example-robot-data=4.0.1 robotpkg-py310-example-robot-data=4.0.1 \
  robotpkg-py310-crocoddyl=1.9.0 \
  && rm -rf /var/lib/apt/lists/*

# Copy the entrypoint and bashrc scripts so we have 
# our container's environment set up correctly
COPY build_packages.sh /build_packages.sh
COPY bashrc /home/${USERNAME}/.bashrc
COPY build_biconvex_mpc.sh /build_biconvex_mpc.sh

# Set up entrypoint and default command
#ENTRYPOINT ["/bin/bash", "/build_dependencies.sh"]

CMD ["bash"]

# Good practice
USER root