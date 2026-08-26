###############################################################################
# BLT Tutorial Dockerfile
#
# Runs openvscode-server (browser-based VSCode) under supervisord so tutorial
# participants can work through the BLT tutorial (docs/tutorial) in their
# browser. Provides the C/C++/Fortran + OpenMPI toolchain expected by
# host-configs/other/radiuss-docker.cmake.
###############################################################################

FROM ghcr.io/llnl/radiuss:gcc-11-ubuntu-22.04

ARG USER=bltdev
ARG OPENVSCODE_VERSION=v1.109.5
ENV HOME=/home/${USER}

USER root

# Base image already provides gcc/g++ (11) and cmake. Add Fortran, OpenMPI,
# and supervisor (which runs openvscode-server in the foreground).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gfortran \
      openmpi-bin \
      libopenmpi-dev \
      git \
      curl \
      ca-certificates \
      supervisor \
 && rm -rf /var/lib/apt/lists/*

# Create the non-root user the tutorial session runs as
RUN useradd --create-home --shell /bin/bash ${USER}

# Install openvscode-server. The stable symlink at /opt/openvscode-server
# means supervisord.conf doesn't need to know the exact version.
WORKDIR /opt/archives
RUN curl -fL \
        "https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-${OPENVSCODE_VERSION}/openvscode-server-${OPENVSCODE_VERSION}-linux-x64.tar.gz" \
        -o "openvscode-server.tar.gz" \
 && tar xzf "openvscode-server.tar.gz" \
 && chown -R ${USER}:${USER} "openvscode-server-${OPENVSCODE_VERSION}-linux-x64" \
 && ln -s "/opt/archives/openvscode-server-${OPENVSCODE_VERSION}-linux-x64" /opt/openvscode-server \
 && rm "openvscode-server.tar.gz"

# Copy the BLT repo into the tutorial user's home
COPY --chown=${USER}:${USER} . ${HOME}/blt/

# supervisord config + log file
COPY ./containers/tutorial/supervisord.conf /etc/supervisord.conf
RUN touch /var/log/openvscode-server.log \
 && chown ${USER}:${USER} /var/log/openvscode-server.log

USER ${USER}
WORKDIR ${HOME}

EXPOSE 3000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
