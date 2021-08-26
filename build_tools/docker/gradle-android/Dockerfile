# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android using CMake.

# Setup Java
# FROM openjdk:11 AS install-jdk
# WORKDIR /install-jdk
# CMD ["ls", "-las", "."]


FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS install-jdk

ARG JDK_VERSION=11
WORKDIR /install-jdk
RUN apt-get update && apt-get install -y openjdk-11-jdk

ARG GRADLE_VERSION=7.1.1
ARG GRADLE_DIST=bin
WORKDIR /install-gradle
RUN cd /opt && \
    wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-${GRADLE_DIST}.zip && \
    unzip gradle*.zip && \
    rm gradle*.zip && \
    ln -s /opt/gradle-${GRADLE_VERSION}/bin/gradle /usr/bin/gradle
