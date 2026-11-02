# Deliberately separate build and clean desktop roots. Neither is a release image.
ARG UBUNTU_IMAGE=ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956
FROM ${UBUNTU_IMAGE} AS common
ARG DEBIAN_FRONTEND=noninteractive
# The ordinary Ubuntu archive, accepted only with this image's archive keyring.
# The dated snapshot answered 503 for every suite (run 39601252706), so apt
# inputs are the archive state at build time and are not reproducible; the
# build keeps the signed InRelease files and exact package/source versions.
# Upgrading the image first keeps every installed version fetchable as source.
RUN printf '%s\n' \
    'deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://archive.ubuntu.com/ubuntu jammy main universe' \
    'deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://archive.ubuntu.com/ubuntu jammy-updates main universe' \
    'deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://security.ubuntu.com/ubuntu jammy-security main universe' \
    > /etc/apt/sources.list \
    && apt-get update && apt-get -y --no-install-recommends dist-upgrade \
    && apt-get install -y --no-install-recommends ca-certificates python3 git curl
FROM common AS builder
RUN sed -n 's/^deb /deb-src /p' /etc/apt/sources.list > /etc/apt/sources.list.d/native-sources.list \
    && apt-get update && apt-get -y --no-install-recommends dist-upgrade
# Jammy's libwebkit2gtk-4.1-dev does not depend on the WebKit2 typelib package.
RUN apt-get install -y --no-install-recommends build-essential pkg-config file \
    libgirepository1.0-dev libcairo2-dev libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1 \
    libgtk-3-bin librsvg2-common glib-networking gsettings-desktop-schemas \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good bubblewrap xdg-dbus-proxy \
    adwaita-icon-theme hicolor-icon-theme shared-mime-info
RUN curl -fLsS https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-unknown-linux-gnu.tar.gz -o /tmp/uv.tar.gz \
    && echo '90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb  /tmp/uv.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/uv.tar.gz -C /opt && ln -s /opt/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv
FROM common AS desktop
# libGL/libEGL/libGLESv2 (libglvnd), Mesa, gbm and drm are the declared host
# graphics ABI. The shipped libepoxy aborts when a non-GLX GLES2 context needs
# libGLESv2.so.2 and dlopen fails (run 39636484833 had no libgles2).
RUN apt-get install -y --no-install-recommends xvfb openbox wmctrl x11-utils x11-apps \
    dbus-x11 fonts-dejavu-core libgl1-mesa-dri libgl1 libegl1 libgles2 libgbm1 libdrm2 imagemagick \
    && for package in libwebkit2gtk-4.1-0 libgtk-3-0 gir1.2-webkit2-4.1 python3-gi; do \
         if dpkg-query -W "$package" 2>/dev/null; then exit 1; fi; \
       done
# No WebKit sandbox variable is set either way; qualification records WebKit's
# own choice and never runs as root or privileged.
RUN useradd -m -u 1000 qualifier
USER qualifier
