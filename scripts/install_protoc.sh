#!/usr/bin/env bash
set -euo pipefail

# Pinned protoc install (from vllm-project/vllm tools/install_protoc.sh).
# Distro protobuf-compiler versions are often too old for vLLM's Rust frontend.

if [[ $(id -u) -ne 0 ]]; then
    echo "Must be run as root" >&2
    exit 1
fi

VERSION="${PROTOC_VERSION:-34.2}"

ARCH="$(uname -m)"
case "${ARCH}" in
    aarch64|arm64) URL_ARCH="aarch_64" ;;
    x86_64|amd64) URL_ARCH="x86_64" ;;
    *) echo "Unsupported arch for protoc binary: ${ARCH}" >&2; exit 1 ;;
esac

URL="https://github.com/protocolbuffers/protobuf/releases/download/v${VERSION}/protoc-${VERSION}-linux-${URL_ARCH}.zip"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

echo "Downloading: ${URL}"
curl -fsSL -o "${TMPDIR}/protoc.zip" "${URL}"
unzip -q -o "${TMPDIR}/protoc.zip" -d /usr/local
echo "Installed $(protoc --version)"
