#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
proto_dir="$repo_root/extern/graph_frontend/chakra/schema/protobuf"
ns3_dir="$repo_root/extern/network_backend/ns-3"
jobs=${ASTRA_BUILD_JOBS:-16}
configure_args=()

if [[ -n ${ASTRA_YAML_CPP_SOURCE:-} ]]; then
    yaml_cpp_source=$(realpath -e "$ASTRA_YAML_CPP_SOURCE")
    [[ -f "$yaml_cpp_source/CMakeLists.txt" ]] || {
        echo "invalid ASTRA_YAML_CPP_SOURCE: $yaml_cpp_source" >&2
        exit 2
    }
    configure_args=(-- "-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=$yaml_cpp_source")
fi

protoc et_def.proto --proto_path="$proto_dir" --cpp_out="$proto_dir"
protoc et_def.proto --proto_path="$proto_dir" --python_out="$proto_dir"
cd "$ns3_dir"
# The AE servers log in as root; ns-3 checks only USER before doing a normal
# unprivileged build, so give its launcher a non-root label without using sudo.
USER=astra-sim-builder ./ns3 configure --enable-mpi "${configure_args[@]}"
USER=astra-sim-builder ./ns3 build AstraSimNetwork -j "$jobs"
