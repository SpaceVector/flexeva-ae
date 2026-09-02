#!/usr/bin/env bash
set -euo pipefail

action=${1:-build}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
artifact_root=$(cd -- "$script_dir/../../.." && pwd)
bundle_root="$script_dir/bundle"
source_root=${ASTRA_TABLE7_ROOT:-"$artifact_root/.deps/astra-sim-table7"}
source_mirror=${ASTRA_TABLE7_MIRROR:-gitee}
fetch_jobs=${ASTRA_TABLE7_FETCH_JOBS:-8}
gitee_root=https://gitee.com/space-line-vector
upstream_commit=518bd513ae110428cd62eb60efc0f3993fd53c70
chakra_commit=0e3cd40c569f0a4cacb6d961bb56be53407abd2f
marker="$source_root/.flexmaya-table7-source"
binary="$source_root/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default"
python_bin=${CANONICAL_PYTHON:-${PYTHON:-python3}}
required_submodules=(
    extern/graph_frontend/chakra
    extern/helper/fmt
    extern/helper/spdlog
    extern/network_backend/ns-3
    extern/remote_memory_backend/analytical
)

die() {
    echo "setup-astra-table7: $*" >&2
    exit 2
}

case "$source_mirror" in
    gitee) upstream_url="$gitee_root/astra-sim.git" ;;
    github) upstream_url=https://github.com/astra-sim/astra-sim.git ;;
    *) die "ASTRA_TABLE7_MIRROR must be gitee or github" ;;
esac

check_bundle() {
    command -v sha256sum >/dev/null || die "sha256sum is required"
    (cd "$bundle_root" && sha256sum --quiet -c CHECKSUMS.sha256) || die "bundle checksum mismatch"
    git apply --numstat "$bundle_root/astra-ras-ns3.patch" >/dev/null || die "invalid ASTRA-Sim patch"
    "$python_bin" "$bundle_root/overlay/scripts/measure_table7_backend_generality.py" self-test
}

check_location() {
    mkdir -p "$(dirname -- "$source_root")"
    local artifact_real source_real
    artifact_real=$(realpath -e "$artifact_root")
    source_real=$(realpath -m "$source_root")
    case "$source_real/" in
        "$artifact_real/.deps/"*) ;;
        *) die "ASTRA_TABLE7_ROOT must remain below $artifact_real/.deps" ;;
    esac
}

check_source() {
    [[ -f "$marker" ]] || die "source is not prepared: $source_root"
    [[ $(git -C "$source_root" rev-parse HEAD) == "$upstream_commit" ]] || die "unexpected ASTRA-Sim commit"
    [[ $(git -C "$source_root/extern/graph_frontend/chakra" rev-parse HEAD) == "$chakra_commit" ]] || die "unexpected Chakra commit"
    (cd "$source_root" && sha256sum --quiet -c "$bundle_root/INSTALLED_CHECKSUMS.sha256") || die "installed source checksum mismatch"
    "$python_bin" "$source_root/scripts/measure_table7_backend_generality.py" self-test
}

prepare() {
    check_bundle
    check_location
    if [[ -f "$marker" ]]; then
        check_source
        return
    fi
    [[ ! -e "$source_root" ]] || die "$source_root exists without a valid marker; move it aside and retry"
    command -v git >/dev/null || die "git is required"
    git clone --filter=blob:none --no-checkout "$upstream_url" "$source_root" \
        || die "unable to clone ASTRA-Sim from $source_mirror"
    git -C "$source_root" checkout --detach "$upstream_commit"
    if [[ "$source_mirror" == gitee ]]; then
        while read -r path repository; do
            git -C "$source_root" config "submodule.$path.url" "$gitee_root/$repository.git"
        done <<'EOF'
extern/graph_frontend/chakra chakra
extern/helper/fmt fmt
extern/helper/spdlog spdlog
extern/network_backend/ns-3 astra-network-ns3
extern/remote_memory_backend/analytical astra-memory-analytical
EOF
    fi
    git -C "$source_root" submodule update --init --depth 1 --jobs "$fetch_jobs" \
        "${required_submodules[@]}" \
        || die "unable to fetch ASTRA-Sim submodules from $source_mirror"
    git -C "$source_root" apply "$bundle_root/astra-ras-ns3.patch"
    cp -a "$bundle_root/overlay/." "$source_root/"
    printf 'mirror=%s\nupstream=%s\ncommit=%s\nbundle_sha256=%s\n' \
        "$source_mirror" "$upstream_url" "$upstream_commit" \
        "$(sha256sum "$bundle_root/CHECKSUMS.sha256" | cut -d' ' -f1)" >"$marker"
    check_source
}

case "$action" in
    self-test)
        check_bundle
        ;;
    prepare)
        prepare
        ;;
    build)
        prepare
        command -v cmake >/dev/null || die "cmake is required"
        command -v protoc >/dev/null || die "protoc is required"
        command -v mpicxx >/dev/null || die "an MPI C++ compiler is required"
        ASTRA_YAML_CPP_SOURCE=${ASTRA_YAML_CPP_SOURCE:-"$bundle_root/vendor/yaml-cpp"} \
            "$source_root/scripts/build_astra_ns3.sh"
        [[ -x "$binary" ]] || die "build completed without $binary"
        echo "ASTRA-Sim Table 7 build: PASS"
        ;;
    check)
        check_location
        check_source
        [[ -x "$binary" ]] || die "ASTRA-Sim binary is unavailable: $binary"
        echo "ASTRA-Sim Table 7 check: PASS"
        ;;
    *)
        die "usage: $0 {self-test|prepare|build|check}"
        ;;
esac
