#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
artifact_root=$(realpath -e "$script_dir/..")
proot_version=5.3.1
proot_commit=99a8417521645c6d0b6d2b64a504bf27fea5d4da
proot_sha256=966afe32bf9a9d0e80836a8874d4dd829c51750060d9e0f30d330b1ed7eec8c2
proot_url=https://codeload.github.com/proot-me/proot/tar.gz/refs/tags/v5.3.1
talloc_version=2.3.3
talloc_sha256=6be95b2368bd0af1c4cd7a88146eb6ceea18e46c3ffc9330bf6262b40d1d8aaa
talloc_url=https://download.samba.org/pub/talloc/talloc-2.3.3.tar.gz
deps_root=$(realpath -m "${AE_DEPS_DIR:-$artifact_root/.deps}")
proot_bin="$deps_root/proot-$proot_version/bin/proot"

die() {
    echo "setup-proot: $*" >&2
    exit 1
}

valid_sha256() {
    [[ $1 =~ ^[0-9a-f]{64}$ ]]
}

safe_deps_root() {
    [[ -n $1 && $1 != / ]] || return 1
    case "$1/" in
        "$artifact_root/"*) return 0 ;;
        *) return 1 ;;
    esac
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

download() {
    local url=$1 expected=$2 destination=$3 partial
    if [[ -f $destination ]]; then
        printf '%s  %s\n' "$expected" "$destination" | sha256sum -c - >/dev/null \
            || die "existing download has the wrong SHA-256: $destination"
        return
    fi
    partial="$destination.partial.$$"
    trap 'rm -f -- "$partial"' RETURN
    curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
        --connect-timeout 15 --max-time 120 --retry 3 --retry-all-errors \
        --output "$partial" "$url"
    printf '%s  %s\n' "$expected" "$partial" | sha256sum -c - >/dev/null \
        || die "downloaded archive has the wrong SHA-256: $url"
    mv -- "$partial" "$destination"
    trap - RETURN
}

extract() {
    local archive=$1 expected=$2 destination=$3 temporary
    if [[ -f $destination/.source.sha256 ]]; then
        [[ $(<"$destination/.source.sha256") == "$expected" ]] \
            || die "source marker mismatch: $destination"
        return
    fi
    [[ ! -e $destination ]] || die "incomplete source directory exists: $destination"
    temporary="$destination.partial.$$"
    mkdir "$temporary"
    trap 'find "$temporary" -depth -delete 2>/dev/null || true' RETURN
    tar --extract --gzip --file "$archive" --directory "$temporary" \
        --strip-components=1 --no-same-owner
    printf '%s\n' "$expected" > "$temporary/.source.sha256"
    mv -- "$temporary" "$destination"
    trap - RETURN
}

find_talloc_runtime() {
    local ldconfig_bin candidate
    for ldconfig_bin in "$(command -v ldconfig 2>/dev/null || true)" /sbin/ldconfig /usr/sbin/ldconfig; do
        [[ -n $ldconfig_bin && -x $ldconfig_bin ]] || continue
        candidate=$($ldconfig_bin -p 2>/dev/null | awk '$1 == "libtalloc.so.2" { print $NF; exit }')
        [[ -n $candidate && -f $candidate ]] && { printf '%s\n' "$candidate"; return 0; }
    done
    for candidate in /lib/x86_64-linux-gnu/libtalloc.so.2 /usr/lib/x86_64-linux-gnu/libtalloc.so.2; do
        [[ -f $candidate ]] && { realpath -e "$candidate"; return 0; }
    done
    return 1
}

build_talloc_runtime() {
    local source=$1 prefix=$2 jobs=$3 python_bin candidate log_dir
    if [[ -f $prefix/lib/libtalloc.so.2 ]]; then
        printf '%s\n' "$prefix/lib/libtalloc.so.2"
        return
    fi
    [[ -f /usr/include/tirpc/rpc/rpc.h ]] \
        || die "building talloc requires /usr/include/tirpc (Debian/Ubuntu package: libtirpc-dev)"
    for candidate in "${TALLOC_BUILD_PYTHON:-}" /usr/bin/python3 "$(command -v python3 2>/dev/null || true)"; do
        [[ -n $candidate && -x $candidate ]] || continue
        if "$candidate" -c 'import distutils' >/dev/null 2>&1; then
            python_bin=$candidate
            break
        fi
    done
    [[ -n ${python_bin:-} ]] \
        || die "building talloc 2.3.3 requires Python with distutils (validated: /usr/bin/python3 on Ubuntu 22.04)"
    log_dir="$deps_root/logs"
    mkdir -p "$log_dir"
    if ! (
        cd "$source"
        PYTHON="$python_bin" CFLAGS=-I/usr/include/tirpc LDFLAGS=-ltirpc \
            ./configure --disable-python --prefix="$prefix"
        make --jobs "$jobs"
        make install
    ) > "$log_dir/talloc-build.log" 2>&1; then
        tail -80 "$log_dir/talloc-build.log" >&2 || true
        die "private talloc build failed; see $log_dir/talloc-build.log"
    fi
    [[ -f $prefix/lib/libtalloc.so.2 ]] || die "private talloc build did not produce libtalloc.so.2"
    echo "private talloc build: PASS" >&2
    printf '%s\n' "$prefix/lib/libtalloc.so.2"
}

check_proot() {
    [[ -x $proot_bin ]] || die "PRoot is not built; run: make setup-proot"
    mkdir -p "$deps_root/tmp"
    PROOT_TMP_DIR="$deps_root/tmp" "$proot_bin" /bin/true
    PROOT_TMP_DIR="$deps_root/tmp" "$proot_bin" --version \
        | grep -q "v$proot_version-${proot_commit:0:8}" \
        || die "unexpected PRoot version: $proot_bin"
    echo "proot check: PASS ($proot_bin)"
}

build_proot() {
    local downloads sources proot_source talloc_source talloc_prefix pkgconfig runtime runtime_origin jobs proot_log
    safe_deps_root "$deps_root" \
        || die "AE_DEPS_DIR must stay below the artifact root: $artifact_root"
    [[ $(uname -m) == x86_64 ]] || die "the validated PRoot build supports x86_64 only"
    for command_name in awk curl file gcc grep make pkg-config realpath sed sha256sum tar; do
        require_command "$command_name"
    done

    downloads="$deps_root/downloads"
    sources="$deps_root/sources"
    proot_source="$sources/proot-$proot_version"
    talloc_source="$sources/talloc-$talloc_version"
    talloc_prefix="$deps_root/talloc-$talloc_version"
    pkgconfig="$deps_root/pkgconfig"
    mkdir -p "$downloads" "$sources" "$pkgconfig" "$deps_root/proot-$proot_version/bin" "$deps_root/tmp"
    download "$proot_url" "$proot_sha256" "$downloads/proot-$proot_version.tar.gz"
    download "$talloc_url" "$talloc_sha256" "$downloads/talloc-$talloc_version.tar.gz"
    extract "$downloads/proot-$proot_version.tar.gz" "$proot_sha256" "$proot_source"
    extract "$downloads/talloc-$talloc_version.tar.gz" "$talloc_sha256" "$talloc_source"

    # The release tarball has no .git directory, while this upstream makefile
    # invokes a hard-coded git command even when GIT=false. Route both lookups
    # through the documented override; this affects version discovery only.
    if grep -q '\$(shell git ' "$proot_source/src/GNUmakefile"; then
        sed -i 's/$(shell git /$(shell $(GIT) /g' "$proot_source/src/GNUmakefile"
    fi
    if grep -q '`git rev-list' "$proot_source/src/GNUmakefile"; then
        sed -i 's/`git rev-list/`$(GIT) rev-list/' "$proot_source/src/GNUmakefile"
    fi

    jobs=${PROOT_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
    (( jobs > 8 )) && jobs=8
    if runtime=$(find_talloc_runtime); then
        runtime_origin=host
    else
        runtime=$(build_talloc_runtime "$talloc_source" "$talloc_prefix" "$jobs")
        runtime_origin=source_build
    fi
    cat > "$pkgconfig/talloc.pc" <<EOF
prefix=$talloc_source
includedir=\${prefix}
Name: talloc
Description: pinned talloc header with host runtime
Version: $talloc_version
Libs: -L$(dirname "$runtime") -Wl,-rpath,$(dirname "$runtime") -l:$(basename "$runtime")
Cflags: -I\${includedir}
EOF

    proot_log="$deps_root/logs/proot-build.log"
    mkdir -p "$deps_root/logs"
    if ! (
        cd "$proot_source/src"
        export PKG_CONFIG_PATH="$pkgconfig"
        export TMPDIR="$deps_root/tmp"
        make build.h GIT=false
        if grep -q '#define VERSION "-"' build.h; then
            sed -i "s/#define VERSION \"-\"/#define VERSION \"v$proot_version-${proot_commit:0:8}\"/" build.h
        fi
        make --jobs "$jobs" proot GIT=false
    ) > "$proot_log" 2>&1; then
        tail -80 "$proot_log" >&2 || true
        die "PRoot build failed; see $proot_log"
    fi
    echo "PRoot build: PASS"
    install -m 0755 "$proot_source/src/proot" "$proot_bin"
    {
        printf 'component=PRoot\nversion=%s\ncommit=%s\nsource_url=%s\nsource_sha256=%s\n' \
            "$proot_version" "$proot_commit" "$proot_url" "$proot_sha256"
        printf 'talloc_version=%s\ntalloc_source_url=%s\ntalloc_source_sha256=%s\n' \
            "$talloc_version" "$talloc_url" "$talloc_sha256"
        printf 'build_patch=route upstream version lookup through GIT=false\n'
        printf 'talloc_runtime=%s\n' "$runtime"
        printf 'talloc_runtime_origin=%s\n' "$runtime_origin"
        file "$proot_bin"
        sha256sum "$proot_bin"
    } > "$deps_root/proot-$proot_version/PROVENANCE.txt"
    check_proot
}

case "${1:-build}" in
    build)
        [[ $# == 0 || $# == 1 ]] || die "usage: $0 [build|check|self-test]"
        build_proot
        ;;
    check)
        [[ $# == 1 ]] || die "usage: $0 [build|check|self-test]"
        check_proot
        ;;
    self-test)
        [[ $# == 1 ]] || die "usage: $0 [build|check|self-test]"
        valid_sha256 "$proot_sha256"
        valid_sha256 "$talloc_sha256"
        ! valid_sha256 bad
        safe_deps_root "$artifact_root/.deps"
        ! safe_deps_root /
        echo "setup-proot self-test: PASS"
        ;;
    *)
        die "usage: $0 [build|check|self-test]"
        ;;
esac
