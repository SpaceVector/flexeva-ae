#!/usr/bin/env bash

AE_ROOT="${AE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
AE_PROFILE="$AE_ROOT/.deps/ae_env.sh"

if [[ -r "$AE_PROFILE" ]]; then
    source "$AE_PROFILE"
    return 0
fi

if [[ -z "${FLEXMAYA_PEER_TARGET:-}" ]]; then
    ssh_config="$HOME/.ssh/config"
    [[ -r "$ssh_config" ]] || {
        echo "AE environment: SSH config is unavailable" >&2
        return 2
    }
    mapfile -t peer_aliases < <(
        awk 'tolower($1) == "host" {
            for (i = 2; i <= NF; i++) if ($i !~ /[*?!]/) print $i
        }' "$ssh_config"
    )
    [[ "${#peer_aliases[@]}" == 1 ]] || {
        echo "AE environment: expected exactly one concrete SSH peer alias" >&2
        return 2
    }
    FLEXMAYA_PEER_TARGET="${peer_aliases[0]}"
fi

[[ "$FLEXMAYA_PEER_TARGET" =~ ^[A-Za-z0-9_.-]+(@[A-Za-z0-9_.:-]+)?$ ]] || {
    echo "AE environment: invalid SSH peer target" >&2
    return 2
}

if [[ -z "${FLEXMAYA_PEER_PORT:-}" ]]; then
    FLEXMAYA_PEER_PORT="$(ssh -G "$FLEXMAYA_PEER_TARGET" 2>/dev/null \
        | awk '$1 == "port" { print $2; exit }')"
fi
[[ "$FLEXMAYA_PEER_PORT" =~ ^[0-9]+$ \
    && "$FLEXMAYA_PEER_PORT" -ge 1 && "$FLEXMAYA_PEER_PORT" -le 65535 ]] || {
    echo "AE environment: invalid SSH peer port" >&2
    return 2
}

AE_NODE_ROOT="${AE_NODE_ROOT:-$(cd -- "$AE_ROOT/.." && pwd -P)}"
FLEXMAYA_MASTER_ADDR="${FLEXMAYA_MASTER_ADDR:-}"
FLEXMAYA_MASTER_PORT="${FLEXMAYA_MASTER_PORT:-29500}"
FLEXMAYA_CONTROL_PORT="${FLEXMAYA_CONTROL_PORT:-29600}"

export AE_NODE_ROOT FLEXMAYA_MASTER_ADDR FLEXMAYA_MASTER_PORT FLEXMAYA_CONTROL_PORT
export FLEXMAYA_PEER_TARGET FLEXMAYA_PEER_PORT
