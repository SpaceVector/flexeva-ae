#!/usr/bin/env bash
set -euo pipefail

GITEE_URL="https://gitee.com/space-line-vector/flexeva-ae.git"

fail() {
    echo "setup: $*" >&2
    return 2
}

sync_checkout() {
    local repo="$1" expected="$2" remote_url="$3" current branch fetched

    [[ "$repo" =~ ^/[A-Za-z0-9._/-]+$ ]] \
        || { fail "peer checkout path is invalid"; return 2; }
    [[ "$expected" =~ ^[0-9a-f]{40}$ ]] \
        || { fail "coordinator commit is invalid"; return 2; }
    [[ -d "$repo/.git" ]] \
        || { fail "peer checkout is missing: $repo"; return 2; }
    git -C "$repo" diff --quiet && git -C "$repo" diff --cached --quiet \
        || { fail "peer checkout has tracked changes; refusing to overwrite them"; return 2; }

    current="$(git -C "$repo" rev-parse HEAD)"
    [[ "$current" != "$expected" ]] || return 0

    branch="$(git -C "$repo" symbolic-ref --short -q HEAD || true)"
    [[ "$branch" == main ]] \
        || { fail "peer checkout must be on main before it can be updated"; return 2; }

    GIT_TERMINAL_PROMPT=0 git -C "$repo" fetch --quiet --no-tags "$remote_url" main \
        || { fail "cannot fetch peer update from Gitee"; return 2; }
    fetched="$(git -C "$repo" rev-parse FETCH_HEAD)"
    [[ "$fetched" == "$expected" ]] \
        || { fail "Gitee main does not match the coordinator commit"; return 2; }
    git -C "$repo" merge --ff-only --quiet FETCH_HEAD \
        || { fail "peer checkout cannot be fast-forwarded to the coordinator commit"; return 2; }
    [[ "$(git -C "$repo" rev-parse HEAD)" == "$expected" ]] \
        || { fail "peer checkout update did not reach the coordinator commit"; return 2; }
    echo "setup: peer checkout updated to $expected"
}

self_test() {
    local test_root remote source peer old_commit expected newer
    test_root="$(mktemp -d)"
    remote="$test_root/remote.git"
    source="$test_root/source"
    peer="$test_root/peer"

    (
        trap 'rm -r -- "$test_root"' EXIT
        git init -q --bare --initial-branch=main "$remote"
        git init -q --initial-branch=main "$source"
        git -C "$source" config user.name "AE self-test"
        git -C "$source" config user.email "ae-self-test@example.invalid"
        printf 'old\n' >"$source/state.txt"
        git -C "$source" add state.txt
        git -C "$source" commit -qm old
        git -C "$source" remote add origin "$remote"
        git -C "$source" push -q origin main
        old_commit="$(git -C "$source" rev-parse HEAD)"
        git clone -q "$remote" "$peer"

        printf 'new\n' >"$source/state.txt"
        git -C "$source" commit -qam new
        git -C "$source" push -q origin main
        expected="$(git -C "$source" rev-parse HEAD)"
        [[ "$(git -C "$peer" rev-parse HEAD)" == "$old_commit" ]]
        sync_checkout "$peer" "$expected" "$remote"
        [[ "$(git -C "$peer" rev-parse HEAD)" == "$expected" ]]

        printf 'reviewer change\n' >>"$peer/state.txt"
        printf 'newer\n' >"$source/state.txt"
        git -C "$source" commit -qam newer
        git -C "$source" push -q origin main
        newer="$(git -C "$source" rev-parse HEAD)"
        if sync_checkout "$peer" "$newer" "$remote" >/dev/null 2>&1; then
            fail "self-test overwrote a tracked peer change"
        fi
        [[ "$(git -C "$peer" rev-parse HEAD)" == "$expected" ]]
        echo "peer checkout sync self-test: PASS"
    )
}

if [[ "${1:-}" == self-test ]]; then
    [[ $# == 1 ]] || fail "usage: script/lib/sync_peer_checkout.sh self-test"
    self_test
else
    [[ $# == 2 ]] || fail "peer checkout sync requires REPO and COMMIT"
    sync_checkout "$1" "$2" "$GITEE_URL"
fi
