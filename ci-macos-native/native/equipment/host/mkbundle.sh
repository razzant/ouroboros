#!/bin/bash
# Host side: the guest's candidate bundle and its bind file (test equipment). Reads the product repository's
# objects only (GIT_OPTIONAL_LOCKS=0; no ref, index, worktree or config of any product checkout is written).
#   mkbundle.sh commit SHA OUT_DIR            the parent's committed final product (preferred): bundle SHA as-is
#   mkbundle.sh lab TREE PARENT... OUT_DIR    only after the parent approves the lab seam: ONE lab-only commit object
#                                             wrapping TREE (fixed lab identity and date), written to a fresh scratch
#                                             object store under OUT_DIR that borrows the repository's objects through
#                                             objects/info/alternates (not a clone; nothing written to the repository)
# The bundle carries refs/heads/qb1-candidate and the official tags v7.7.0 v7.7.1 v7.8.0 (refused unless they name their
# pinned official commits); everything reachable from v7.7.0's commit is left out (the guest mirror has it from the
# official 7.7.0/7.8.0 app seeds; the guest verifies). SHA is a full commit id. With PUBLISHED_O1=<full id of a published
# official commit that CAND lacks> it also carries refs/heads/qb1-published-o1 and binds PUB_O0 = merge-base(CAND, O1)
# and PUB_O1 (the readiness range; that both are published is the caller's own readback, not checked here).
# VTAG_OFFICIAL is the repository's v<CAND's VERSION> commit, if any: the seed's lab provenance tag shadows that name.
# Writes OUT_DIR/cand-<tree8>.bundle, OUT_DIR/bind.env and OUT_DIR/mkbundle.json, create-only.
set -euo pipefail
export GIT_OPTIONAL_LOCKS=0 LC_ALL=C
REPO="${OUROBOROS_QA_PRODUCT_REPO:?Name the exact parent candidate repository}"
BASE_EXCLUDE=e41596c389c3c33c3155637aaf67f8bebbf4934d   # v7.7.0's commit
OFFICIAL_TAGS='v7.7.0=e41596c389c3c33c3155637aaf67f8bebbf4934d v7.7.1=fc8cd8146671fdcb71d336b9b49970b6f9893a0a v7.8.0=6f3ddff05ff3e6c6f8d007ff96358df6af872e83'
O1="${PUBLISHED_O1:-}"
full_id() { [ "${#1}" = 40 ] && [ -z "$(printf '%s' "$1" | tr -d 0-9a-f)" ]; }
LAB_DATE='2026-10-29T12:00:00+00:00'
mode="${1:?commit|lab}"; shift
case "$mode" in
    commit) CAND="${1:?SHA}"; OUT="${2:?OUT_DIR}"; STORE="$REPO"
            full_id "$CAND" || { echo "not a full commit id: $CAND" >&2; exit 2; }
            [ "$(git -C "$REPO" cat-file -t "$CAND")" = commit ] || { echo "not a commit: $CAND" >&2; exit 2; } ;;
    lab)    TREE="${1:?TREE}"; shift; OUT="${!#}"; PARENTS=("${@:1:$#-1}")
            [ "${#PARENTS[@]}" -ge 1 ] || { echo "lab TREE PARENT... OUT_DIR" >&2; exit 2; }
            [ "$(git -C "$REPO" cat-file -t "$TREE")" = tree ] || { echo "not a tree: $TREE" >&2; exit 2; }
            mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
            STORE=$(/usr/bin/mktemp -d "$OUT/object-store.XXXXXX")
            git init -q --bare "$STORE"
            git -C "$REPO" rev-parse --path-format=absolute --git-path objects > "$STORE/objects/info/alternates"
            args=(); for p in "${PARENTS[@]}"; do args+=(-p "$p"); done
            CAND=$(GIT_AUTHOR_NAME=qb1-lab GIT_AUTHOR_EMAIL=lab@lab.invalid GIT_AUTHOR_DATE="$LAB_DATE" \
                   GIT_COMMITTER_NAME=qb1-lab GIT_COMMITTER_EMAIL=lab@lab.invalid GIT_COMMITTER_DATE="$LAB_DATE" \
                   git -C "$STORE" commit-tree "$TREE" "${args[@]}" -m "lab: final product tree $TREE (bd01bf0f guest fixture)") ;;
    *) echo "commit|lab" >&2; exit 2 ;;
esac
TREE=$(git -C "$STORE" rev-parse "$CAND^{tree}")
for pin in $OFFICIAL_TAGS; do
    got=$(git -C "$REPO" rev-parse -q --verify "refs/tags/${pin%%=*}^{commit}" || true)
    [ "$got" = "${pin#*=}" ] || { echo "tag ${pin%%=*} names ${got:-nothing}, not the official ${pin#*=}" >&2; exit 2; }
done
O0=
if [ -n "$O1" ]; then
    full_id "$O1" && [ "$(git -C "$REPO" cat-file -t "$O1")" = commit ] || { echo "PUBLISHED_O1 is not a full commit id: $O1" >&2; exit 2; }
    ! git -C "$STORE" merge-base --is-ancestor "$O1" "$CAND" || { echo "PUBLISHED_O1 is already contained in CAND" >&2; exit 2; }
    O0=$(git -C "$STORE" merge-base "$CAND" "$O1") || { echo "CAND and PUBLISHED_O1 share no history" >&2; exit 2; }
fi
VER=$(git -C "$STORE" cat-file blob "$CAND:VERSION" | tr -d ' \n')
VTAG=$(git -C "$REPO" rev-parse -q --verify "refs/tags/v$VER^{commit}" || true)
[ -z "$VTAG" ] || [ "$VTAG" = "$CAND" ] || echo "disclosure: v$VER is $VTAG here; the seed's lab tag v$VER -> CAND shadows it" >&2
NAME="cand-${TREE:0:8}.bundle"
[ ! -e "$OUT/$NAME" ] && [ ! -e "$OUT/bind.env" ] && [ ! -e "$OUT/mkbundle.json" ] || { echo "exists in $OUT" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
REFS=$(/usr/bin/mktemp -d "$OUT/bundle-refs.XXXXXX")
git init -q --bare "$REFS"
git -C "$STORE" rev-parse --path-format=absolute --git-path objects > "$REFS/objects/info/alternates"
if [ "$STORE" != "$REPO" ]; then
    git -C "$REPO" rev-parse --path-format=absolute --git-path objects >> "$REFS/objects/info/alternates"
fi
git -C "$REFS" update-ref refs/heads/qb1-candidate "$CAND"
for t in v7.7.0 v7.7.1 v7.8.0; do git -C "$REFS" update-ref "refs/tags/$t" "$(git -C "$REPO" rev-parse "refs/tags/$t")"; done
HEADS=(refs/heads/qb1-candidate)
if [ -n "$O1" ]; then git -C "$REFS" update-ref refs/heads/qb1-published-o1 "$O1"; HEADS+=(refs/heads/qb1-published-o1); fi
git -C "$REFS" bundle create "$OUT/$NAME" "${HEADS[@]}" refs/tags/v7.7.0 refs/tags/v7.7.1 refs/tags/v7.8.0 \
    "^$BASE_EXCLUDE"
git -C "$REFS" bundle verify "$OUT/$NAME" > "$OUT/$NAME.verify.txt" 2>&1
SHA=$(/usr/bin/shasum -a 256 "$OUT/$NAME" | cut -c1-64)
LOCK_FROM='pyyaml==6.0.3'; LOCK_TO='pyyaml==6.0.2'
[ "$(git -C "$REFS" cat-file blob "$CAND:requirements-runtime.lock" | grep -c -x -F "$LOCK_FROM")" = 1 ] \
    || { echo "lock lacks exactly one $LOCK_FROM" >&2; exit 2; }
CLOCK=$(git -C "$REFS" cat-file blob "$CAND:requirements-runtime.lock" | /usr/bin/shasum -a 256 | cut -c1-64)
printf 'CAND=%s\nCTREE=%s\nCLOCK=%s\nCBUNDLE_NAME=%s\nCBUNDLE_SHA=%s\nPLUS_FROM=%s\nPLUS_TO=%s\nPLUS_DATE=%s\nPUB_O0=%s\nPUB_O1=%s\nVTAG_OFFICIAL=%s\n' \
    "$CAND" "$TREE" "$CLOCK" "$NAME" "$SHA" "$LOCK_FROM" "$LOCK_TO" "$LAB_DATE" "$O0" "$O1" "$VTAG" > "$OUT/bind.env"
{ printf '{"mode": "%s", "cand": "%s", "tree": "%s", "parents": "%s", "bundle": "%s", "bundle_sha256": "%s", ' \
      "$mode" "$CAND" "$TREE" "$(git -C "$REFS" rev-parse "$CAND^@" | tr '\n' ' ')" "$NAME" "$SHA"
  printf '"official_tags": "%s", "published_o0": "%s", "published_o1": "%s", "version": "%s", "version_tag_official": "%s", ' \
      "$OFFICIAL_TAGS" "$O0" "$O1" "$VER" "$VTAG"
  printf '"lock_sha256": "%s", ' "$CLOCK"
  printf '"bundle_bytes": %s, "store": "%s", "excluded": "%s", "bind_env_sha256": "%s"}\n' \
      "$(stat -f %z "$OUT/$NAME")" "$STORE" "$BASE_EXCLUDE" "$(/usr/bin/shasum -a 256 "$OUT/bind.env" | cut -c1-64)"; } > "$OUT/mkbundle.json"
git -C "$REFS" bundle list-heads "$OUT/$NAME" >> "$OUT/$NAME.verify.txt"
cat "$OUT/mkbundle.json"
