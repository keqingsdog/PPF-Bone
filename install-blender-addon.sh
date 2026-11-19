#!/bin/bash
# Fork modification: changed from the upstream baseline.
#
# Install / uninstall the addon as a manifest-compatible Blender extension.
#
# Drops a symlink at:
#   <BLENDER_BASE>/<version>/extensions/user_default/ppf_contact_solver
# pointing at a private ABI-specific staged addon directory. The manifest inside
# blender_addon/blender_manifest.toml is what makes Blender treat it
# as an extension; from there it loads under the module name
# bl_ext.user_default.ppf_contact_solver.
#
# Usage:
#   Install:   ./install-blender-addon.sh
#   Uninstall: ./install-blender-addon.sh --uninstall
#                 prompts (interactively, default No) to also wipe
#                 third-party deps the addon pip-installed into
#                 Blender's user `scripts/addons/modules/` dir.
#
# Cold-start (e.g. fresh CI runner where Blender hasn't booted yet):
#   PPF_BLENDER_BIN=/opt/blender/blender ./install-blender-addon.sh
# The version dir is derived from `<bin> --version` and created.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_SOURCE="$SCRIPT_DIR/blender_addon"
ADDON_NAME="ppf_contact_solver"
UNINSTALL=false

if [ "$1" = "--uninstall" ]; then
  UNINSTALL=true
fi

case "$(uname -s)" in
Darwin)
  BLENDER_BASE="$HOME/Library/Application Support/Blender"
  ;;
Linux)
  BLENDER_BASE="${XDG_CONFIG_HOME:-$HOME/.config}/blender"
  ;;
MINGW* | MSYS* | CYGWIN*)
  BLENDER_BASE="$APPDATA/Blender Foundation/Blender"
  ;;
*)
  echo "Unsupported OS: $(uname -s)"
  exit 1
  ;;
esac

# Resolve the explicit binary first; otherwise use the newest profile folder.
detect_version_from_existing() {
  if [ ! -d "$BLENDER_BASE" ]; then
    return 1
  fi
  ls -1 "$BLENDER_BASE" 2>/dev/null \
    | grep -E '^[0-9]+\.[0-9]+$' \
    | sort -V \
    | tail -n1
}

detect_version_from_binary() {
  local bin="$1"
  [ -x "$bin" ] || return 1
  # `blender --version` prints lines like "Blender 5.2.0 LTS\n  build date: ".
  # Keep the patch for the manifest gate; the profile directory is derived
  # separately as major.minor after the gate. LTS suffixes are accepted.
  "$bin" --version 2>/dev/null \
    | head -n1 \
    | sed -nE 's/^Blender ([0-9]+)\.([0-9]+)\.([0-9]+).*/\1.\2.\3/p'
}

read_manifest_minimum() {
  sed -nE 's/^[[:space:]]*blender_version_min[[:space:]]*=[[:space:]]*"([0-9]+\.[0-9]+\.[0-9]+)"[[:space:]]*(#.*)?$/\1/p' "$1"
}

version_at_least() {
  local found="$1" minimum="$2"
  local major minor patch minimum_major minimum_minor minimum_patch
  [[ "$found" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || return 1
  [[ "$minimum" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  IFS=. read -r major minor patch <<< "$found"
  IFS=. read -r minimum_major minimum_minor minimum_patch <<< "$minimum"
  patch="${patch:-0}"
  if [ "$major" -ne "$minimum_major" ]; then
    [ "$major" -gt "$minimum_major" ]; return
  fi
  if [ "$minor" -ne "$minimum_minor" ]; then
    [ "$minor" -gt "$minimum_minor" ]; return
  fi
  [ "$patch" -ge "$minimum_patch" ]
}

resolve_blender_version() {
  if [ -n "${PPF_BLENDER_BIN:-}" ]; then
    detect_version_from_binary "$PPF_BLENDER_BIN"
  else
    detect_version_from_existing
  fi
}

# Prefer the version a real binary reports when PPF_BLENDER_BIN is
# set: a stale `<version>/` directory left over from an earlier setup
# (e.g. 5.0/ when only 5.1 is now installed) would otherwise win the
# scan and the addon would land in the wrong tree. Without
# PPF_BLENDER_BIN, fall back to the newest version dir under
# `BLENDER_BASE` so an interactive install (Blender already booted at
# least once) keeps working.
BLENDER_VERSION="$(resolve_blender_version || true)"
if [ -n "${PPF_BLENDER_BIN:-}" ]; then
  if [ -z "$BLENDER_VERSION" ]; then
    echo "Could not parse version from PPF_BLENDER_BIN=$PPF_BLENDER_BIN"
    exit 1
  fi
  echo "Using Blender version $BLENDER_VERSION (from PPF_BLENDER_BIN)"
fi

if [ -z "$BLENDER_VERSION" ]; then
  echo "No Blender version directory found in $BLENDER_BASE."
  echo "Either launch Blender once to materialize it, or set "
  echo "PPF_BLENDER_BIN=<path/to/blender> and re-run."
  exit 1
fi

MINIMUM_BLENDER_VERSION="$(read_manifest_minimum "$ADDON_SOURCE/blender_manifest.toml")"
if [[ ! "$MINIMUM_BLENDER_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "blender_manifest.toml must declare blender_version_min as major.minor.patch"
  exit 1
fi
# The manifest is the single minimum-version source for both installers.
if ! version_at_least "$BLENDER_VERSION" "$MINIMUM_BLENDER_VERSION"; then
  echo "Blender $BLENDER_VERSION is not supported."
  echo "This addon requires Blender $MINIMUM_BLENDER_VERSION or later (blender_manifest.toml)."
  exit 1
fi
IFS=. read -r blender_major blender_minor blender_patch <<< "$BLENDER_VERSION"
BLENDER_VERSION="$blender_major.$blender_minor"

PROFILE_DIR="${BLENDER_USER_RESOURCES:-$BLENDER_BASE/$BLENDER_VERSION}"
EXT_DIR="$PROFILE_DIR/extensions/user_default"
EXT_LINK="$EXT_DIR/$ADDON_NAME"

# Legacy locations we sweep on every run so an old install can't shadow
# the canonical extension symlink. Two flavors of legacy:
#   1. scripts/addons/<id>            -- pre-extension layout (5.0 still loads it)
#   2. scripts/addons/ppf-contact-solver -- hyphenated dirname from very old installs
LEGACY_ADDONS_DIR="$PROFILE_DIR/scripts/addons"
LEGACY_LINK="$LEGACY_ADDONS_DIR/$ADDON_NAME"
LEGACY_HYPHEN_LINK="$LEGACY_ADDONS_DIR/ppf-contact-solver"

echo "Blender version: $BLENDER_VERSION"
echo "Extensions directory: $EXT_DIR"

# Handle uninstall
if [ "$UNINSTALL" = true ]; then
  removed=false
  for link in "$EXT_LINK" "$LEGACY_LINK" "$LEGACY_HYPHEN_LINK"; do
    if [ -e "$link" ] || [ -L "$link" ]; then
      rm -rf "$link"
      echo "Removed: $link"
      removed=true
    fi
  done
  if [ "$removed" = false ]; then
    echo "Addon not installed: $EXT_LINK"
  fi

  # Offer to wipe third-party Python deps that the addon's Install
  # operators (paramiko, docker) and the test rig's orchestrator
  # (cbor2) pip-install into Blender's user modules dir. The dir is
  # `bpy.utils.user_resource('SCRIPTS', path='addons/modules')`; see
  # blender_addon/core/module.py:get_install_target. Default is No;
  # non-tty stdin (CI) skips the prompt and leaves them in place.
  MODULES_DIR="$PROFILE_DIR/scripts/addons/modules"
  if [ -d "$MODULES_DIR" ] && [ -n "$(ls -A "$MODULES_DIR" 2>/dev/null)" ]; then
    echo
    echo "Third-party packages in $MODULES_DIR:"
    ls -1A "$MODULES_DIR" | sed 's/^/  /'
    if [ -t 0 ]; then
      printf "Also remove these? [y/N] "
      read -r answer
      case "$answer" in
        y | Y | yes | YES)
          rm -rf "$MODULES_DIR"
          echo "Removed: $MODULES_DIR"
          ;;
        *)
          echo "Kept: $MODULES_DIR"
          ;;
      esac
    else
      echo "(non-interactive stdin; left in place)"
    fi
  fi
  exit 0
fi

# Stage a separate extension for the target Blender's actual Python ABI.
[ -n "${PPF_BLENDER_BIN:-}" ] || { echo 'Set PPF_BLENDER_BIN to the target Blender executable.'; exit 1; }
PYTHON_BIN="${PPF_PYTHON_BIN:-python3}"
stage_parent="$PROFILE_DIR/extensions/.ppf-addon-builds"
mkdir -p "$stage_parent"
stage_root="$(mktemp -d "$stage_parent/build-XXXXXXXX")"
"$PYTHON_BIN" "$ADDON_SOURCE/wheels/build_package.py" --blender "$PPF_BLENDER_BIN" --stage-dir "$stage_root/addon"
ADDON_SOURCE="$stage_root/addon"
mkdir -p "$EXT_DIR"

# Sweep legacy installs unconditionally. Either the user is upgrading
# from the old layout or never had one; in both cases an existing
# scripts/addons/<id> symlink would have Blender load both the
# extension and the legacy copy, conflicting on operator names.
for legacy in "$LEGACY_LINK" "$LEGACY_HYPHEN_LINK"; do
  if [ -e "$legacy" ] || [ -L "$legacy" ]; then
    rm -rf "$legacy"
    echo "Removed legacy install: $legacy"
  fi
done

if [ -e "$EXT_LINK" ] && [ ! -L "$EXT_LINK" ]; then
  echo "Existing extension directory: $EXT_LINK. Remove or upgrade it through Blender first."
  exit 1
fi

# Idempotent replacement of the canonical symlink. ``ln -sfn`` swings
# the link atomically when the destination already exists, so a re-run
# is a no-op if nothing changed and a clean rebind otherwise.
if [ -L "$EXT_LINK" ] && [ "$(readlink "$EXT_LINK")" = "$ADDON_SOURCE" ]; then
  echo "Symlink already correct: $EXT_LINK -> $ADDON_SOURCE"
else
  ln -sfn "$ADDON_SOURCE" "$EXT_LINK"
  echo "Created symlink: $EXT_LINK -> $ADDON_SOURCE"
fi

echo
echo "Enable in Blender with:"
echo "    blender --addons bl_ext.user_default.$ADDON_NAME"
