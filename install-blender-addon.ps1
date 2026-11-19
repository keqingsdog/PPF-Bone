# Fork modification: changed from the upstream baseline.
# Install / uninstall the addon as a manifest-compatible Blender extension on Windows.
#
# Drops a directory junction at:
#   <BlenderBase>\<version>\extensions\user_default\ppf_contact_solver
# pointing at a private ABI-specific staged addon directory. The manifest inside
# blender_addon\blender_manifest.toml is what makes Blender treat it
# as an extension; from there it loads under the module name
# bl_ext.user_default.ppf_contact_solver.
#
# Usage:
#   Install:   .\install-blender-addon.ps1
#   Uninstall: .\install-blender-addon.ps1 -Uninstall
#                 prompts (interactively, default No) to also wipe
#                 third-party deps the addon pip-installed into
#                 Blender's user scripts\addons\modules\ dir.
#
# Cold-start (e.g. fresh runner where Blender hasn't booted yet):
#   $env:PPF_BLENDER_BIN="C:\path\to\blender.exe"; .\install-blender-addon.ps1
# The version dir is derived from `<bin> --version` and created.

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AddonSource = Join-Path $ScriptDir "blender_addon"
$AddonName = "ppf_contact_solver"
# Stale dirname from very old installs that used a hyphen. Python module
# names cannot contain hyphens, so this variant could never have worked
# under addon_enable; we sweep it on install/uninstall.
$LegacyAddonName = "ppf-contact-solver"

$BlenderBase = Join-Path $env:APPDATA "Blender Foundation\Blender"

# Resolve a Blender version directory. An explicit PPF_BLENDER_BIN wins
# over existing preference folders, so targeting 4.2 cannot install into
# an unrelated 5.1 profile. With no override, use the newest existing one.
function Get-VersionFromExisting {
    if (-not (Test-Path $BlenderBase)) { return $null }
    return Get-ChildItem -Path $BlenderBase -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } |
        Select-Object -Last 1 -ExpandProperty Name
}

function Get-VersionFromBinary($BinPath) {
    if (-not (Test-Path $BinPath)) { return $null }
    $out = & $BinPath --version 2>$null | Select-Object -First 1
    if ($out -match '^Blender (\d+)\.(\d+)(?:\.(\d+))?') {
        $patch = if ($Matches[3]) { $Matches[3] } else { '0' }
        return "$($Matches[1]).$($Matches[2]).$patch"
    }
    return $null
}

function Get-AddonMinimumVersion($ManifestPath) {
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw
    if ($manifest -notmatch '(?m)^\s*blender_version_min\s*=\s*"(\d+\.\d+\.\d+)"\s*(?:#.*)?$') {
        throw "blender_manifest.toml must declare blender_version_min as major.minor.patch"
    }
    return [version]$Matches[1]
}

function Test-BlenderMinimumVersion($DetectedVersion, [version]$MinimumVersion) {
    if ($DetectedVersion -notmatch '^\d+\.\d+(?:\.\d+)?$') { return $false }
    $normalized = if (($DetectedVersion -split '\.').Count -eq 2) { "$DetectedVersion.0" } else { $DetectedVersion }
    return ([version]$normalized -ge $MinimumVersion)
}

function Resolve-BlenderVersion($BinPath) {
    if ($BinPath) {
        $detected = Get-VersionFromBinary $BinPath
        if (-not $detected) { throw "Could not parse version from PPF_BLENDER_BIN=$BinPath" }
        return $detected
    }
    return Get-VersionFromExisting
}

$MinimumBlenderVersion = Get-AddonMinimumVersion (Join-Path $AddonSource "blender_manifest.toml")
$DetectedBlenderVersion = Resolve-BlenderVersion $env:PPF_BLENDER_BIN
if (-not $DetectedBlenderVersion) {
    Write-Error "No Blender version directory found in $BlenderBase. Either launch Blender once to materialize it, or set `$env:PPF_BLENDER_BIN and re-run."
    exit 1
}

# The manifest is the single minimum-version source for both installers.
if (-not (Test-BlenderMinimumVersion $DetectedBlenderVersion $MinimumBlenderVersion)) {
    Write-Error "Blender $DetectedBlenderVersion is not supported. This addon requires Blender $MinimumBlenderVersion or later (blender_manifest.toml)."
    exit 1
}
$BlenderVersion = ([version]$DetectedBlenderVersion).ToString(2)
if ($env:PPF_BLENDER_BIN) {
    Write-Host "Using Blender version $DetectedBlenderVersion (from PPF_BLENDER_BIN)"
}

$ExtDir = Join-Path $BlenderBase "$BlenderVersion\extensions\user_default"
$ExtLink = Join-Path $ExtDir $AddonName

# Legacy locations we sweep on every install run so an old layout cannot
# shadow the canonical extension junction.
$LegacyAddonsDir = Join-Path $BlenderBase "$BlenderVersion\scripts\addons"
$LegacyLink = Join-Path $LegacyAddonsDir $AddonName
$LegacyHyphenLink = Join-Path $LegacyAddonsDir $LegacyAddonName

Write-Host "Blender version: $BlenderVersion"
Write-Host "Extensions directory: $ExtDir"

if ($Uninstall) {
    $removed = $false
    foreach ($link in @($ExtLink, $LegacyLink, $LegacyHyphenLink)) {
        if (Test-Path $link) {
            Remove-Item -Path $link -Recurse -Force
            Write-Host "Removed: $link"
            $removed = $true
        }
    }
    if (-not $removed) {
        Write-Host "Addon not installed: $ExtLink"
    }

    # Offer to wipe third-party Python deps that the addon's Install
    # operators (paramiko, docker) and the test rig's orchestrator
    # (cbor2) pip-install into Blender's user modules dir. The dir is
    # bpy.utils.user_resource('SCRIPTS', path='addons/modules'); see
    # blender_addon\core\module.py:get_install_target. Default is No;
    # non-tty stdin (CI) skips the prompt and leaves them in place.
    $ModulesDir = Join-Path $BlenderBase "$BlenderVersion\scripts\addons\modules"
    if (Test-Path $ModulesDir) {
        $entries = Get-ChildItem -Path $ModulesDir -Force
        if ($entries) {
            Write-Host ""
            Write-Host "Third-party packages in ${ModulesDir}:"
            foreach ($entry in $entries) {
                Write-Host "  $($entry.Name)"
            }
            if (-not [Console]::IsInputRedirected) {
                $answer = Read-Host "Also remove these? [y/N]"
                if ($answer -match '^(y|Y|yes|YES)$') {
                    Remove-Item -Path $ModulesDir -Recurse -Force
                    Write-Host "Removed: $ModulesDir"
                } else {
                    Write-Host "Kept: $ModulesDir"
                }
            } else {
                Write-Host "(non-interactive stdin; left in place)"
            }
        }
    }
    exit 0
}

# The source manifest intentionally has no wheels. Build a private, ABI-specific
# extension directory; do not rewrite the source tree or its manifest.
if (-not $env:PPF_BLENDER_BIN) { throw 'Set PPF_BLENDER_BIN to the target Blender executable.' }
$pythonCandidates = @($env:PPF_BUILD_PYTHON, $env:PPF_PYTHON_BIN, (Join-Path $ScriptDir 'build-win-native\dist\python\python.exe'), 'python', 'python3', 'py')
$pythonBin = $null
foreach ($candidate in $pythonCandidates) {
    if (-not $candidate) { continue }
    try { $v = & $candidate --version 2>&1; if ($LASTEXITCODE -eq 0 -and "$v" -match 'Python \d') { $pythonBin = $candidate; break } } catch {}
}
if (-not $pythonBin) { throw 'A working Python 3 is required; set PPF_BUILD_PYTHON.' }
$stageParent = Join-Path $BlenderBase "$BlenderVersion\extensions\.ppf-addon-builds"
New-Item -ItemType Directory -Force -Path $stageParent | Out-Null
$stageAddon = Join-Path $stageParent ([guid]::NewGuid().ToString())
& $pythonBin (Join-Path $AddonSource 'wheels\build_package.py') --blender $env:PPF_BLENDER_BIN --stage-dir $stageAddon
if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare the ABI-specific extension.' }
$AddonSource = $stageAddon
if (-not (Test-Path $ExtDir)) {
    New-Item -ItemType Directory -Path $ExtDir -Force | Out-Null
}

# Sweep legacy installs unconditionally so the new extension junction
# isn't shadowed by an old scripts/addons copy.
foreach ($legacy in @($LegacyLink, $LegacyHyphenLink)) {
    if (Test-Path $legacy) {
        Remove-Item -Path $legacy -Recurse -Force
        Write-Host "Removed legacy install: $legacy"
    }
}

# Idempotent rebind. Junctions can't be atomically swung like POSIX
# symlinks, so we delete + recreate.
if (Test-Path $ExtLink) {
    $existing = Get-Item -LiteralPath $ExtLink -Force
    if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Existing extension directory: $ExtLink. Remove or upgrade it through Blender first." }
    Remove-Item -LiteralPath $ExtLink -Force
}
cmd /c mklink /J "$ExtLink" "$AddonSource" | Out-Null
Write-Host "Created junction: $ExtLink -> $AddonSource"
Write-Host ""
Write-Host "Enable in Blender with:"
Write-Host "    blender --addons bl_ext.user_default.$AddonName"
