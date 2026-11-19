# Build the Blender extension from source

The source tree intentionally contains no wheel binaries. Use Python 3 and a
Blender executable to create a package for the target Blender Python ABI:

    python blender_addon/wheels/build_package.py --abi cp311 --blender /path/to/blender --output ppf-cp311.zip
    python blender_addon/wheels/build_package.py --abi cp313 --blender /path/to/blender --output ppf-cp313.zip

Choose cp311 for Blender 4.2 and cp313 for Blender 5.1. For another version,
check the target Blender's `sys.version_info`; unsupported ABIs are rejected.
The tool stages the addon separately, fetches the five unmodified official
cbor2 6.0.1 wheels for that ABI, inserts them in the staged manifest, and asks
Blender to build and validate the ZIP. The original wheel metadata and license
remain in the wheels. Network access to files.pythonhosted.org is needed.
The source manifest is not an installable distribution by itself.

To build one ZIP for both supported ABIs, use:

    python blender_addon/wheels/build_package.py --abi universal --blender /path/to/blender --output ppf-universal.zip

The universal build stages the official wheel contents privately by ABI and
platform, including their licenses. The matching dependency is loaded at startup;
it is not listed in the Blender manifest wheel array.

On Windows the selected cbor2 package is copied once to Blender's extension
user directory before loading, so an enabled extension can be updated without
locking the installed ZIP contents. This cache is separated by dependency
version, Python ABI and platform.
