from importlib.metadata import PackageNotFoundError, version

# Single source of truth is pyproject.toml; setuptools records it in the
# installed package metadata. The release workflow also asserts the tag
# matches it. Falls back when run from an uninstalled source tree.
try:
    __version__ = version("voicepipe-engine")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
