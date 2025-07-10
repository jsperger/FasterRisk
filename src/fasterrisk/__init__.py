try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("fasterrisk")
    except PackageNotFoundError:
        # package is not installed, e.g., when running tests locally
        __version__ = "0.0.0-localtesting"
except ImportError:
    # Fallback for Python < 3.8 or if importlib.metadata is not available
    # For this project, targeting Python 3.11, this block is less likely to be hit
    # but kept for robustness or if backporting.
    try:
        from pkg_resources import get_distribution, DistributionNotFound
        try:
            __version__ = get_distribution("fasterrisk").version
        except DistributionNotFound:
            # package is not installed
            __version__ = "0.0.0-localtesting-pkgresources"
    except ImportError:
        # pkg_resources not available
        __version__ = "0.0.0-unknown"