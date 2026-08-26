"""Sphinx configuration for the SilicaMS documentation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "SilicaMS"
author = "PoreMS and SilicaMS contributors"
copyright = "2019-2026 PoreMS contributors; 2026 SilicaMS contributors"

version_ns = {}
exec(
    (Path(__file__).resolve().parents[1] / "silicams" / "_version.py").read_text(
        encoding="utf-8"
    ),
    version_ns,
)
version = version_ns["__version__"]
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "numpydoc",
]
numpydoc_show_class_members = False

templates_path = []
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = f"SilicaMS {release}"
html_static_path = ["_static"]
html_theme_options = {
    "description": "Periodic amorphous silica models for molecular simulation",
    "github_user": "DrDomenicoMarson",
    "github_repo": "SilicaMS",
    "github_button": True,
}
