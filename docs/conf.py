import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "LinMulT"
copyright = "2024, Ádám Fodor"
author = "Ádám Fodor"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# ── sphinx-autoapi ─────────────────────────────────────────────────────────────

autoapi_dirs = ["../linmult"]
autoapi_options = [
    "members",
    "undoc-members",  # required — without this autoapi skips submodule pages
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_add_toctree_entry = True
autoapi_member_order = "source"
autoapi_python_class_content = "both"


def skip_undocumented_attributes(app, what, name, obj, skip, options):
    """Hide undocumented attributes (instance variables) from API docs.

    Keeps all classes, functions, methods, and modules even without docstrings,
    but hides bare attributes that have no docstring (e.g. self.dropout_input = ...).
    """
    if what == "attribute" and not obj.docstring:
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", skip_undocumented_attributes)


# ── Napoleon (Google docstrings) ───────────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── MyST (config-reference.md) ────────────────────────────────────────────────

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3  # auto-generate heading anchors for internal #links

# ── Furo theme ─────────────────────────────────────────────────────────────────
# Furo follows OS prefers-color-scheme; dark/light toggle is built in.

html_theme = "furo"
html_logo = "assets/logo.svg"
html_static_path = ["assets"]
