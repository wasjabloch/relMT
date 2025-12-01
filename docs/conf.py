import importlib.metadata
import collections

project = "relMT"
copyright = "2025, Wasja Bloch"
author = "Wasja Bloch"
version = release = importlib.metadata.version("relMT")

extensions = [
    "myst_parser",  # Markdown parsing
    "sphinx_markdown_tables",
    "sphinx.ext.autodoc",  # Get docs from source code
    "sphinx.ext.intersphinx",  # Cross-link other documentation
    "sphinx.ext.mathjax",  # Write math formulas
    "sphinx.ext.napoleon",  # More documentation styles
    "sphinx.ext.viewcode",  # Show link to actual code
    # Handle type hints. Note: This module may fail disgracefully when building
    # ill-formated documentation. Uncomment and debug Critical, Errors, Warnings
    # when build of documentation is failing
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",  # Copy button for code snippets
]

typehints_use_rtype = False
napoleon_use_rtype = False  # Don't show extra return type
always_use_bars_union = True  # Show multiple types with | bars
always_document_param_types = True

# source_suffix = [".rst", ".md"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "CMakeLists.txt",
]

html_theme = "pydata_sphinx_theme"
html_permalinks_icon = "<span>#</span>"
html_theme_options = dict(
    sidebarwidth="22em",
    pygments_light_style="blinds-light",
    pygments_dark_style="blinds-dark",
)
html_logo = "_static/relMT-alpha.png"

autodoc_member_order = "bysource"  # groupwise, alphabetical

myst_enable_extensions = [
    "colon_fence",  # Nicer tables
    "dollarmath",  # Nicer tables
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "obspy": ("https://docs.obspy.org/", None),
    "matplotlib": (" https://matplotlib.org/", None),
    # To include other projects, provide path to objects.inv file
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),  # Ignore these interlinked attributes
]


# Remove 'Alias for field number' docstring from named tuple
def remove_namedtuple_attrib_docstring(app, what, name, obj, skip, options):
    if type(obj) is collections._tuplegetter:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", remove_namedtuple_attrib_docstring)
