# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DynCluster'
copyright = '2025, Yingxiao Wang'
author = 'Yingxiao Wang'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# -- Options for LaTeX output -------------------------------------------------

# 使用 XeLaTeX 引擎以支持中文
latex_engine = 'xelatex'

latex_elements = {
    # 让 chapter 从任意一页开始（取消默认的“右开”空白页）
    'classoptions': ',openany,oneside',

    # 在导言区加载 ctex 宏包以支持中文
    'preamble': r'''
\usepackage{ctex}
''',
}