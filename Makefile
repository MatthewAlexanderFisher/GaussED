# This Makefile is used to manage the installation and development of the pcg-stein package.
.PHONY: \
	install install-e install-dev-e install-docs-e install-all-e \
	docs-html docs-serve docs-open docs-all docs-make

# Installation commands (-e for editable install)
install-dev-e:
	pip install -e .[dev]

install-docs-e:
	pip install -e .[docs]

install-all-e:
	pip install -e .[dev,docs]

install:
	pip install .

install-e:
	pip install -e .

# Format code using Black
format:
	black src/ experiments/

# Build HTML documentation
docs-html:
	sphinx-build -b html docs docs/_build/html

# Generate API documentation using sphinx-apidoc
docs-make:
	sphinx-apidoc -f -o docs src/pcg_stein --implicit-namespaces

# Serve the built docs locally at http://localhost:8000
docs-serve:
	python -m http.server --directory docs/_build/html 8000

# Build and open docs in browser 
docs-open: docs-html
	python -m webbrowser http://localhost:8000

# Live previewing using sphinx-autobuild
docs-live:
	sphinx-autobuild docs docs/_build/html --open-browser

# Full doc workflow: build + serve + open
docs-all: docs-make docs-html docs-serve

