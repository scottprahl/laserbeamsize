SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	pyroma -d .
	check-manifest

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf dist
	rm -rf laserbeamsize.egg-info
	rm -rf laserbeamsize/__pycache__
	rm -rf docs/_build/*
	rm -rf docs/api/*
	rm -rf .tox
	
.PHONY: clean realclean check