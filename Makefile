SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pylint laserbeamsize/m2.py
	-pydocstyle laserbeamsize/m2.py
	-pylint laserbeamsize/laserbeamsize.py
	-pydocstyle laserbeamsize/laserbeamsize.py
	-pylint laserbeamsize/__init__.py
	-pydocstyle laserbeamsize/__init__.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf dist
	rm -rf laserbeamsize.egg-info
	rm -rf laserbeamsize/__pycache__
	rm -rf docs/_build/*
	rm -rf docs/_build/.buildinfo
	rm -rf docs/_build/.doctrees/
	rm -rf docs/api/*

realclean:
	make clean

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodule docs/laserbeamsize.rst

rcheck:
	make clean
	make check
	make rstcheck
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .

.PHONY: clean realclean rcheck html