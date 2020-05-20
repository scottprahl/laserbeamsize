SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pyroma -d .
	-check-manifest
	-pylint laserbeamsize/laserbeamsize.py
	-pep257 laserbeamsize/laserbeamsize.py

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
	rm -rf .tox

rcheck:
	make clean
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .
#	tox

.PHONY: clean realclean check