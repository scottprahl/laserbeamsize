SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pylint laserbeamsize/m2.py
	-pydocstyle laserbeamsize/m2.py
	-pylint laserbeamsize/laserbeamsize.py
	-pydocstyle laserbeamsize/laserbeamsize.py
	-pyroma -d .
	-check-manifest

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

rcheck:
	make clean
	make check
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .

.PHONY: clean realclean rcheck html