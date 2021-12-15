SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

pycheck:
	-pylint laserbeamsize/m2.py
	-pydocstyle laserbeamsize/m2.py
	-pylint laserbeamsize/laserbeamsize.py
	-pydocstyle laserbeamsize/laserbeamsize.py
	-pylint laserbeamsize/__init__.py
	-pydocstyle laserbeamsize/__init__.py

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodule docs/laserbeamsize.rst

notecheck:
	make clean
	pytest --verbose test_all_notebooks.py

rcheck:
	make notecheck
	make pycheck
	make rstcheck
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .

clean:
	rm -rf dist
	rm -rf laserbeamsize.egg-info
	rm -rf laserbeamsize/__pycache__
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	rm -rf build

realclean:
	make clean


.PHONY: clean realclean rcheck html notecheck pycheck rstcheck