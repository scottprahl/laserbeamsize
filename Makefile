HTMLOPTS    ?=
PDFOPTS     ?= 
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

test:
	pytest tests

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(HTMLOPTS)
	open docs/_build/index.html

pdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)"  $(PDFOPTS)

pycheck:
	-pylint laserbeamsize/m2.py
	-pylint laserbeamsize/laserbeamsize.py
	-pylint laserbeamsize/__init__.py

doccheck:
	-pydocstyle laserbeamsize/m2.py
	-pydocstyle laserbeamsize/laserbeamsize.py
	-pydocstyle laserbeamsize/__init__.py

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodapi docs/laserbeamsize.rst

rcheck:
	make clean
	make test
	make pycheck
	make doccheck
	make rstcheck
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .

clean:
	rm -rf __pycache__
	rm -rf dist
	rm -rf laserbeamsize.egg-info
	rm -rf laserbeamsize/__pycache__
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf docs/.ipynb_checkpoints
	rm -rf tests/__pycache__
	rm -rf build
	rm -rf .eggs
	rm -rf .pytest_cache


.PHONY: clean rcheck html notecheck pycheck doccheck test rstcheck