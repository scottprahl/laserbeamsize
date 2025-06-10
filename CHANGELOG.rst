Changelog
=========

2.1.0
-----
* make fixed angle work much better (thanks @przemororo)
* use major axis (d_major) instead of dx (near x-axis)
* trim distribution down to core files
* eliminate dependence on pillow and scikit-image
* correct angle definitions
* use black for all formatting
* adopt ``pyproject.toml`` with PEP 621 metadata and eliminate setup.cfg
* lint YAML files and run ``ruff`` during ``make rcheck``
* allow ``iso_noise`` option in display and background functions
* rename ``tests/test_all_notebooks.py`` to ``tests/all_test_notebooks.py``
* replace ``master`` branch references with ``main`` in docs
* numerous style and docstring improvements
* remove flake8
* bump minimum python version to 3.9
* eliminate non-standard use of semi-major and semi-minor

2.0.5
-----
* fix mistake in focused_diameter (thanks @rdgraham)

2.0.4
-----
* fix zenodo and CITATION.cff
* fix pypi github action
* fix copyright
* fix manifest
* correct README.rst to use `plot_image_analysis()`
* tweak badges
* allow fixed phi angle of 0° (thanks @cptyannick)

2.0.3
-----
* readthedocs does not do well with new module names

2.0.2
-----
* splitting monolithic m2.py
* added tests for functions in gaussian.py
* improved docstrings in gaussian.py
* improved module docstrings in m2_fit and m2_display

2.0.1
-----
* trying to get conda release working

2.0.0
-----
* change default background removal to match ISO recommendation
* split monolithic laserbeamsize module in separate modules
* rationalized functions names
* add citation stuff
* add github actions to test and update citation
* improve readme so images work on pypi.org
* add conda-forge support
* update copyright years
* more badges

v1.9.4
------
* allow beam angle to be specified during fitting
* new notebook to illustrate constrained fits
* improve docstrings and therefore api documentation
* still better ellipse and rect outlines
* start adding examples

v1.9.3
------
* use faster version of creating rotated rect mask
* move tests to their own directory
* avoid deprecated np.float
* improve drawing of rect and ellipse outlines
* improve some docstrings

v1.9.2
------
* use both black and white dashed lines
* fit to d and not d**2
* add more dunders to __init__.py
* fix residual calculation broken in v1.9.1

v1.9.1
------
* centralize version number to a single place

v1.9.0
------
* add beam_ellipticity()
* add beam_parameter_product()
* rotate x-tick labels when more than 10 ticks
* removed deprecated use of np.matrix()
* M2_report now includes BPP values
* improve API docs
* code linting

v1.8.0
------
* handle rotated masks properly
* fix readthedoc configuration

v1.7.3
------
* create pure python packaging
* include wheel file
* package as python3 only

v1.7.2
------
* allow non-integer beam centers
* add badges to docs
* use sphinx-book-theme for docs

v1.7.1
------
* explicit warning for non-monochrome images in `beam_size()`
* improve help() messages

v1.7.0
------
* fix error in identifying major/minor axes in `beam_size_plot()`

v1.6.1
------
* fix deprecation warning for register_cmap
* use entire perimeter of background rectangle for tilted background fit
* fix sphinx and docstyle warnings

v1.6.0
------
* Add `subtract_tilted_background()`
* Add M²=1 line to `M2_radius_plot()`
* try to autoselect line color on images (still imperfect)
* more documentation tweaks

v1.5.0
------
* Add M² fitting and plotting
* rename `visual_report()` to `beam_size_plot()`
* add `plot_size_montage()`
* hopefully stable API now
* allow any colormap, but default to `gist_ncar`
* extensive documentation of M² fitting process

v1.3.0
------
* Add another figure to readme
* Improve `visual_report()`
* Add `plot_beam_fit()`

v1.2.0
------
* Add routines to plot values along semi axes
* Add `visual_report()` for simple beam analysis
* Fix error when calculating circular radius
* Add missing scipy requirement
* Improve README.rst with figure

v1.1.0
------
* Works dramatically better across a wide range of images
* Minor API changes to `beam_size()`
* Use ISO 11146 integration areas
* Add background routines for corners
* Add functions for rotations
* Eliminate old threshold technique
* Use google docstyle for functions
* Explain background and integration areas in notebooks
* Tweak notebooks for clarity

v1.0.2
------
* use sphinx for documentation
* revise Jupyter notebooks

v1.0.1
------
* trivial fix for release.txt
* improve text
* remove completed tasks
* initial commit of 07-M2-Measurement.ipynb
* bump version

v1.0.0
------
* first pass at docs
* General doc improvements
* add routine to draw default figure
* handle symmetric case dx=dy better
* add new definitions and test dx=dy case

v0.2.0
------
* initial commit
* ensure float used for sums
* first public release
