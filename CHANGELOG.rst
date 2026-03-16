Changelog
=========

2.3.3 (2026-03-15)
------------------
  * fix copy-paste bug: ``plot_beam_diagram()`` used ``d_major`` for minor-rectangle dimension
  * fix copy-paste bug: ``plot_image_analysis()`` minor-axis label compared major-axis variables
  * fix ``beam_size()`` crash when ``d_minor`` becomes ``None`` during iterative refinement
  * fix NaN propagation in ``basic_beam_fit()`` when measured diameter is smaller than fitted waist
  * fix divide-by-zero in ``beam_parameter_product()`` and M² error propagation when waist or divergence is zero
  * fix hardcoded figure number in ``m2_display`` that caused matplotlib warnings on repeated calls
  * fix variable shadowing of plane-fit coefficient ``b`` in ``subtract_tilted_background()``
  * replace ``print()`` with ``warnings.warn()`` in ``M2_fit()`` for strict-mode messages
  * replace Python loops with numpy in ``max_index_in_focal_zone()`` and ``min_index_in_outer_zone()``
  * remove dead code: ``basic_beam_size_naive()``, ``rotated_rect_mask_slow()``, ``_mean_filter()``, ``_std_filter()``, ``image_background2()``
  * remove unused ``scipy.ndimage`` import from ``background.py``
  * add comprehensive tests for all fixed issues; pylint score 10.00/10

2.3.2 (2026-03-05)
------------------
  * complete `M2_report()` support for focal length 
  * add tests for major/minor fits
  * add entry to 14-M2-Example.ipynb for new M2_report()

2.3.1 (2026-02-04)
------------------
  * fix regression when units=None
  * fix image display in docs
  * update github workflows
  * default python is 3.12
  * all requirements are in pyproject.toml
  * faster testing with pytest
  * fix errors in .yaml files

2.3.0 (2025-11-30)
------------------
  * Jupyterlite updated to 0.64
  * fix bug with strongly asymmetric rotated beams
  * fix jupyterlite import of npy files
  * fix integration rect drawing when mask_diameters is not the default
  * fix handling for failed d_minor calculation
  * add make lab target
  * add make readme target for images
  * improve handling of small images
  * improve plot_image_analysis
  * improve citations
  * refactored functions in display.py
  * background.py works with images having masks

2.2.0 (2025-11-10)
------------------
  * Jupyterlite support on github pages
  * eliminate circular imports
  * use venv for packaging and testing
  * remove setup.py and setup.sh
  * no longer package test files
  * clean up MANIFEST.IN
  * improve pyproject.toml
  * improve README.rst
  * requirements-dev.txt is now complete

2.1.0 (2025-06-10)
------------------
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

2.0.5 (2023-11-28)
------------------
  * fix mistake in focused_diameter (thanks @rdgraham)

2.0.4 (2023-10-24)
------------------
  * fix zenodo and CITATION.cff
  * fix pypi github action
  * fix copyright
  * fix manifest
  * correct README.rst to use `plot_image_analysis()`
  * tweak badges
  * allow fixed phi angle of 0° (thanks @cptyannick)

2.0.3 (2023-09-16)
------------------
  * readthedocs does not do well with new module names

2.0.2 (2023-09-16)
------------------
  * splitting monolithic m2.py
  * added tests for functions in gaussian.py
  * improved docstrings in gaussian.py
  * improved module docstrings in m2_fit and m2_display

2.0.1 (2023-09-14)
------------------
  * trying to get conda release working

2.0.0 (2023-09-14)
------------------
  * change default background removal to match ISO recommendation
  * split monolithic laserbeamsize module in separate modules
  * rationalized functions names
  * add citation stuff
  * add github actions to test and update citation
  * improve readme so images work on pypi.org
  * add conda-forge support
  * update copyright years
  * more badges

v1.9.4 (2022-03-19)
-------------------
  * allow beam angle to be specified during fitting
  * new notebook to illustrate constrained fits
  * improve docstrings and therefore api documentation
  * still better ellipse and rect outlines
  * start adding examples

v1.9.3 (2022-03-15)
-------------------
  * use faster version of creating rotated rect mask
  * move tests to their own directory
  * avoid deprecated np.float
  * improve drawing of rect and ellipse outlines
  * improve some docstrings

v1.9.2 (2022-03-12)
-------------------
  * use both black and white dashed lines
  * fit to d and not d**2
  * add more dunders to __init__.py
  * fix residual calculation broken in v1.9.1

v1.9.1 (2022-03-10)
-------------------
  * centralize version number to a single place

v1.9.0 (2022-03-09)
-------------------
  * add beam_ellipticity()
  * add beam_parameter_product()
  * rotate x-tick labels when more than 10 ticks
  * removed deprecated use of np.matrix()
  * M2_report now includes BPP values
  * improve API docs
  * code linting

v1.8.0 (2021-12-15)
-------------------
  * handle rotated masks properly
  * fix readthedoc configuration

v1.7.3 (2021-08-07)
-------------------
  * create pure python packaging
  * include wheel file
  * package as python3 only

v1.7.2 (2021-03-22)
-------------------
  * allow non-integer beam centers
  * add badges to docs
  * use sphinx-book-theme for docs

v1.7.1 (2021-01-03)
-------------------
  * explicit warning for non-monochrome images in `beam_size()`
  * improve help() messages

v1.7.0 (2020-11-11)
-------------------
  * fix error in identifying major/minor axes in `beam_size_plot()`

v1.6.1 (2020-09-30)
-------------------
  * fix deprecation warning for register_cmap
  * use entire perimeter of background rectangle for tilted background fit
  * fix sphinx and docstyle warnings

v1.6.0 (2020-08-03)
-------------------
  * Add `subtract_tilted_background()`
  * Add M²=1 line to `M2_radius_plot()`
  * try to autoselect line color on images (still imperfect)
  * more documentation tweaks

v1.5.0 (2020-07-28)
-------------------
  * Add M² fitting and plotting
  * rename `visual_report()` to `beam_size_plot()`
  * add `plot_size_montage()`
  * hopefully stable API now
  * allow any colormap, but default to `gist_ncar`
  * extensive documentation of M² fitting process

v1.3.0 (2020-07-28)
-------------------
  * Add another figure to readme
  * Improve `visual_report()`
  * Add `plot_beam_fit()`

v1.2.0 (2020-06-07)
-------------------
  * Add routines to plot values along semi axes
  * Add `visual_report()` for simple beam analysis
  * Fix error when calculating circular radius
  * Add missing scipy requirement
  * Improve README.rst with figure

v1.1.0 (2020-06-02)
-------------------
  * Works dramatically better across a wide range of images
  * Minor API changes to `beam_size()`
  * Use ISO 11146 integration areas
  * Add background routines for corners
  * Add functions for rotations
  * Eliminate old threshold technique
  * Use google docstyle for functions
  * Explain background and integration areas in notebooks
  * Tweak notebooks for clarity

v1.0.2 (2020-03-29)
-------------------
  * use sphinx for documentation
  * revise Jupyter notebooks

v1.0.1 (2017-11-25)
-------------------
  * trivial fix for release.txt
  * improve text
  * remove completed tasks
  * initial commit of 07-M2-Measurement.ipynb
  * bump version

v1.0.0 (2017-11-25)
-------------------
  * first pass at docs
  * General doc improvements
  * add routine to draw default figure
  * handle symmetric case dx=dy better
  * add new definitions and test dx=dy case

v0.2.0 (2017-11-25)
-------------------
  * initial commit
  * ensure float used for sums
  * first public release
