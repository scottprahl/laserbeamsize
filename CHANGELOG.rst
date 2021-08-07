Changelog
=================================================

v1.7.3
------
*    create pure python packaging
*    include wheel file
*    package as python3 only

v1.7.2
------
*    allow non-integer beam centers
*    add badges to docs
*    use sphinx-book-theme for docs

v1.7.1
------
*    explicit warning for non-monochrome images in `beam_size()`
*    improve help() messages

v1.7.0
------
*    fix error in identifying major/minor axes in `beam_size_plot()`

v1.6.1
------
*    fix deprecation warning for register_cmap
*    use entire perimeter of background rectangle for tilted background fit
*    fix sphinx and docstyle warnings

v1.6.0
------
*    Add `subtract_tilted_background()`
*    Add M²=1 line to `M2_radius_plot()`
*    try to autoselect line color on images (still imperfect)
*    more documentation tweaks

v1.5.0
------
*    Add M² fitting and plotting
*    rename `visual_report()` to `beam_size_plot()`
*    add `plot_size_montage()`
*    hopefully stable API now
*    allow any colormap, but default to `gist_ncar`
*    extensive documentation of M² fitting process

v1.3.0
------
*    Add another figure to readme
*    Improve `visual_report()`
*    Add `plot_beam_fit()`

v1.2.0
------
*    Add routines to plot values along semi axes
*    Add `visual_report()` for simple beam analysis
*    Fix error when calculating circular radius
*    Add missing scipy requirement
*    Improve README.rst with figure

v1.1.0
------
*    Works dramatically better across a wide range of images
*    Minor API changes to `beam_size()`
*    Use ISO 11146 integration areas
*    Add background routines for corners
*    Add functions for rotations
*    Eliminate old threshold technique
*    Use google docstyle for functions
*    Explain background and integration areas in notebooks
*    Tweak notebooks for clarity

v1.0.2
------
*    use sphinx for documentation
*    revise Jupyter notebooks

v1.0.1
------
*    trivial fix for release.txt
*    improve text
*    remove completed tasks
*    initial commit of 07-M2-Measurement.ipynb
*    bump version

v1.0.0
------
*    first pass at docs
*    General doc improvements
*    add routine to draw default figure
*    handle symmetric case dx=dy better
*    add new definitions and test dx=dy case

v0.2.0
------
*    initial commit
*    ensure float used for sums
*    first public release
