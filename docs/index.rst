.. pyradiomics documentation master file, created by
   sphinx-quickstart on Tue Jun 28 13:27:28 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyradiomics documentation!
=====================================

This is an open-source python package for the extraction of Radiomics features from 2D and 3D images and
binary masks.

Image loading and preprocessing (e.g. resampling and cropping) are first done using ``SimpleITK``.
Then, loaded data are converted into numpy arrays for further calculation using feature classes
outlined below.

Feature Classes
---------------

Currently supports the following feature classes:

* First Order Statistics
* Shape-based
* `Gray Level Cooccurence Matrix <https://en.wikipedia.org/wiki/Co-occurrence_matrix>`_ (GLCM)
* `Gray Level Run Length Matrix <http://www.insight-journal.org/browse/publication/231>`_ (GLRLM)
* `Gray Level Size Zone Matrix <https://en.wikipedia.org/wiki/Gray_level_size_zone_matrix>`_ (GLSZM)
* Neighbouring Gray Tone Difference Matrix (NGTDM)

Filter Classes
--------------

Aside from the feature classes, there are also some built-in optional filters:

* Laplacian of Gaussian (LoG, based on SimpleITK functionality)
* Wavelet (using the PyWavelets package)
* Square
* Square Root
* Logarithm
* Exponential

Most of the provided features and filters are based on methods described in the following publications:

HJWL Aerts, ER Velazquez, RTH Leijenaar, et al., "Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach", vol. 5, Nat Communication, 2014. Available `here <http://www.nature.com/ncomms/2014/140603/ncomms5006/full/ncomms5006.html>`_.

Specifically, the formulation of the individual feature calculation is covered in this `supplement <http://www.nature.com/ncomms/2014/140603/ncomms5006/extref/ncomms5006-s1.pdf>`_


Supporting reproducible extraction
----------------------------------

Aside from calculating features, the pyradiomics package includes provenance information in the
output. This information contains information on used image and mask, as well as applied settings
and filters, thereby enabling fully reproducible feature extraction.

Citation
--------

If you publish any work which uses this package, please cite the following publication:

Joost J.M. van Griethuysen et al, “Computational Radiomics System to Decode the Radiographic Phenotype”; Submitted

3rd-party packages used in pyradiomics
--------------------------------------

* SimpleITK
* numpy
* PyWavelets (Wavelet filter)
* pykwalify (Enabling yaml parameters file checking)
* tqdm (Progressbar)
* sphinx (Generating documentation)
* sphinx_rtd_theme (Template for documentation)
* nose-parameterized (Testing)

Installation
------------

* Clone the repository

  * ``git clone git://github.com/Radiomics/pyradiomics``

* Install on your system, with prerequisites:

  * ``cd pyradiomics``
  * ``sudo python -m pip install -r requirements.txt``
  * ``sudo python setup.py install``

* For more detailed installation instructions see :ref:`Installation Details<installation-label>`

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   installation
   usage
   radiomics

Pyradiomics Indices and Tables
------------------------------

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`

License
-------

This package is covered by the 3D Slicer License.

**This work was supported in part by the US National Cancer Institute grant
5U24CA194354, QUANTITATIVE RADIOMICS SYSTEM DECODING THE TUMOR PHENOTYPE.**

