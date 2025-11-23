<!-- filepath: /Users/lennart/Research/software/r_ns_2025/README.rst -->

.. |arxiv-badge| image:: https://img.shields.io/badge/arXiv-TBD-b31b1b.svg
   :target: TBD
   :alt: Paper

.. |streamlit-badge| image:: https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit
   :target: TBD
   :alt: Streamlit App

.. _arxiv-url: TBD
.. _streamlit-url: TBD
.. _paper-url: TBD

Inflation in 2025: Constraints on :math:`r` and :math:`n_s` using the latest CMB and BAO data
=======================================================================================================

|arxiv-badge| |streamlit-badge|

This repository accompanies `our paper <paper-url_>`_, which presents constraints on the tensor-to-scalar ratio :math:`r` and the scalar spectral index :math:`n_s` based on the latest cosmic microwave background (CMB) and baryon acoustic oscillation (BAO) data available in 2025.
Here, you can find the data products of the Monte Carlo Markov Chain analysis performed in the paper as well as plotting scripts that allow you to reproduce the paper plots and make your own modified versions.

Making :math:`r` - :math:`n_s` Plots
=====================================

To make your own :math:`r` - :math:`n_s` plots you can use the **online plotting app** `available here <streamlit-url_>`_ (no installation required!).

The app allows you to:

- Toggle between different data constraints and forecasts
- Add your own forecast contours
- Show theoretical predictions (e.g. polynomial potentials, Higgs inflation)
- Add your own theoretical prediction
- Adjust plot appearance (aspect ratio, axis limits, log scale)
- **Export plots as PDF/PNG** for your publications
- **Export Python code** to tweak the plot further locally

If you want to make or adjust plots locally, clone the git repository.
You then have access to two scripts:

- **r_ns_plot.ipynb:**
  A jupyter notebook that recreates the two paper plots. It also features a simpler version of the data plot in the paper as a starting point for modifications.
  
- **r_ns_plot.py:**
  A light python script that creates a simpler version of the data plot in the paper as a starting point for modifications. 

Note that these scripts come with the following requirements:

- Python 3.8+
- numpy
- matplotlib
- scipy
- seaborn
- getdist
- jupyter (to run the notebook)

MCMC products
=============

The ``chains/`` folder contains MCMC chains and best-fit points obtained using `Cobaya <https://github.com/CobayaSampler/cobaya>`_ and `CLASS <http://class-code.net/>`_ for :math:`\Lambda\mathrm{CDM}\!+\!r`:

**SPA_BK/:**

- Primary CMB: Planck PR3 (no low-E), SPT-3G D1, ACT DR6, BK18
- CMB Lensing: Planck PR4, SPT-3G MUSE, ACT DR6
- Planck-based :math:`\tau_{\rm reio}` prior

**SPA_BK_DESI/:**

- Primary CMB: Planck PR3 (no low-E), SPT-3G D1, ACT DR6, BK18
- CMB Lensing: Planck PR4, SPT-3G MUSE, ACT DR6
- Planck-based :math:`\tau_{\rm reio}` prior
- BAO: DESI DR2

Please see the paper for more details. All chains have had burn-in removed. Minimum files are also provided.

Repository Structure
====================

.. code-block:: text

    r_ns_2025/
    ├── streamlit_app.py          # Online plotting app
    ├── r_ns_plot.ipynb           # Jupyter notebook of paper plots
    ├── theory_models.py          # Inflation model predictions
    ├── plot_style.py             # Matplotlib styling and configuration
    ├── legend_utils.py           # Custom legend handlers
    └── chains/                   # MCMC chains
        ├── SPA_BK/
        └── SPA_BK_DESI/

Citing this work
================

If you use this code or the provided data constraints in your research, please cite `the release paper <paper-url_>`_.
If you use the online plotting app, please include a `link to it <streamlit-url_>`_ as a footnote.
When showing data constraints in your plots, please also cite the associated paper.

Acknowledgments
===============

The plotting code is based on the `BICEP/Keck 2018 plotting script <http://bicepkeck.org/bk18_2021_release.html>`_ (`arXiv:2110.00483 <https://arxiv.org/abs/2110.00483>`_) and uses `GetDist <https://github.com/cmbant/getdist>`_ (`arXiv:1910.13970 <https://arxiv.org/abs/1910.13970>`_).

----

.. |cnrs| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/cnrs_logo.jpeg
   :alt: CNRS
   :height: 100px
   :width: 100px

.. |erc| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/erc_logo.jpeg
   :alt: ERC
   :height: 100px
   :width: 100px

.. |NEUCosmoS| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/neucosmos_logo.png
   :alt: NEUCosmoS
   :height: 100px
   :width: 159px

.. |IAP| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/IAP_logo.png
   :alt: IAP
   :height: 100px
   :width: 149px

.. |Sorbonne| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/sorbonne_logo.jpeg
   :alt: Sorbonne
   :height: 100px
   :width: 248px

|cnrs| |erc| |NEUCosmoS| |IAP| |Sorbonne|