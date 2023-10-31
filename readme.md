
# Xarray-Healpy

# Xarray-Healpy

**Xarray-Healpy** is designed to convert georeferenced data expressed in latitude and longitude into a Healpix grid (https://healpix.sourceforge.io) and make use of the array indexing system provided by **Xarray** (http://xarray.pydata.org).

The development of Xarray-Healpy was initiated to meet the specific requirements of oceanography studies, which involve the analysis of geospatial data with varying precisions. This tool enables the performance of computations such as convolution while considering land masks through the utilization of a Hierarchical Equal Area Grid.

Given the particular characteristics of the Hierarchical Equal Area Grid, our aim is to provide solutions for Travel Time Analysis (like, H3 Travel Times - https://observablehq.com/@nrabinowitz/h3-travel-times), taking into account land masks and oceanic physical properties using Xarray-Healpy, with the goal of improving the tracking of fish habitats.

