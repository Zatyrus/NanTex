## Bugs found in the project go here
- [ ] Bug 1



### Bug 1
- **where**: `NanTex_backend/data_preparation/overlay_generation.py`
- **what**: If passed a list of data sources with only one element, numpy raises an inhomognous shape error. Probably due to some numpy operation that expects a 2D array and receives a 1D array.