pyfisheye - a small Python wrapper around the opencv (>=3) fisheye model
=======================================================================


opencv 3 introduces a fisheye calibration model. Currently the python
bindings of this model are buggy. pyfisheye is a small wrapper for the
python bindings that is both a workaround for the bug and a syntactic
sugar to simplify the use of the bindings.

Requirements
------------
* Tested on python 3.4 (though should work with version 2.6 and higher).
* numpy
* opencv > 3

Installation
------------
Download the source files of pyfisheye.
Then, execute 'python setup.py install' from the root folder.

Basic Usage
-----------
See the example in the folder 'example'.

License
-------
See included license file.
The repository includes images downloaded from opencv extra repositroy (https://github.com/Itseez/opencv_extra). These images belong to their rightful owners and are included here only for assisting the testing of this code.

Thank-you to the people at <http://wingware.com/> for their policy of **free licenses for non-commercial open source developers**.

![wingware-logo](http://wingware.com/images/wingware-logo-180x58.png)
