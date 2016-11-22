#include <Python.h>
#include <numpy/arrayobject.h>
#include "cshape.h"

static char module_docstring[] = "";
static char surface_docstring[] = "";

static PyObject *cshape_calculate_surfacearea(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
	//{"calculate_", cmatrices_, METH_VARARGS, _docstring},
	{ "calculate_surfacearea", cshape_calculate_surfacearea, METH_VARARGS, surface_docstring },
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
init_cshape(void) {
	PyObject *m = Py_InitModule3("_cshape", module_methods, module_docstring);
	if (m == NULL)
		return;

	// Initialize numpy functionality
	import_array();
}

static PyObject *cshape_calculate_surfacearea(PyObject *self, PyObject *args)
{
	PyObject *mask_obj, *spacing_obj;
	PyArrayObject *mask_arr, *spacing_arr;
	int size[3];
	char *mask;
	double *spacing;
	double SA;

	// Parse the input tuple
	if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj))
		return NULL;

	// Interpret the input as numpy arrays
	mask_arr = (PyArrayObject *)PyArray_FROM_OTF(mask_obj, NPY_BYTE, NPY_FORCECAST | NPY_UPDATEIFCOPY | NPY_IN_ARRAY);
	spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(spacing_obj, NPY_DOUBLE, NPY_FORCECAST | NPY_UPDATEIFCOPY | NPY_IN_ARRAY);

	if (mask_arr == NULL || spacing_arr == NULL)
	{
		Py_XDECREF(mask_arr);
		Py_XDECREF(spacing_arr);
		return NULL;
	}

	// Check if Image and Mask have 3 dimensions
	if (PyArray_NDIM(mask_arr) != 3)
	{
		Py_DECREF(mask_arr);
		Py_DECREF(spacing_arr);
		PyErr_SetString(PyExc_RuntimeError, "Expected a 3D array for mask.");
		return NULL;
	}

	// Get sizes of the arrays
	size[2] = (int)PyArray_DIM(mask_arr, 2);
	size[1] = (int)PyArray_DIM(mask_arr, 1);
	size[0] = (int)PyArray_DIM(mask_arr, 0);

	// Get arrays in Ctype
	mask = (char *)PyArray_DATA(mask_arr);
	spacing = (double *)PyArray_DATA(spacing_arr);

	//Calculate GLCM
	SA = calculate_surfacearea(mask, size, spacing);

	// Clean up
	Py_DECREF(mask_arr);
	Py_DECREF(spacing_arr);

	if (SA < 0)
	{
		PyErr_SetString(PyExc_RuntimeError, "Calculation of GLCM Failed.");
		return NULL;
	}

	return Py_BuildValue("f", SA);
}