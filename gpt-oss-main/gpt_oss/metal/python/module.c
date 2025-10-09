#include <Python.h>

#include "module.h"


static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

static PyModuleDef metal_module = {
    PyModuleDef_HEAD_INIT,
    "_metal",
    "Local GPT-OSS inference",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__metal(void) {
    PyObject* module = NULL;
    PyObject* model_type = NULL;
    PyObject* tokenizer_type = NULL;
    PyObject* context_type = NULL;

    if (PyType_Ready(&PyGPTOSSModel_Type) < 0) {
        goto error;
    }
    model_type = (PyObject*) &PyGPTOSSModel_Type;
    Py_INCREF(model_type);

    if (PyType_Ready(&PyGPTOSSTokenizer_Type) < 0) {
        goto error;
    }
    tokenizer_type = (PyObject*) &PyGPTOSSTokenizer_Type;
    Py_INCREF(tokenizer_type);

    if (PyType_Ready(&PyGPTOSSContext_Type) < 0) {
        goto error;
    }
    context_type = (PyObject*) &PyGPTOSSContext_Type;
    Py_INCREF(context_type);

    module = PyModule_Create(&metal_module);
    if (module == NULL) {
        goto error;
    }

    if (PyModule_AddObject(module, "Model", model_type) < 0) {
        goto error;
    }

    if (PyModule_AddObject(module, "Tokenizer", tokenizer_type) < 0) {
        goto error;
    }

    if (PyModule_AddObject(module, "Context", context_type) < 0) {
        goto error;
    }

    return module;

error:
    Py_XDECREF(context_type);
    Py_XDECREF(tokenizer_type);
    Py_XDECREF(model_type);
    Py_XDECREF(module);
    return NULL;
}
