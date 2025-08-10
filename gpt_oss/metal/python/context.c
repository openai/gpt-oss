#include <Python.h>

#include <gpt-oss.h>

#include "module.h"


static int PyGPTOSSContext_init(PyGPTOSSContext* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"model", "context_length", NULL};
    PyObject* model = NULL;
    Py_ssize_t context_length = 0; // Default to 0 if None

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$i", kwlist,
                                     &model, &context_length)) {
        return -1;
    }
    if (!PyObject_TypeCheck(model, &PyGPTOSSModel_Type)) {
        PyErr_SetString(PyExc_TypeError, "model must be an gptoss.Model object");
        return -1;
    }
    if (context_length < 0) {
        PyErr_SetString(PyExc_ValueError, "context_length must be a positive integer");
        return -1;
    }

    enum gptoss_status status = gptoss_context_create(
        ((const PyGPTOSSModel*) model)->handle,
        (size_t) context_length,
        &self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        goto error;
    }

    return 0;

error:
    gptoss_context_release(self->handle);
    self->handle = NULL;
    return -1;
}

static void PyGPTOSSContext_dealloc(PyGPTOSSContext* self) {
    (void) gptoss_context_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject*) self);
}

static PyObject* PyGPTOSSContext_copy(PyGPTOSSContext *self) {
    PyGPTOSSContext* copy = (PyGPTOSSContext*) PyObject_New(PyGPTOSSContext, Py_TYPE(self));
    if (copy == NULL) {
        return NULL;
    }

    (void) gptoss_context_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject*) copy;
}

static PyObject* PyGPTOSSContext_append(PyGPTOSSContext* self, PyObject* arg) {
    if (PyBytes_Check(arg)) {
        char* string_ptr = NULL;
        Py_ssize_t string_size = 0;
        if (PyBytes_AsStringAndSize(arg, &string_ptr, &string_size) < 0) {
            return NULL;
        }

        const enum gptoss_status status = gptoss_context_append_chars(
            self->handle, string_ptr, string_size, /*num_tokens_out=*/NULL);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else if (PyUnicode_Check(arg)) {
        Py_ssize_t string_size = 0;
        const char* string_ptr = PyUnicode_AsUTF8AndSize(arg, &string_size);
        if (string_ptr == NULL) {
            return NULL;
        }

        const enum gptoss_status status = gptoss_context_append_chars(
            self->handle, string_ptr, string_size, /*num_tokens_out=*/NULL);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else if (PyLong_Check(arg)) {
        // Handle potential overflow from Python int to unsigned long
        const unsigned long token_as_ulong = PyLong_AsUnsignedLong(arg);
        if (token_as_ulong == (unsigned long) -1 && PyErr_Occurred()) {
            return NULL;  // PyLong_AsUnsignedLong sets appropriate overflow error
        }

        // Check for overflow when converting to uint32_t
        if (token_as_ulong > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "Token value too large for uint32_t");
            return NULL;
        }
        const uint32_t token = (uint32_t) token_as_ulong;
        const enum gptoss_status status = gptoss_context_append_tokens(
            self->handle, /*num_tokens=*/1, &token);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else {
        PyErr_SetString(PyExc_TypeError, "expected a bytes or integer argument");
        return NULL;
    }
}

static PyObject* PyGPTOSSContext_process(PyGPTOSSContext* self) {
    const enum gptoss_status status = gptoss_context_process(self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* PyGPTOSSContext_sample(PyGPTOSSContext* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"temperature", "seed", NULL};

    unsigned long long seed = 0;
    float temperature = 1.0f;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$fK", kwlist,
            &temperature, &seed))
    {
        return NULL;
    }

    // Initialize to max value to detect potential initialization issues
    uint32_t token_out = UINT32_MAX;
    
    // Note: seed is uint64_t, no overflow check needed as it's already the right size
    // temperature is float, handled by Python's float parsing
    enum gptoss_status status = gptoss_context_sample(
        self->handle, temperature, (uint64_t) seed, &token_out);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromUnsignedLong((unsigned long) token_out);
}

static PyObject* PyGPTOSSContext_reset(PyGPTOSSContext* self) {
    const enum gptoss_status status = gptoss_context_reset(self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef PyGPTOSSContext_methods[] = {
    {"__copy__", (PyCFunction) PyGPTOSSContext_copy, METH_NOARGS, "Create a copy of the Context"},
    {"append", (PyCFunction) PyGPTOSSContext_append, METH_O, "Append bytes to the Context"},
    {"process", (PyCFunction) PyGPTOSSContext_process, METH_NOARGS, "Process tokens in the Context"},
    {"sample", (PyCFunction) PyGPTOSSContext_sample, METH_VARARGS | METH_KEYWORDS, "Sample token prediction from the Context"},
    {"reset", (PyCFunction) PyGPTOSSContext_reset, METH_NOARGS, "Discard the content of the Context"},
    {NULL},
};

static PyObject* PyGPTOSSContext_get_num_tokens(PyGPTOSSContext* self, void* closure) {
    size_t num_tokens = 0;
    const enum gptoss_status status = gptoss_context_get_num_tokens(self->handle, &num_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(num_tokens);
}

static PyObject* PyGPTOSSContext_get_max_tokens(PyGPTOSSContext* self, void* closure) {
    size_t max_tokens = 0;
    const enum gptoss_status status = gptoss_context_get_max_tokens(self->handle, &max_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(max_tokens);
}

static PyObject* PyGPTOSSContext_get_tokens(PyGPTOSSContext* self, void* closure) {
    PyObject* token_list_obj = NULL;
    PyObject* token_obj = NULL;
    uint32_t* token_ptr = NULL;
    size_t num_tokens = 0;

    // Get number of tokens with proper error handling
    enum gptoss_status status = gptoss_context_get_num_tokens(self->handle, &num_tokens);
    if (status != gptoss_status_success) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get number of tokens");
        return NULL;
    }

    // Early return if no tokens
    if (num_tokens == 0) {
        return PyList_New(0);
    }

    // Check for potential multiplication overflow in allocation size
    if (num_tokens > SIZE_MAX / sizeof(uint32_t)) {
        PyErr_SetString(PyExc_OverflowError, "Token count too large for memory allocation");
        goto error;
    }

    // Allocate memory for tokens
    token_ptr = (uint32_t*) PyMem_Malloc(num_tokens * sizeof(uint32_t));
    if (token_ptr == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    // Verify num_tokens doesn't exceed Py_ssize_t max for Python list
    if (num_tokens > (size_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Token count exceeds maximum Python list size");
        goto error;
    }

        status = gptoss_context_get_tokens(self->handle, token_ptr, /*max_tokens=*/num_tokens, &num_tokens);
        if (status != gptoss_status_success) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get tokens");
            goto error;
        }
    }

    token_list_obj = PyList_New((Py_ssize_t) num_tokens);
    if (token_list_obj == NULL) {
        goto error;
    }

    for (size_t t = 0; t < num_tokens; t++) {
        // uint32_t to unsigned long conversion is safe on all platforms
        // as unsigned long is at least 32 bits
        token_obj = PyLong_FromUnsignedLong((unsigned long) token_ptr[t]);
        if (token_obj == NULL) {
            goto error;  // PyLong_FromUnsignedLong sets memory error if allocation fails
        }
        if (PyList_SetItem(token_list_obj, (Py_ssize_t) t, token_obj) < 0) {
            goto error;
        }
        token_obj = NULL;  // PyList_SetItem stole the reference
    }

    PyMem_Free(token_ptr);
    return token_list_obj;

error:
    PyMem_Free(token_ptr);
    Py_XDECREF(token_obj);
    Py_XDECREF(token_list_obj);
    return NULL;
}

static PyGetSetDef PyGPTOSSContext_getseters[] = {
    (PyGetSetDef) {
        .name = "num_tokens",
        .get = (getter) PyGPTOSSContext_get_num_tokens,
        .doc = "Current number of tokens in the context",
    },
    (PyGetSetDef) {
        .name = "max_tokens",
        .get = (getter) PyGPTOSSContext_get_max_tokens,
        .doc = "Maximum number of tokens in the context",
    },
    (PyGetSetDef) {
        .name = "tokens",
        .get = (getter) PyGPTOSSContext_get_tokens,
        .doc = "List of token IDs in the context",
    },
    {NULL}  /* Sentinel */
};

PyTypeObject PyGPTOSSContext_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gptoss.Context",
    .tp_basicsize = sizeof(PyGPTOSSContext),
    .tp_flags = 0
        | Py_TPFLAGS_DEFAULT
        | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Context object",
    .tp_methods = PyGPTOSSContext_methods,
    .tp_getset = PyGPTOSSContext_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PyGPTOSSContext_init,
    .tp_dealloc = (destructor) PyGPTOSSContext_dealloc,
};
