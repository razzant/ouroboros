/* Python host only. Bootstrap, locking, updates and lifecycle stay in launcher.py.
 * A real executable in Contents/MacOS, signed with the application's identifier,
 * keeps both the ordinary app's WebKit profile and notification authorization.
 */
#include <Python.h>
#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    char image[PATH_MAX], executable[PATH_MAX], home[PATH_MAX], python[PATH_MAX];
    uint32_t size = sizeof(image);
    if (_NSGetExecutablePath(image, &size) != 0 || realpath(image, executable) == NULL) {
        fputs("Ouroboros: cannot locate the app entry\n", stderr);
        return 1;
    }
    strcpy(image, executable);
    char *slash = strrchr(image, '/');
    if (slash == NULL) return 1;
    *slash = '\0';
    if (snprintf(home, sizeof(home), "%s/../Frameworks/python-standalone", image) >= (int)sizeof(home)
            || snprintf(python, sizeof(python), "%s/bin/python3", home) >= (int)sizeof(python)) {
        fputs("Ouroboros: bundled Python path is too long\n", stderr);
        return 1;
    }

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.write_bytecode = 0;  /* before site imports: never mutate the signed bundle */
    PyStatus status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetBytesString(&config, &config.home, home);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetBytesString(&config, &config.executable, executable);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetBytesString(&config, &config.base_executable, python);
    if (PyStatus_Exception(status)) goto error;
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto error;
    PyConfig_Clear(&config);
    return Py_RunMain();

error:
    PyConfig_Clear(&config);
    Py_ExitStatusException(status);
    return 1;
}
