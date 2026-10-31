# Sourced by the packaged Ouroboros entry BEFORE the frozen loader and GI hooks.
# bundle_linux_gtk.py relocates WebKit's compiled /usr paths to ././ paths.
# Keep unset distinct from empty, including LD_LIBRARY_PATH_ORIG (the loader
# overwrites that too). Values are expanded only after eval parses the fixed
# variable names: quotes, newlines and shell syntax in caller values stay data.
# desktop_media owns the matching list and restores these in child env copies.
if [ "${OUROBOROS_DESKTOP_ENV_SAVED:-}" != 1 ]; then
    for key in LD_LIBRARY_PATH LD_LIBRARY_PATH_ORIG XDG_DATA_DIRS PYWEBVIEW_GUI \
        GI_TYPELIB_PATH GIO_MODULE_DIR GIO_EXTRA_MODULES GSETTINGS_SCHEMA_DIR \
        GTK_DATA_PREFIX GTK_EXE_PREFIX GTK_PATH GTK_IM_MODULE_FILE \
        GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR \
        GST_PLUGIN_SYSTEM_PATH_1_0 GST_PLUGIN_SCANNER PANGO_LIBDIR PANGO_SYSCONFDIR \
        GST_PLUGIN_PATH GST_PLUGIN_SYSTEM_PATH GST_REGISTRY GST_REGISTRY_FORK; do
        eval 'export OUROBOROS_DESKTOP_HOST_'"$key"'="${'"$key"'+1}${'"$key"'-}"'
    done
    export OUROBOROS_DESKTOP_ENV_SAVED=1
fi
GTK_ROOT="$OUROBOROS_BUNDLE_DIR/linux-gtk"
export LD_LIBRARY_PATH="$OUROBOROS_BUNDLE_DIR:$GTK_ROOT/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Resource paths are set by desktop_media after PyInstaller's own GI hooks.
# A KDE session must still select the shipped GTK backend, not an ambient Qt.
export PYWEBVIEW_GUI=gtk
cd "$GTK_ROOT"
