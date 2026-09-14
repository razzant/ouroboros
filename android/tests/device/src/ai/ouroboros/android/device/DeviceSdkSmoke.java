package ai.ouroboros.android.device;

import android.app.Activity;
import android.app.Instrumentation;
import android.content.Context;
import android.content.pm.PackageInstaller;
import android.net.LocalSocket;
import android.net.LocalSocketAddress;
import android.os.Bundle;
import android.util.Log;
import java.io.Closeable;
import java.lang.reflect.Method;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Small on-device contract check for SDK-gated PackageInstaller readback.
 *
 * Own a bridge instance in the instrumented host process, create a session
 * under its UID and call the actual RPC over the authenticated local socket.
 * No APK bytes are committed or installed; no CoreService or root is needed.
 */
public final class DeviceSdkSmoke extends Instrumentation {
    private static final String TAG = "OuroborosDeviceSdkSmoke";
    private static final String PASS = "OBO_DEVICE_SDK_SMOKE=PASS";
    private static final String FAIL = "OBO_DEVICE_SDK_SMOKE=FAIL";

    @Override public void onCreate(Bundle arguments) {
        super.onCreate(arguments);
        start();
    }

    @Override public void onStart() {
        super.onStart();
        Bundle result = new Bundle();
        int code = Activity.RESULT_OK;
        int sessionId = -1;
        Closeable bridge = null;
        try {
            Context target = getTargetContext();
            Class<?> bridgeClass = target.getClassLoader().loadClass("ai.ouroboros.android.AndroidBridge");
            Method startBridge = bridgeClass.getDeclaredMethod("start", Context.class);
            startBridge.setAccessible(true);
            bridge = (Closeable) startBridge.invoke(null, target);
            PackageInstaller installer = target.getPackageManager().getPackageInstaller();
            PackageInstaller.SessionParams params = new PackageInstaller.SessionParams(
                    PackageInstaller.SessionParams.MODE_FULL_INSTALL);
            sessionId = installer.createSession(params);
            PackageInstaller.Session session = installer.openSession(sessionId);
            session.close();

            JSONObject capabilities = rpc("capabilities");
            if (!capabilities.getJSONArray("methods").toString().contains("packages.sessions"))
                throw new AssertionError("capabilities omitted packages.sessions");
            if (!capabilities.has("can_request_package_installs"))
                throw new AssertionError("capabilities omitted install special-access state");
            JSONObject snapshot = rpc("packages.sessions");
            JSONArray rows = snapshot.getJSONArray("rows");
            boolean found = false;
            for (int i = 0; i < rows.length(); i++) {
                if (rows.getJSONObject(i).optInt("session_id", -1) == sessionId) {
                    found = true;
                    break;
                }
            }
            if (!found) throw new AssertionError("created session missing from packages.sessions");
            // The system picker parses WallpaperInfo; a declared service alone is insufficient.
            boolean wallpaperFound = false;
            for (android.content.pm.ResolveInfo info : target.getPackageManager().queryIntentServices(
                    new android.content.Intent(android.service.wallpaper.WallpaperService.SERVICE_INTERFACE)
                            .setPackage(target.getPackageName()), android.content.pm.PackageManager.GET_META_DATA)) {
                new android.app.WallpaperInfo(target, info);
                wallpaperFound = true;
            }
            if (!wallpaperFound) throw new AssertionError("host wallpaper metadata is unavailable");
            result.putString("obo_marker", PASS);
            result.putInt("session_id", sessionId);
            result.putInt("sdk", android.os.Build.VERSION.SDK_INT);
        } catch (Throwable error) {
            code = Activity.RESULT_CANCELED;
            result.putString("obo_marker", FAIL);
            result.putString("error", error.toString());
            Log.e(TAG, FAIL + " sdk=" + android.os.Build.VERSION.SDK_INT, error);
        } finally {
            if (sessionId >= 0) {
                try { getTargetContext().getPackageManager().getPackageInstaller().abandonSession(sessionId); }
                catch (Throwable error) {
                    code = Activity.RESULT_CANCELED;
                    result.putString("obo_marker", FAIL);
                    result.putString("cleanup_error", error.toString());
                    Log.e(TAG, FAIL + " could not abandon smoke session " + sessionId, error);
                }
            }
            if (bridge != null) {
                try { bridge.close(); }
                catch (Throwable error) {
                    code = Activity.RESULT_CANCELED;
                    result.putString("obo_marker", FAIL);
                    result.putString("bridge_cleanup_error", error.toString());
                    Log.e(TAG, FAIL + " could not close smoke bridge", error);
                }
            }
            if (code == Activity.RESULT_OK) Log.i(TAG, PASS + " sdk=" + android.os.Build.VERSION.SDK_INT
                    + " session=" + sessionId);
            sendStatus(2, result);
            finish(code, result);
        }
    }

    /** Call the running host bridge over its authenticated local socket. */
    private static JSONObject rpc(String method) throws Exception {
        LocalSocket socket = new LocalSocket();
        try {
            socket.connect(new LocalSocketAddress("ai.ouroboros.android.rpc",
                    LocalSocketAddress.Namespace.ABSTRACT));
            socket.setSoTimeout(10000);
            socket.getOutputStream().write((new JSONObject().put("id", 1).put("method", method)
                    .put("params", new JSONObject()).toString() + "\n").getBytes("UTF-8"));
            socket.getOutputStream().flush();
            StringBuilder response = new StringBuilder();
            int value;
            while ((value = socket.getInputStream().read()) != -1 && value != '\n') {
                if (response.length() >= 1024 * 1024) throw new AssertionError("oversized bridge response");
                response.append((char) value);
            }
            JSONObject envelope = new JSONObject(response.toString());
            if (!envelope.optBoolean("ok", false)) throw new AssertionError(envelope.opt("error"));
            return envelope.getJSONObject("result");
        } finally { socket.close(); }
    }
}
