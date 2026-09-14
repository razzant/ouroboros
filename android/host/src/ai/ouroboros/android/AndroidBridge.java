package ai.ouroboros.android;

import android.app.PendingIntent;
import android.content.ComponentName;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ComponentInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageInstaller;
import android.content.pm.PackageManager;
import android.content.pm.ProviderInfo;
import android.content.pm.ResolveInfo;
import android.database.Cursor;
import android.location.Location;
import android.location.LocationManager;
import android.net.LocalServerSocket;
import android.net.LocalSocket;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONArray;
import org.json.JSONObject;

/** App-UID SDK calls for the existing Linux core; no task or process supervisor. */
final class AndroidBridge implements Closeable {
    static final String SOCKET = "ai.ouroboros.android.rpc";
    private static final int MAX_REQUEST_BYTES = 4 * 1024 * 1024;
    private static final String[] METHODS = {"capabilities", "packages.list", "packages.inspect", "packages.sessions",
            "packages.install", "packages.install.status", "packages.install.abandon",
            "providers.list", "intent.resolve", "intent.start", "content.query", "content.call",
            "content.insert", "content.update", "content.delete", "content.read", "content.write",
            "location.state", "location.get", "accessibility.state", "accessibility.windows",
            "accessibility.perform", "notifications.state", "notifications.list"};
    private final Context context;
    private final LocalServerSocket server;
    private final ExecutorService calls = Executors.newCachedThreadPool();
    private final Set<LocalSocket> sockets = ConcurrentHashMap.newKeySet();
    private final Handler main = new Handler(Looper.getMainLooper());
    private volatile boolean closed;

    static AndroidBridge start(Context context) throws IOException {
        AndroidBridge bridge = new AndroidBridge(context);
        new Thread(bridge::accept, "ouroboros-android-rpc").start();
        return bridge;
    }

    private AndroidBridge(Context context) throws IOException {
        this.context = context.getApplicationContext();
        server = new LocalServerSocket(SOCKET);
    }

    private void accept() {
        while (!closed) {
            try {
                LocalSocket socket = server.accept();
                sockets.add(socket);
                if (closed) { sockets.remove(socket); socket.close(); break; }
                calls.execute(() -> serve(socket));
            } catch (Exception error) {
                if (!closed) android.util.Log.e("OuroborosHost", "Android RPC accept failed", error);
                break;
            }
        }
    }

    private void serve(LocalSocket socket) {
        Object id = JSONObject.NULL;
        boolean dispatched = false;
        try {
            // Abstract sockets have no filesystem mode bits. Authenticate the actual peer UID.
            int uid = socket.getPeerCredentials().getUid();
            if (uid != 0 && uid != android.os.Process.myUid()) throw new SecurityException("Root or host UID required");
            socket.setSoTimeout(15000); // Request framing only, not a provider operation deadline.
            InputStream input = new BufferedInputStream(socket.getInputStream());
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            int value;
            while ((value = input.read()) != '\n') {
                if (value == -1) throw new IOException("Request ended before newline");
                if (bytes.size() == MAX_REQUEST_BYTES) throw new IOException("Request exceeds 4194304 bytes");
                bytes.write(value);
            }
            JSONObject request = new JSONObject(bytes.toString("UTF-8"));
            id = request.opt("id");
            if (id == null) id = JSONObject.NULL;
            JSONObject params = request.optJSONObject("params");
            if (request.has("params") && params == null) throw new IllegalArgumentException("params must be an object");
            dispatched = true;
            Object result = dispatch(request.getString("method"), params == null ? new JSONObject() : params);
            send(socket, new JSONObject().put("id", id).put("ok", true).put("result", result));
        } catch (Exception error) {
            try {
                JSONObject detail = new JSONObject().put("type", error.getClass().getName())
                        .put("message", error.getMessage() == null ? "" : error.getMessage())
                        .put("outcome", dispatched ? "not_confirmed" : "not_dispatched");
                if (error instanceof UiTimeout) detail.put("outcome", ((UiTimeout) error).started ? "unknown" : "not_dispatched");
                send(socket, new JSONObject().put("id", id).put("ok", false).put("error", detail));
            } catch (Exception disconnected) { /* A lost response never triggers a repeated operation. */ }
        } finally {
            sockets.remove(socket);
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private static void send(LocalSocket socket, JSONObject response) throws IOException {
        OutputStream output = socket.getOutputStream();
        output.write((response.toString() + "\n").getBytes(StandardCharsets.UTF_8));
        output.flush();
    }

    @Override public void close() {
        closed = true;
        try { server.close(); } catch (IOException ignored) { }
        for (LocalSocket socket : sockets) try { socket.close(); } catch (IOException ignored) { }
        calls.shutdownNow();
    }

    private Object dispatch(String method, JSONObject p) throws Exception {
        PackageManager pm = context.getPackageManager();
        switch (method) {
            case "capabilities": return capabilities(pm);
            case "packages.list": {
                JSONArray rows = new JSONArray();
                for (PackageInfo info : pm.getInstalledPackages(0)) rows.put(packageInfo(info, pm));
                return page(rows, p).put("coverage", "packages_visible_to_host_uid");
            }
            case "packages.inspect": {
                PackageInfo info = pm.getPackageInfo(p.getString("package"), PackageManager.GET_ACTIVITIES
                        | PackageManager.GET_SERVICES | PackageManager.GET_RECEIVERS | PackageManager.GET_PROVIDERS
                        | PackageManager.GET_PERMISSIONS | PackageManager.MATCH_DISABLED_COMPONENTS);
                return packageInfo(info, pm).put("activities", components(info.activities))
                        .put("services", components(info.services)).put("receivers", components(info.receivers))
                        .put("providers", components(info.providers)).put("permissions", permissions(info, pm));
            }
            case "packages.sessions": return packageSessions(pm, p);
            case "packages.install": return installPackage(pm, p);
            case "packages.install.status": return installStatus(p);
            case "packages.install.abandon": return abandonInstall(pm, p);
            case "providers.list": {
                JSONArray rows = new JSONArray();
                List<ProviderInfo> providers = pm.queryContentProviders(null, 0, PackageManager.MATCH_DISABLED_COMPONENTS);
                if (providers != null) for (ProviderInfo info : providers) rows.put(component(info));
                return page(rows, p).put("coverage", "providers_visible_to_host_uid");
            }
            case "intent.resolve": {
                JSONArray rows = new JSONArray();
                for (ResolveInfo info : pm.queryIntentActivities(intent(p), 0)) rows.put(component(info.activityInfo));
                return page(rows, p).put("coverage", "visible_matching_activities");
            }
            case "intent.start": return startActivity(intent(p));
            case "location.state": return locationState(p);
            case "location.get": return locationGet(p);
            case "accessibility.state": return OuroborosAccessibilityService.state();
            case "accessibility.windows": return OuroborosAccessibilityService.windows(p);
            case "accessibility.perform": return OuroborosAccessibilityService.perform(p);
            case "notifications.state": return OuroborosNotificationListener.state();
            case "notifications.list": return OuroborosNotificationListener.list(p);
            default:
                if (Arrays.asList(METHODS).contains(method)) return content(method, p);
                throw new IllegalArgumentException("Unknown method: " + method);
        }
    }

    /** Inspect installer-owned sessions so a lost response never causes a duplicate install. */
    private JSONObject packageSessions(PackageManager pm, JSONObject p) throws Exception {
        JSONArray rows = new JSONArray();
        PackageInstaller installer = pm.getPackageInstaller();
        for (PackageInstaller.SessionInfo info : installer.getAllSessions()) {
            if (!context.getPackageName().equals(info.getInstallerPackageName())) continue;
            rows.put(sessionInfo(info));
        }
        return page(rows, p).put("coverage", "installer_owned_packageinstaller_sessions")
                .put("source_total_known", true);
    }

    /**
     * Stage one APK in Android's PackageInstaller. The operation is asynchronous:
     * the returned receipt proves only that bytes were staged and commit submitted.
     * packages.install.status or packages.sessions is required to observe completion.
     */
    private synchronized JSONObject installPackage(PackageManager pm, JSONObject p) throws Exception {
        String source = p.getString("source_uri");
        Uri uri = Uri.parse(source);
        if (!uri.isAbsolute()) throw new IllegalArgumentException("source_uri must be an absolute URI");
        String key = p.optString("idempotency_key", "");
        if (key.isEmpty()) throw new IllegalArgumentException("idempotency_key is required");
        if (!key.matches("[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"))
            throw new IllegalArgumentException("idempotency_key must be 1-128 ASCII characters");
        SourceDigest digest = digestSource(uri);
        android.content.SharedPreferences receipts = receipts();
        String previous = receipts.getString(key, null);
        if (previous != null) {
            JSONObject receipt = new JSONObject(previous);
            if (!digest.sha256.equals(receipt.optString("source_sha256")))
                throw new IllegalArgumentException("idempotency_key was already used for another APK");
            int sessionId = receipt.optInt("session_id", -1);
            PackageInstaller.SessionInfo active = sessionId < 0 ? null : pm.getPackageInstaller().getSessionInfo(sessionId);
            if (active != null) return receipt.put("deduplicated", true).put("completion_observed", false);
            if (!"submitted".equals(receipt.optString("status")))
                return receipt.put("deduplicated", true);
            // A sealed or completed session may no longer be listed. The prior receipt remains authoritative
            // for idempotency, and callers must inspect packages.install.status before deciding to retry.
            return receipt.put("deduplicated", true).put("outcome", "unknown")
                    .put("retry_automatically", false);
        }
        PackageInstaller installer = pm.getPackageInstaller();
        PackageInstaller.SessionParams params = new PackageInstaller.SessionParams(
                PackageInstaller.SessionParams.MODE_FULL_INSTALL);
        params.setInstallReason(PackageManager.INSTALL_REASON_USER);
        params.setSize(digest.size);
        int sessionId = installer.createSession(params);
        PackageInstaller.Session session = installer.openSession(sessionId);
        boolean commitSubmitted = false;
        try {
            try (InputStream input = context.getContentResolver().openInputStream(uri)) {
                if (input == null) throw new IOException("source_uri could not be opened");
                MessageDigest stagedDigest = MessageDigest.getInstance("SHA-256");
                try (OutputStream output = session.openWrite("base.apk", 0, digest.size)) {
                    byte[] buffer = new byte[64 * 1024]; int count;
                    long copied = 0;
                    while ((count = input.read(buffer)) != -1) {
                        copied += count;
                        stagedDigest.update(buffer, 0, count); output.write(buffer, 0, count);
                    }
                    if (copied != digest.size || !digest.sha256.equals(hex(stagedDigest.digest())))
                        throw new IOException("source changed while staging APK");
                    session.fsync(output);
                }
            }
            Intent callback = new Intent(context, PackageInstallReceiver.class)
                    .setAction(PackageInstallReceiver.ACTION)
                    .putExtra(PackageInstallReceiver.KEY, key)
                    .putExtra(PackageInstallReceiver.SESSION, sessionId);
            int pendingFlags = PendingIntent.FLAG_UPDATE_CURRENT;
            // PackageInstaller fills status extras into the callback Intent. Mutable PendingIntent is
            // required for that fill-in on Android 12+, while the explicit non-exported receiver keeps
            // the callback private to this package.
            if (Build.VERSION.SDK_INT >= 31) pendingFlags |= PendingIntent.FLAG_MUTABLE;
            PendingIntent pending = PendingIntent.getBroadcast(context, sessionId, callback, pendingFlags);
            JSONObject receipt = new JSONObject().put("idempotency_key", key)
                    .put("source_uri", source).put("source_sha256", digest.sha256)
                    .put("source_size", digest.size).put("session_id", sessionId)
                    .put("status", "submitted").put("outcome", "submitted")
                    .put("completion_observed", false).put("retry_automatically", false)
                    .put("rollback", new JSONObject().put("supported", false)
                            .put("reason", "PackageInstaller commit does not expose a rollback handle here"));
            if (!receipts.edit().putString(key, receipt.toString()).commit())
                throw new IOException("Install receipt could not be saved; commit was not submitted");
            commitSubmitted = true;
            session.commit(pending.getIntentSender());
            return receipt;
        } catch (Exception error) {
            if (!commitSubmitted) receipts.edit().remove(key).apply();
            if (!commitSubmitted) try { session.abandon(); } catch (Exception ignored) { }
            throw error;
        } finally { session.close(); }
    }

    private JSONObject installStatus(JSONObject p) throws Exception {
        String key = p.getString("idempotency_key");
        String value = receipts().getString(key, null);
        if (value == null) return new JSONObject().put("known", false).put("idempotency_key", key);
        return new JSONObject(value).put("known", true);
    }

    /** Abandon one installer-owned session after an unknown or rejected outcome. */
    private synchronized JSONObject abandonInstall(PackageManager pm, JSONObject p) throws Exception {
        String key = p.getString("idempotency_key");
        String value = receipts().getString(key, null);
        if (value == null) return new JSONObject().put("known", false).put("idempotency_key", key);
        JSONObject receipt = new JSONObject(value);
        int sessionId = receipt.optInt("session_id", -1);
        if (sessionId < 0) return receipt.put("known", true).put("outcome", "unknown")
                .put("retry_automatically", false);
        if (pm.getPackageInstaller().getSessionInfo(sessionId) == null)
            return receipt.put("known", true).put("outcome", "unknown")
                    .put("retry_automatically", false);
        pm.getPackageInstaller().abandonSession(sessionId);
        receipt.put("status", "abandoned").put("outcome", "abandoned")
                .put("completion_observed", true).put("status_message", "Session abandoned by owner");
        receipts().edit().putString(key, receipt.toString()).commit();
        PackageInstallReceiver.cancelNotification(context, sessionId);
        return receipt.put("known", true);
    }

    private android.content.SharedPreferences receipts() {
        return context.getSharedPreferences("package_install_receipts", Context.MODE_PRIVATE);
    }

    private static JSONObject sessionInfo(PackageInstaller.SessionInfo info) throws Exception {
        JSONObject result = new JSONObject().put("session_id", info.getSessionId())
                .put("installer_package", nullable(info.getInstallerPackageName()))
                .put("package", nullable(info.getAppPackageName())).put("label", nullable(info.getAppLabel()))
                .put("active", info.isActive()).put("sealed", info.isSealed())
                .put("staged", Build.VERSION.SDK_INT >= 29 && info.isStaged())
                .put("progress", info.getProgress());
        result.put("size_bytes", Build.VERSION.SDK_INT >= 27 ? info.getSize() : JSONObject.NULL);
        if (Build.VERSION.SDK_INT >= 30) result.put("created_ms", info.getCreatedMillis());
        return result;
    }

    private SourceDigest digestSource(Uri uri) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        long size = 0;
        try (InputStream input = context.getContentResolver().openInputStream(uri)) {
            if (input == null) throw new IOException("source_uri could not be opened");
            byte[] buffer = new byte[64 * 1024]; int count;
            while ((count = input.read(buffer)) != -1) {
                size += count;
                digest.update(buffer, 0, count);
            }
        }
        return new SourceDigest(hex(digest.digest()), size);
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) result.append(String.format("%02x", value & 0xff));
        return result.toString();
    }

    private static final class SourceDigest {
        final String sha256; final long size;
        SourceDigest(String sha256, long size) { this.sha256 = sha256; this.size = size; }
    }

    private JSONObject capabilities(PackageManager pm) throws Exception {
        PackageInfo own = pm.getPackageInfo(context.getPackageName(), PackageManager.GET_PERMISSIONS);
        return new JSONObject().put("protocol", 1).put("sdk", Build.VERSION.SDK_INT)
                .put("package", context.getPackageName()).put("uid", android.os.Process.myUid())
                .put("methods", new JSONArray(Arrays.asList(METHODS))).put("permissions", permissions(own, pm))
                .put("can_request_package_installs", pm.canRequestPackageInstalls())
                .put("typed_values", "null,string,boolean,int,long,float,double,uri,bytes,string[],int[],long[],bundle")
                .put("provider_authority", "host_app_uid; root caller does not bypass Android provider permissions")
                .put("root_commands", "android-exec").put("source_total_known", false)
                .put("request_max_bytes", MAX_REQUEST_BYTES);
    }

    /** Return provider and permission state without exposing coordinates. */
    private JSONObject locationState(JSONObject p) throws Exception {
        LocationManager manager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        if (manager == null) throw new IOException("LocationManager unavailable");
        JSONArray providers = new JSONArray();
        for (String provider : new String[]{LocationManager.GPS_PROVIDER,
                LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER}) {
            boolean enabled = false;
            try { enabled = manager.isProviderEnabled(provider); } catch (Exception ignored) { }
            providers.put(new JSONObject().put("provider", provider).put("enabled", enabled));
        }
        return new JSONObject().put("permission_fine", context.checkSelfPermission(
                        android.Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
                .put("permission_coarse", context.checkSelfPermission(
                        android.Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED)
                .put("permission_background", Build.VERSION.SDK_INT < 29 || context.checkSelfPermission(
                        android.Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED)
                .put("providers", providers).put("source_total_known", false);
    }

    /** Read a current location with a bounded provider request; no background tracking is retained. */
    private JSONObject locationGet(JSONObject p) throws Exception {
        LocationManager manager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        if (manager == null) throw new IOException("LocationManager unavailable");
        if (context.checkSelfPermission(android.Manifest.permission.ACCESS_COARSE_LOCATION)
                != PackageManager.PERMISSION_GRANTED)
            throw new SecurityException("Location permission denied");
        String provider = p.optString("provider", "");
        if (provider.isEmpty()) {
            for (String candidate : new String[]{LocationManager.GPS_PROVIDER,
                    LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER}) {
                try { if (manager.isProviderEnabled(candidate)) { provider = candidate; break; } }
                catch (Exception ignored) { }
            }
        }
        if (provider.isEmpty()) return new JSONObject().put("available", false)
                .put("reason", "no_enabled_provider");
        if (Build.VERSION.SDK_INT < 30) {
            Location last = manager.getLastKnownLocation(provider);
            return locationResult(last, false, provider);
        }
        long timeoutMs = Math.max(1000, Math.min(30000, p.optLong("timeout_ms", 10000)));
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Location> result = new AtomicReference<>();
        android.os.CancellationSignal cancellation = new android.os.CancellationSignal();
        manager.getCurrentLocation(provider, cancellation, calls, location -> {
            result.set(location); done.countDown();
        });
        boolean completed;
        try { completed = done.await(timeoutMs, TimeUnit.MILLISECONDS); }
        finally { cancellation.cancel(); }
        Location location = result.get();
        if (location == null) return new JSONObject().put("provider", provider)
                .put("available", false).put("fresh", false)
                .put("reason", completed ? "provider_returned_null" : "no_fix_within_timeout")
                .put("permission_background", Build.VERSION.SDK_INT < 29 || context.checkSelfPermission(
                        android.Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED);
        return locationResult(location, true, provider);
    }

    private static JSONObject locationResult(Location location, boolean fresh, String provider) throws Exception {
        JSONObject result = new JSONObject().put("provider", provider).put("fresh", fresh)
                .put("available", location != null);
        if (location == null) return result.put("reason", fresh ? "no_fix_within_timeout" : "no_last_known_fix");
        return result.put("latitude", location.getLatitude()).put("longitude", location.getLongitude())
                .put("accuracy_m", location.hasAccuracy() ? location.getAccuracy() : JSONObject.NULL)
                .put("time_ms", location.getTime()).put("mock", location.isFromMockProvider())
                .put("age_ms", Math.max(0L, (android.os.SystemClock.elapsedRealtimeNanos()
                        - location.getElapsedRealtimeNanos()) / 1000000L));
    }

    private static JSONArray permissions(PackageInfo info, PackageManager pm) throws Exception {
        JSONArray result = new JSONArray();
        if (info.requestedPermissions != null) for (String permission : info.requestedPermissions)
            result.put(new JSONObject().put("name", permission)
                    .put("granted", pm.checkPermission(permission, info.packageName) == PackageManager.PERMISSION_GRANTED));
        return result;
    }

    private static JSONObject packageInfo(PackageInfo info, PackageManager pm) throws Exception {
        return new JSONObject().put("package", info.packageName).put("version", nullable(info.versionName))
                .put("label", info.applicationInfo == null ? JSONObject.NULL : pm.getApplicationLabel(info.applicationInfo).toString())
                .put("enabled", info.applicationInfo != null && info.applicationInfo.enabled);
    }

    private static JSONArray components(ComponentInfo[] infos) throws Exception {
        JSONArray result = new JSONArray();
        if (infos != null) for (ComponentInfo info : infos) result.put(component(info));
        return result;
    }

    private static JSONObject component(ComponentInfo info) throws Exception {
        JSONObject result = new JSONObject().put("package", info.packageName).put("name", info.name)
                .put("component", new ComponentName(info.packageName, info.name).flattenToString())
                .put("enabled", info.enabled).put("exported", info.exported);
        if (info instanceof ProviderInfo) {
            ProviderInfo provider = (ProviderInfo) info;
            result.put("authority", nullable(provider.authority)).put("read_permission", nullable(provider.readPermission))
                    .put("write_permission", nullable(provider.writePermission)).put("grant_uri_permissions", provider.grantUriPermissions);
        }
        return result;
    }

    private static JSONObject page(JSONArray source, JSONObject p) throws Exception {
        int offset = nonnegative(p, "offset", 0), limit = positive(p, "limit", 100);
        JSONArray rows = new JSONArray();
        for (int i = offset; i < source.length() && rows.length() < limit; i++) rows.put(source.get(i));
        boolean more = (long) offset + rows.length() < source.length();
        return new JSONObject().put("rows", rows).put("offset", offset).put("count", rows.length())
                .put("visible_total", source.length()).put("source_total_known", false)
                .put("truncated", more).put("next_offset", more ? offset + rows.length() : JSONObject.NULL);
    }

    private Intent intent(JSONObject p) throws Exception {
        Intent intent = new Intent();
        if (p.has("action")) intent.setAction(p.getString("action"));
        Uri data = p.isNull("data") ? null : Uri.parse(p.getString("data"));
        intent.setDataAndType(data, string(p, "type"));
        if (p.has("package")) intent.setPackage(p.getString("package"));
        if (p.has("component")) {
            ComponentName component = ComponentName.unflattenFromString(p.getString("component"));
            if (component == null) throw new IllegalArgumentException("component must be package/class");
            intent.setComponent(component);
        }
        String[] categories = strings(array(p, "categories"));
        if (categories != null) for (String category : categories) intent.addCategory(category);
        intent.setFlags(p.has("flags") ? Integer.parseInt(p.getString("flags")) : 0);
        intent.putExtras(bundle(object(p, "extras")));
        return intent;
    }

    private JSONObject startActivity(Intent intent) throws Exception {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        AtomicBoolean started = new AtomicBoolean();
        FutureTask<JSONObject> call = new FutureTask<>(() -> {
            started.set(true);
            context.startActivity(intent);
            return new JSONObject().put("submitted", true).put("completion_observed", false)
                    .put("flags", intent.getFlags());
        });
        if (!main.post(call)) throw new IOException("Main looper rejected activity dispatch");
        try { return call.get(5, TimeUnit.SECONDS); }
        catch (TimeoutException error) {
            call.cancel(false); main.removeCallbacks(call);
            throw new UiTimeout(started.get());
        } catch (InterruptedException error) {
            call.cancel(false); main.removeCallbacks(call);
            Thread.currentThread().interrupt();
            throw new UiTimeout(started.get());
        } catch (java.util.concurrent.ExecutionException error) {
            Throwable cause = error.getCause();
            if (cause instanceof Exception) throw (Exception) cause;
            throw error;
        }
    }

    private static final class UiTimeout extends IOException {
        final boolean started;
        UiTimeout(boolean started) { super("Activity dispatch timed out; verify screen before retrying"); this.started = started; }
    }

    private Object content(String method, JSONObject p) throws Exception {
        ContentResolver resolver = context.getContentResolver();
        Uri uri = Uri.parse(p.getString("uri"));
        JSONObject result = new JSONObject().put("source_uri", uri.toString());
        switch (method) {
            case "content.query": {
                int offset = nonnegative(p, "offset", 0), limit = positive(p, "limit", 100);
                try (Cursor cursor = resolver.query(uri, strings(array(p, "projection")),
                        string(p, "selection"), strings(array(p, "selection_args")), string(p, "sort"))) {
                    if (cursor == null) throw new IOException("Provider returned a null cursor; coverage unknown");
                    JSONArray rows = new JSONArray();
                    boolean more = cursor.moveToPosition(offset);
                    while (more && rows.length() < limit) {
                        JSONObject values = new JSONObject();
                        for (int i = 0; i < cursor.getColumnCount(); i++) {
                            Object value;
                            switch (cursor.getType(i)) {
                                case Cursor.FIELD_TYPE_NULL: value = null; break;
                                case Cursor.FIELD_TYPE_INTEGER: value = cursor.getLong(i); break;
                                case Cursor.FIELD_TYPE_FLOAT: value = cursor.getDouble(i); break;
                                case Cursor.FIELD_TYPE_BLOB: value = cursor.getBlob(i); break;
                                default: value = cursor.getString(i);
                            }
                            values.put(cursor.getColumnName(i), typed(value));
                        }
                        rows.put(new JSONObject().put("row_index", offset + rows.length())
                                .put("source_uri", uri.toString()).put("values", values));
                        more = cursor.moveToNext();
                    }
                    return result.put("columns", new JSONArray(Arrays.asList(cursor.getColumnNames())))
                            .put("rows", rows).put("offset", offset).put("count", rows.length())
                            .put("truncated", more).put("next_offset", more ? offset + rows.length() : JSONObject.NULL)
                            .put("source_total_known", false).put("coverage", "requested_uri_cursor_only; repeat pages are not a snapshot");
                }
            }
            case "content.call": return result.put("bundle", typed(resolver.call(uri, p.getString("method"),
                    string(p, "arg"), bundle(object(p, "extras")))));
            case "content.insert": return result.put("inserted_uri", nullable(resolver.insert(uri, values(p.getJSONObject("values")))));
            case "content.update": return result.put("affected_rows", resolver.update(uri, values(p.getJSONObject("values")),
                    string(p, "selection"), strings(array(p, "selection_args"))));
            case "content.delete": return result.put("affected_rows", resolver.delete(uri,
                    string(p, "selection"), strings(array(p, "selection_args"))));
            case "content.read": {
                int offset = nonnegative(p, "offset", 0), length = positive(p, "length", 65536);
                try (InputStream input = resolver.openInputStream(uri)) {
                    if (input == null) throw new IOException("Provider returned no input stream");
                    long skipped = 0;
                    while (skipped < offset) {
                        long count = input.skip(offset - skipped);
                        if (count == 0) { if (input.read() == -1) break; count = 1; }
                        skipped += count;
                    }
                    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                    byte[] buffer = new byte[Math.min(length, 65536)]; int count;
                    while (bytes.size() < length && (count = input.read(buffer, 0, Math.min(buffer.length, length - bytes.size()))) != -1)
                        bytes.write(buffer, 0, count);
                    boolean more = input.read() != -1;
                    return result.put("data_base64", Base64.encodeToString(bytes.toByteArray(), Base64.NO_WRAP))
                            .put("offset", skipped).put("count", bytes.size()).put("truncated", more)
                            .put("next_offset", more ? skipped + bytes.size() : JSONObject.NULL);
                }
            }
            case "content.write": {
                byte[] bytes = Base64.decode(p.getString("data_base64"), Base64.DEFAULT);
                try (OutputStream output = resolver.openOutputStream(uri, p.getString("mode"))) {
                    if (output == null) throw new IOException("Provider returned no output stream");
                    output.write(bytes); output.flush();
                }
                return result.put("written_bytes", bytes.length).put("mode", p.getString("mode"));
            }
            default: throw new IllegalArgumentException("Unknown content method: " + method);
        }
    }

    private static Bundle bundle(JSONObject values) throws Exception {
        Bundle bundle = new Bundle();
        if (values == null) return bundle;
        for (java.util.Iterator<String> it = values.keys(); it.hasNext();) {
            String key = it.next(); Object value = decode(values.getJSONObject(key));
            if (value == null || value instanceof String) bundle.putString(key, (String) value);
            else if (value instanceof Boolean) bundle.putBoolean(key, (Boolean) value);
            else if (value instanceof Integer) bundle.putInt(key, (Integer) value);
            else if (value instanceof Long) bundle.putLong(key, (Long) value);
            else if (value instanceof Float) bundle.putFloat(key, (Float) value);
            else if (value instanceof Double) bundle.putDouble(key, (Double) value);
            else if (value instanceof Uri) bundle.putParcelable(key, (Uri) value);
            else if (value instanceof Bundle) bundle.putBundle(key, (Bundle) value);
            else if (value instanceof byte[]) bundle.putByteArray(key, (byte[]) value);
            else if (value instanceof String[]) bundle.putStringArray(key, (String[]) value);
            else if (value instanceof int[]) bundle.putIntArray(key, (int[]) value);
            else if (value instanceof long[]) bundle.putLongArray(key, (long[]) value);
        }
        return bundle;
    }

    private static Object decode(JSONObject typed) throws Exception {
        switch (typed.getString("type")) {
            case "null": return null;
            case "string": return typed.getString("value");
            case "boolean": return typed.getBoolean("value");
            case "int": return Integer.valueOf(typed.getString("value"));
            case "long": return Long.valueOf(typed.getString("value"));
            case "float": return Float.valueOf(typed.getString("value"));
            case "double": return Double.valueOf(typed.getString("value"));
            case "uri": return Uri.parse(typed.getString("value"));
            case "bytes": return Base64.decode(typed.getString("value"), Base64.DEFAULT);
            case "bundle": return bundle(typed.getJSONObject("value"));
            case "string[]": return strings(typed.getJSONArray("value"));
            case "int[]": {
                JSONArray source = typed.getJSONArray("value"); int[] result = new int[source.length()];
                for (int i = 0; i < result.length; i++) result[i] = Integer.parseInt(source.getString(i));
                return result;
            }
            case "long[]": {
                JSONArray source = typed.getJSONArray("value"); long[] result = new long[source.length()];
                for (int i = 0; i < result.length; i++) result[i] = Long.parseLong(source.getString(i));
                return result;
            }
            default: throw new IllegalArgumentException("Unsupported typed value: " + typed.getString("type"));
        }
    }

    private static JSONObject typed(Object value) throws Exception {
        String type; Object encoded = value;
        if (value == null) { type = "null"; encoded = JSONObject.NULL; }
        else if (value instanceof String) type = "string";
        else if (value instanceof Boolean) type = "boolean";
        else if (value instanceof Integer) type = "int";
        else if (value instanceof Long) { type = "long"; encoded = value.toString(); }
        else if (value instanceof Float) type = "float";
        else if (value instanceof Double) type = "double";
        else if (value instanceof Uri) { type = "uri"; encoded = value.toString(); }
        else if (value instanceof byte[]) { type = "bytes"; encoded = Base64.encodeToString((byte[]) value, Base64.NO_WRAP); }
        else if (value instanceof Bundle) {
            type = "bundle"; JSONObject result = new JSONObject();
            for (String key : ((Bundle) value).keySet()) result.put(key, typed(((Bundle) value).get(key)));
            encoded = result;
        } else if (value instanceof String[]) { type = "string[]"; encoded = new JSONArray(Arrays.asList((String[]) value)); }
        else if (value instanceof int[]) {
            type = "int[]"; JSONArray result = new JSONArray();
            for (int item : (int[]) value) result.put(item); encoded = result;
        } else if (value instanceof long[]) {
            type = "long[]"; JSONArray result = new JSONArray();
            for (long item : (long[]) value) result.put(Long.toString(item)); encoded = result;
        } else return new JSONObject().put("type", "unsupported").put("java_class", value.getClass().getName())
                .put("value_available", false);
        return new JSONObject().put("type", type).put("value", encoded);
    }

    private static ContentValues values(JSONObject source) throws Exception {
        ContentValues result = new ContentValues();
        for (java.util.Iterator<String> it = source.keys(); it.hasNext();) {
            String key = it.next(); Object value = decode(source.getJSONObject(key));
            if (value == null) result.putNull(key);
            else if (value instanceof String) result.put(key, (String) value);
            else if (value instanceof Boolean) result.put(key, (Boolean) value);
            else if (value instanceof Integer) result.put(key, (Integer) value);
            else if (value instanceof Long) result.put(key, (Long) value);
            else if (value instanceof Float) result.put(key, (Float) value);
            else if (value instanceof Double) result.put(key, (Double) value);
            else if (value instanceof byte[]) result.put(key, (byte[]) value);
            else throw new IllegalArgumentException("Unsupported ContentValues type for " + key);
        }
        return result;
    }

    private static String[] strings(JSONArray values) throws Exception {
        if (values == null) return null;
        String[] result = new String[values.length()];
        for (int i = 0; i < result.length; i++) result[i] = values.getString(i);
        return result;
    }
    private static String string(JSONObject p, String key) throws Exception { return p.isNull(key) ? null : p.getString(key); }
    private static JSONArray array(JSONObject p, String key) throws Exception { return p.isNull(key) ? null : p.getJSONArray(key); }
    private static JSONObject object(JSONObject p, String key) throws Exception { return p.isNull(key) ? null : p.getJSONObject(key); }
    private static Object nullable(Object value) { return value == null ? JSONObject.NULL : value.toString(); }
    private static int nonnegative(JSONObject p, String key, int fallback) throws Exception {
        int value = p.has(key) ? Integer.parseInt(p.getString(key)) : fallback;
        if (value < 0) throw new IllegalArgumentException(key + " must be nonnegative");
        return value;
    }
    private static int positive(JSONObject p, String key, int fallback) throws Exception {
        int value = nonnegative(p, key, fallback);
        if (value == 0) throw new IllegalArgumentException(key + " must be positive");
        return value;
    }
}
