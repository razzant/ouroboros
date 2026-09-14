package ai.ouroboros.android;

import android.app.*;
import android.content.Intent;
import android.os.IBinder;
import android.os.Handler;
import android.net.ConnectivityManager;
import android.net.LinkProperties;
import android.net.Network;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Android bridge lifetime, independent of WebView; only the shared launcher supervises the core. */
public final class CoreService extends Service {
    private final ExecutorService work = Executors.newSingleThreadExecutor();
    private final ExecutorService urgent = Executors.newSingleThreadExecutor();
    private final java.util.concurrent.atomic.AtomicInteger operation = new java.util.concurrent.atomic.AtomicInteger();
    private static final String CHANNEL = "ouroboros_runtime";
    private static final long START_OBSERVATION_MS = 60000;
    private AndroidBridge bridge;
    private ConnectivityManager connectivity;
    private ConnectivityManager.NetworkCallback networkCallback;
    private Network currentNetwork;
    private final java.util.concurrent.atomic.AtomicInteger dnsRevision = new java.util.concurrent.atomic.AtomicInteger();
    private volatile boolean closed;
    private volatile String networkNote = "";
    private volatile String runtimeText = "Проверяю состояние ядра";
    private volatile String queuedDns = null;
    private String queuedNetworkNote = null;

    @Override public void onCreate() {
        super.onCreate();
        getSystemService(NotificationManager.class).createNotificationChannel(
                new NotificationChannel(CHANNEL, "Ouroboros", NotificationManager.IMPORTANCE_LOW));
        connectivity = getSystemService(ConnectivityManager.class);
        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override public void onAvailable(Network network) { currentNetwork = network; }
            @Override public void onLinkPropertiesChanged(Network network, LinkProperties properties) {
                if (network.equals(currentNetwork)) updateNetworkDns(properties);
            }
            @Override public void onLost(Network network) {
                if (network.equals(currentNetwork)) { currentNetwork = null; updateNetworkDns(null); }
            }
        };
        connectivity.registerDefaultNetworkCallback(networkCallback, new Handler(getMainLooper()));
        currentNetwork = connectivity.getActiveNetwork();
        updateNetworkDns(currentNetwork == null ? null : connectivity.getLinkProperties(currentNetwork));
    }

    private Notification notification(String text) {
        runtimeText = text;
        if (!networkNote.isEmpty()) text += " · " + networkNote;
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Intent stop = new Intent(this, CoreService.class).setAction("panic");
        PendingIntent panic = PendingIntent.getService(this, 1, stop,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this, CHANNEL).setSmallIcon(android.R.drawable.ic_menu_manage)
                .setContentTitle("Ouroboros").setContentText(text).setContentIntent(open)
                .addAction(new Notification.Action.Builder(null, "Остановить агента", panic).build())
                .setOngoing(true).build();
    }

    private void updateNetworkDns(LinkProperties properties) {
        if (closed) return;
        String dns = properties == null ? "" : RuntimeClient.dnsConfiguration(properties.getDnsServers());
        String note = dns.isEmpty() ? "Нет текущих DNS: файл Linux оставлен без изменений" : "";
        if (properties != null && android.os.Build.VERSION.SDK_INT >= 28 && properties.isPrivateDnsActive())
            note += (note.isEmpty() ? "" : "; ") + "Android Private DNS не переносится в обычный DNS Linux";
        if (dns.equals(queuedDns) && note.equals(queuedNetworkNote)) return;
        queuedDns = dns; queuedNetworkNote = note;
        networkNote = note;
        int revision = dnsRevision.incrementAndGet();
        work.execute(() -> {
            if (closed || revision != dnsRevision.get()) return;
            try { RuntimeClient.updateDns(dns); }
            catch (Exception error) {
                if (closed || revision != dnsRevision.get()) return;
                queuedDns = null; // A later network/control event may retry after root is granted.
                networkNote = "Не удалось обновить DNS Linux";
                android.util.Log.e("OuroborosHost", "Linux DNS update failed", error);
            }
            if (!closed && revision == dnsRevision.get())
                getSystemService(NotificationManager.class).notify(1, notification(runtimeText));
        });
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        // Android may recreate the sticky service after process death. Restore
        // the bridge alone: replaying a previous Start would defeat Panic.
        String action = intent == null || intent.getAction() == null ? "status" : intent.getAction();
        // A late automatic check must not cancel a queued owner Start or Panic.
        int request = ("start".equals(action) || "panic".equals(action))
                ? operation.incrementAndGet() : operation.get();
        startForegroundOwnerNotification();
        if (!"panic".equals(action)) {
            currentNetwork = connectivity.getActiveNetwork();
            updateNetworkDns(currentNetwork == null ? null : connectivity.getLinkProperties(currentNetwork));
        }
        Exception unavailable = null;
        // Socket lifetime is owned on Android's main thread, like onDestroy.
        if (bridge == null && !"panic".equals(action)) {
            try { bridge = AndroidBridge.start(this); }
            catch (java.io.IOException error) { unavailable = error; }
        }
        final Exception bridgeFailure = unavailable;
        ("panic".equals(action) ? urgent : work).execute(() -> {
            try {
                if (request != operation.get()) return;
                boolean starting = false;
                if ("panic".equals(action)) {
                    try {
                        RuntimeClient.control("port", "owner");
                        RuntimeClient.request("/api/command", "POST", new org.json.JSONObject().put("cmd", "/panic"));
                    } finally { stopForeground(STOP_FOREGROUND_REMOVE); stopSelf(startId); }
                    return;
                }
                if (bridgeFailure != null)
                    android.util.Log.e("OuroborosHost", "Native Android tools unavailable", bridgeFailure);
                if ("start".equals(action) || "boot".equals(action)) {
                    String result = RuntimeClient.control("start", "boot".equals(action) ? "automatic" : "owner");
                    if (result.equals("stopped")) {
                        getSystemService(NotificationManager.class).notify(1, notification("Остановлен владельцем. Нажмите «Запустить» для продолжения."));
                        stopForeground(STOP_FOREGROUND_DETACH);
                        stopSelf(startId); return;
                    }
                    starting = result.equals("starting");
                }
                if (starting) getSystemService(NotificationManager.class).notify(1, notification("Ядро запускается…"));
                long deadline = android.os.SystemClock.elapsedRealtime() + (starting ? START_OBSERVATION_MS : 0);
                while (true) {
                    if (request != operation.get()) return;
                    try {
                        RuntimeClient.control("port", "owner");
                        RuntimeClient.request("/api/health", "GET");
                        break;
                    } catch (Exception error) {
                        if (!starting) throw error;
                        if (android.os.SystemClock.elapsedRealtime() >= deadline) {
                            if (request != operation.get()) return;
                            getSystemService(NotificationManager.class).notify(1, notification(
                                    "Запуск запрошен. Готовность ядра пока не подтверждена; проверьте статус позже."));
                            return;
                        }
                        Thread.sleep(1000);
                    }
                }
                if (request != operation.get()) return;
                getSystemService(NotificationManager.class).notify(1, notification(bridgeFailure == null
                        ? "Ядро работает на телефоне" : "Ядро работает. Android-инструменты недоступны: "
                        + bridgeFailure.getMessage()));
            } catch (Exception error) {
                if (request != operation.get()) return;
                android.util.Log.e("OuroborosHost", "Native control failed: " + action, error);
                getSystemService(NotificationManager.class).notify(1,
                        notification("Действие не выполнено. Откройте статус: " + error.getMessage()));
            }
        });
        return "panic".equals(action) ? START_NOT_STICKY : START_STICKY;
    }

    private void startForegroundOwnerNotification() {
        Notification value = notification("Проверяю состояние ядра");
        if (android.os.Build.VERSION.SDK_INT >= 34) {
            int type = android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE;
            boolean background = checkSelfPermission(android.Manifest.permission.ACCESS_BACKGROUND_LOCATION)
                    == android.content.pm.PackageManager.PERMISSION_GRANTED;
            if (background && (checkSelfPermission(android.Manifest.permission.ACCESS_COARSE_LOCATION)
                    == android.content.pm.PackageManager.PERMISSION_GRANTED
                    || checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION)
                    == android.content.pm.PackageManager.PERMISSION_GRANTED))
                type |= android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION;
            startForeground(1, value, type);
        } else startForeground(1, value);
    }

    @Override public void onDestroy() {
        closed = true; dnsRevision.incrementAndGet();
        if (networkCallback != null) connectivity.unregisterNetworkCallback(networkCallback);
        operation.incrementAndGet();
        if (bridge != null) bridge.close();
        work.shutdownNow(); urgent.shutdown(); super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent) { return null; }
}
