package ai.ouroboros.android;

import android.app.*;
import android.content.*;
import android.net.Uri;
import android.os.*;
import android.app.role.RoleManager;
import android.view.*;
import android.webkit.*;
import android.widget.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** The ordinary Ouroboros SPA, hosted by Android rather than a browser tab. */
public final class MainActivity extends Activity {
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private WebView web;
    private LinearLayout root;
    private TextView title;
    private Button menu;
    private TextView status;
    private ValueCallback<Uri[]> chooser;
    private String pendingDownload;
    private boolean visible;
    private boolean checking;
    private boolean loaded;
    private boolean startWhenMissing = true;
    private static final int RUNTIME_PERMISSIONS = 21;
    private static final int ASSISTANT_ROLE_REQUEST = 31;
    private final ArrayDeque<PermissionPrompt> permissionPrompts = new ArrayDeque<>();
    private final IdentityHashMap<PermissionRequest, PermissionPrompt> mediaPrompts = new IdentityHashMap<>();
    private PermissionPrompt activePermissionPrompt;
    private PermissionPrompt geolocationPrompt;

    private static final class PermissionPrompt {
        final String[] permissions;
        final java.util.function.Consumer<Boolean> finish;
        boolean cancelled;
        PermissionPrompt(String[] permissions, java.util.function.Consumer<Boolean> finish) {
            this.permissions = permissions; this.finish = finish;
        }
    }
    private final Runnable refresh = new Runnable() {
        @Override public void run() { if (visible) check(false); }
    };

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        startWhenMissing = saved == null;
        if (saved != null) pendingDownload = saved.getString("pending_download");
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(android.graphics.Color.rgb(13, 11, 15));
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.ime());
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            } else view.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                    insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets;
        });
        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setGravity(Gravity.CENTER_VERTICAL);
        title = new TextView(this); title.setText("Ouroboros"); title.setTextSize(15);
        title.setTextColor(android.graphics.Color.WHITE); title.setPadding(dp(16), 0, 0, 0);
        title.setOnClickListener(v -> showStatus());
        menu = new Button(this); menu.setText("⋮"); menu.setTextSize(23);
        menu.setTextColor(android.graphics.Color.WHITE); menu.setBackgroundColor(android.graphics.Color.TRANSPARENT);
        menu.setContentDescription("Меню приложения");
        menu.setOnClickListener(v -> {
            PopupMenu popup = new PopupMenu(this, menu);
            popup.getMenu().add(0, 1, 0, "Запустить ядро");
            popup.getMenu().add(0, 2, 1, "Обновить окно");
            popup.getMenu().add(0, 3, 2, "Статус ядра");
            popup.getMenu().add(0, 4, 3, "Остановить агента");
            popup.getMenu().add(0, 5, 4, "Разрешения Android");
            popup.getMenu().add(0, 6, 5, "Назначить ассистентом Android");
            popup.setOnMenuItemClickListener(item -> {
                switch (item.getItemId()) {
                    case 1:
                        status.setVisibility(View.VISIBLE); status.setText("Запускаю ядро…");
                        startForegroundService(new Intent(this, CoreService.class).setAction("start")); break;
                    case 2: if (loaded && web != null) web.reload(); check(true); break;
                    case 3: showStatus(); break;
                    case 4:
                        startWhenMissing = false;
                        startForegroundService(new Intent(this, CoreService.class).setAction("panic")); break;
                    case 5: showAccessSetup(); break;
                    case 6: requestAssistantRole(); break;
                    default: return false;
                }
                return true;
            });
            popup.show();
        });
        toolbar.addView(title, new LinearLayout.LayoutParams(0, dp(40), 1));
        title.setGravity(Gravity.CENTER_VERTICAL);
        toolbar.addView(menu, new LinearLayout.LayoutParams(dp(48), dp(40)));
        root.addView(toolbar);
        status = new TextView(this); status.setText("Подключаюсь к ядру на телефоне…");
        status.setTextColor(android.graphics.Color.LTGRAY);
        status.setPadding(16, 4, 16, 8); root.addView(status);
        web = new WebView(this);
        web.getSettings().setJavaScriptEnabled(true);
        web.getSettings().setDomStorageEnabled(true);
        web.getSettings().setGeolocationEnabled(true);
        web.getSettings().setAllowFileAccess(false);
        web.getSettings().setAllowContentAccess(true);
        web.setWebViewClient(new WebViewClient() {
            @Override public void onPageFinished(WebView view, String url) {
                // Read the shared UI's palette; do not maintain a second phone theme.
                if (!url.startsWith(RuntimeClient.baseUrl() + "/")) return;
                view.evaluateJavascript("[getComputedStyle(document.documentElement).getPropertyValue('--bg-primary').trim(),"
                        + "getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim()]", value -> {
                    try {
                        org.json.JSONArray colors = new org.json.JSONArray(value);
                        int background = android.graphics.Color.parseColor(colors.getString(0));
                        int foreground = android.graphics.Color.parseColor(colors.getString(1));
                        root.setBackgroundColor(background); title.setTextColor(foreground);
                        menu.setTextColor(foreground); status.setTextColor(foreground);
                        boolean light = android.graphics.Color.luminance(background) > 0.5;
                        if (Build.VERSION.SDK_INT >= 30 && getWindow().getInsetsController() != null) {
                            int flags = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
                                    | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS;
                            getWindow().getInsetsController().setSystemBarsAppearance(light ? flags : 0, flags);
                        }
                    } catch (Exception ignored) { /* Keep the readable native fallback palette. */ }
                });
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if (uri.toString().startsWith(RuntimeClient.baseUrl() + "/")) return false;
                openExternal(uri); return true;
            }
            @Override public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) {
                    loaded = false; status.setVisibility(View.VISIBLE);
                    status.setText("Жду подключения к ядру…");
                }
            }
            @Override public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
                // The renderer cannot be reused. The native bridge lives in its
                // own process; replacing this Activity neither stops nor starts core work.
                root.removeView(view); view.destroy(); web = null;
                loaded = false; startWhenMissing = false; recreate();
                return true;
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override public void onPermissionRequest(PermissionRequest request) {
                requestMediaPermission(request);
            }
            @Override public void onPermissionRequestCanceled(PermissionRequest request) {
                PermissionPrompt prompt = mediaPrompts.remove(request);
                if (prompt != null) prompt.cancelled = true;
            }
            @Override public void onGeolocationPermissionsShowPrompt(String origin,
                                                                      GeolocationPermissions.Callback callback) {
                requestGeolocationPermission(origin, callback);
            }
            @Override public void onGeolocationPermissionsHidePrompt() {
                if (geolocationPrompt != null) geolocationPrompt.cancelled = true;
                geolocationPrompt = null;
            }
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback,
                                                       FileChooserParams params) {
                if (chooser != null) chooser.onReceiveValue(null);
                chooser = callback;
                try { startActivityForResult(params.createIntent(), 10); }
                catch (ActivityNotFoundException error) { chooser.onReceiveValue(null); chooser = null; }
                return true;
            }
        });
        web.setDownloadListener((url, agent, disposition, mime, length) -> {
            if (!url.startsWith(RuntimeClient.baseUrl() + "/")) {
                notifyOutcome("Этот формат скачивания пока не поддержан приложением"); return;
            }
            if (pendingDownload != null) {
                notifyOutcome("Сначала выберите место для предыдущего файла"); return;
            }
            pendingDownload = url;
            Intent save = new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                    .setType(mime == null || mime.isEmpty() ? "application/octet-stream" : mime)
                    .putExtra(Intent.EXTRA_TITLE, URLUtil.guessFileName(url, disposition, mime));
            try { startActivityForResult(save, 11); }
            catch (ActivityNotFoundException error) {
                pendingDownload = null; notifyOutcome("На телефоне нет приложения выбора файла");
            }
        });
        root.addView(web, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
        startForegroundService(new Intent(this, CoreService.class).setAction("status"));
        if (Build.VERSION.SDK_INT >= 33)
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(
                    android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, this::goBack);
        if (!getPreferences(MODE_PRIVATE).getBoolean("native_access_setup_seen", false))
            showAccessSetup();
    }

    /** Ask Android's ordinary user-consent flow for the assistant role. */
    private void requestAssistantRole() {
        if (Build.VERSION.SDK_INT < 29) {
            notifyOutcome("Роль системного ассистента доступна начиная с Android 10");
            return;
        }
        RoleManager roles = (RoleManager) getSystemService(RoleManager.class);
        if (roles == null || !roles.isRoleAvailable(RoleManager.ROLE_ASSISTANT)) {
            notifyOutcome("На этом Android нет роли системного ассистента");
            return;
        }
        if (roles.isRoleHeld(RoleManager.ROLE_ASSISTANT)) {
            notifyOutcome("Ouroboros уже выбран системным ассистентом");
            return;
        }
        try {
            startActivityForResult(roles.createRequestRoleIntent(RoleManager.ROLE_ASSISTANT),
                    ASSISTANT_ROLE_REQUEST);
        } catch (RuntimeException error) {
            notifyOutcome("Android не открыл выбор системного ассистента");
        }
    }

    private String[] missingRuntimePermissions() {
        ArrayList<String> missing = new ArrayList<>();
        android.content.pm.PackageManager pm = getPackageManager();
        try {
            String[] declared = pm.getPackageInfo(getPackageName(),
                    android.content.pm.PackageManager.GET_PERMISSIONS).requestedPermissions;
            if (declared != null) for (String permission : declared) {
                // Android 11+ requires background location to be granted separately in Settings.
                if (Build.VERSION.SDK_INT >= 30 && android.Manifest.permission.ACCESS_BACKGROUND_LOCATION.equals(permission))
                    continue;
                try {
                    android.content.pm.PermissionInfo info = pm.getPermissionInfo(permission, 0);
                    if ((info.protectionLevel & android.content.pm.PermissionInfo.PROTECTION_MASK_BASE)
                            == android.content.pm.PermissionInfo.PROTECTION_DANGEROUS && !hasPermission(permission))
                        missing.add(permission);
                } catch (android.content.pm.PackageManager.NameNotFoundException unavailable) {
                    // A manifest may declare a capability introduced by a later Android release.
                }
            }
        } catch (android.content.pm.PackageManager.NameNotFoundException error) {
            notifyOutcome("Не удалось прочитать разрешения приложения");
        }
        return missing.toArray(new String[0]);
    }

    private void showAccessSetup() {
        if (isFinishing() || isDestroyed()) return;
        new AlertDialog.Builder(this).setTitle("Доступ Ouroboros к телефону")
                .setMessage("Ouroboros может использовать камеру, микрофон, геопозицию, контакты, календарь "
                        + "и медиафайлы по вашим поручениям. В следующих окнах Android выберите, что разрешить. "
                        + "Уже выданные разрешения не запрашиваются перед каждым действием. "
                        + "Root-доступ выдаётся отдельно в Magisk. Изменить доступ можно в меню приложения.")
                .setPositiveButton("Настроить доступ", (dialog, which) -> {
                    getPreferences(MODE_PRIVATE).edit().putBoolean("native_access_setup_seen", true).apply();
                    enqueuePermissionPrompt(new PermissionPrompt(missingRuntimePermissions(), proceed -> {
                        if (!isFinishing() && !isDestroyed()) {
                            String[] missing = missingRuntimePermissions();
                            notifyOutcome(missing.length == 0 ? "Разрешения настроены" : "Доступ настроен с выбранными ограничениями");
                        }
                    }));
                })
                .setNeutralButton("Настройки Android", (dialog, which) -> startActivity(new Intent(
                        android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getPackageName()))))
                .setNegativeButton("Позже", (dialog, which) ->
                        getPreferences(MODE_PRIVATE).edit().putBoolean("native_access_setup_seen", true).apply())
                .show();
    }

    private boolean isRuntimeOrigin(Uri origin) {
        Uri runtime = Uri.parse(RuntimeClient.baseUrl());
        return origin != null && runtime.getScheme().equals(origin.getScheme())
                && runtime.getEncodedAuthority().equals(origin.getEncodedAuthority());
    }

    private boolean hasPermission(String permission) {
        return checkSelfPermission(permission) == android.content.pm.PackageManager.PERMISSION_GRANTED;
    }

    private static String mediaPermission(String resource) {
        if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(resource)) return android.Manifest.permission.RECORD_AUDIO;
        if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(resource)) return android.Manifest.permission.CAMERA;
        return null;
    }

    private void requestMediaPermission(PermissionRequest request) {
        if (isFinishing() || !isRuntimeOrigin(request.getOrigin())) { request.deny(); return; }
        ArrayList<String> permissions = new ArrayList<>();
        for (String resource : request.getResources()) {
            String permission = mediaPermission(resource);
            if (permission != null && !permissions.contains(permission)) permissions.add(permission);
        }
        if (permissions.isEmpty()) { request.deny(); return; }
        PermissionPrompt prompt = new PermissionPrompt(permissions.toArray(new String[0]), proceed -> {
            mediaPrompts.remove(request);
            ArrayList<String> granted = new ArrayList<>();
            if (proceed && !isFinishing() && isRuntimeOrigin(request.getOrigin())) {
                // WebView can add resource kinds independently of the Android API level.
                for (String resource : request.getResources()) {
                    String permission = mediaPermission(resource);
                    if (permission != null && hasPermission(permission)) granted.add(resource);
                }
            }
            if (granted.isEmpty()) request.deny();
            else request.grant(granted.toArray(new String[0]));
        });
        mediaPrompts.put(request, prompt);
        enqueuePermissionPrompt(prompt);
    }

    private boolean hasLocationPermission() {
        return hasPermission(android.Manifest.permission.ACCESS_COARSE_LOCATION)
                || hasPermission(android.Manifest.permission.ACCESS_FINE_LOCATION);
    }

    private void requestGeolocationPermission(String origin, GeolocationPermissions.Callback callback) {
        if (isFinishing() || !isRuntimeOrigin(Uri.parse(origin))) {
            callback.invoke(origin, false, false); return;
        }
        if (geolocationPrompt != null) {
            geolocationPrompt.cancelled = true;
            geolocationPrompt.finish.accept(false);
        }
        // Respect an existing approximate grant; do not repeatedly ask for a precision upgrade.
        String[] permissions = hasLocationPermission() ? new String[0] : new String[]{
                android.Manifest.permission.ACCESS_COARSE_LOCATION, android.Manifest.permission.ACCESS_FINE_LOCATION};
        geolocationPrompt = new PermissionPrompt(permissions, proceed -> {
            geolocationPrompt = null;
            // Do not retain a WebView grant beyond Android's current permission state.
            callback.invoke(origin, proceed && !isFinishing() && isRuntimeOrigin(Uri.parse(origin))
                    && hasLocationPermission(), false);
        });
        enqueuePermissionPrompt(geolocationPrompt);
    }

    private void enqueuePermissionPrompt(PermissionPrompt prompt) {
        permissionPrompts.add(prompt);
        showNextPermissionPrompt();
    }

    private void showNextPermissionPrompt() {
        // Android allows one runtime-permission dialog at a time, including notifications.
        if (activePermissionPrompt != null || isDestroyed() || isFinishing()) return;
        while (!permissionPrompts.isEmpty()) {
            PermissionPrompt prompt = permissionPrompts.remove();
            if (prompt.cancelled) continue;
            boolean missing = false;
            for (String permission : prompt.permissions) if (!hasPermission(permission)) missing = true;
            if (!missing) { prompt.finish.accept(true); continue; }
            activePermissionPrompt = prompt;
            // Keep coarse+fine together even if a queued request sees a changed grant.
            requestPermissions(prompt.permissions, RUNTIME_PERMISSIONS);
            return;
        }
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != RUNTIME_PERMISSIONS) return;
        PermissionPrompt prompt = activePermissionPrompt;
        activePermissionPrompt = null;
        if (prompt != null && !prompt.cancelled) prompt.finish.accept(true);
        showNextPermissionPrompt();
    }

    private void check(boolean discover) {
        if (checking) return;
        checking = true;
        io.execute(() -> {
            String failure = null;
            try {
                if (discover) RuntimeClient.control("port", "owner");
                try { RuntimeClient.request("/api/health", "GET"); }
                catch (Exception first) {
                    // The bridge and WebView have separate process-local port caches.
                    RuntimeClient.control("port", "owner");
                    RuntimeClient.request("/api/health", "GET");
                }
            } catch (Exception error) { failure = error.getMessage(); }
            final String message = failure;
            runOnUiThread(() -> {
                checking = false;
                if (isDestroyed() || web == null) return;
                if (message == null) {
                    startWhenMissing = false;
                    status.setVisibility(View.GONE);
                    if (!loaded) { loaded = true; web.loadUrl(RuntimeClient.baseUrl() + "/"); }
                } else if (startWhenMissing) {
                    startWhenMissing = false;
                    status.setVisibility(View.VISIBLE);
                    status.setText("Запускаю ядро…");
                    // Passive window entry must not clear a previous Panic.
                    startForegroundService(new Intent(this, CoreService.class).setAction("boot"));
                } else {
                    status.setVisibility(View.VISIBLE);
                    status.setText("Ядро не отвечает. Откройте меню запуска или статуса.");
                }
                handler.removeCallbacks(refresh);
                if (visible) handler.postDelayed(refresh, 3000);
            });
        });
    }

    private void showStatus() {
        io.execute(() -> {
            String text;
            try {
                JSONObject state = RuntimeClient.request("/api/state", "GET");
                text = "Ядро: " + RuntimeClient.baseUrl() + "\nРаботники: "
                        + state.optInt("workers_alive") + "/" + state.optInt("workers_total")
                        + "\nРежим: " + state.optString("runtime_mode")
                        + "\nОжидают: " + state.optInt("pending_count")
                        + "\nАктивные задачи: " + state.optInt("running_count");
            } catch (Exception error) {
                try { text = RuntimeClient.control("status", "owner"); }
                catch (Exception controlError) { text = "Запуск недоступен: " + controlError.getMessage(); }
            }
            final String message = text;
            runOnUiThread(() -> {
                if (!isDestroyed()) new AlertDialog.Builder(this).setTitle("Ouroboros")
                        .setMessage(message).setPositiveButton("Закрыть", null).show();
            });
        });
    }

    private void openExternal(Uri uri) {
        String scheme = uri.getScheme();
        if (!"https".equals(scheme) && !"http".equals(scheme) && !"mailto".equals(scheme)) return;
        try { startActivity(new Intent(Intent.ACTION_VIEW, uri)); }
        catch (ActivityNotFoundException error) { notifyOutcome("Нет приложения для этой ссылки"); }
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == ASSISTANT_ROLE_REQUEST && Build.VERSION.SDK_INT >= 29) {
            RoleManager roles = (RoleManager) getSystemService(RoleManager.class);
            boolean held = roles != null && roles.isRoleHeld(RoleManager.ROLE_ASSISTANT);
            notifyOutcome(held ? "Ouroboros выбран системным ассистентом"
                    : "Выбор системного ассистента отменён");
            return;
        }
        if (request == 10 && chooser != null) {
            chooser.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result, data)); chooser = null;
        }
        if (request == 11) {
            String source = pendingDownload; pendingDownload = null;
            if (result != RESULT_OK || data == null || data.getData() == null || source == null) return;
            Uri target = data.getData();
            io.execute(() -> {
                HttpURLConnection connection = null;
                try {
                    connection = (HttpURLConnection) new URL(source).openConnection();
                    connection.setConnectTimeout(5000); connection.setReadTimeout(30000);
                    String cookies = CookieManager.getInstance().getCookie(source);
                    if (cookies != null) connection.setRequestProperty("Cookie", cookies);
                    try (java.io.InputStream input = connection.getInputStream();
                         java.io.OutputStream output = getContentResolver().openOutputStream(target)) {
                        if (output == null) throw new java.io.IOException("The document provider refused this file");
                        byte[] buffer = new byte[65536]; int count;
                        while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
                    }
                    runOnUiThread(() -> showSavedFile(target));
                } catch (Exception error) { notifyOutcome("Не удалось сохранить файл"); }
                finally { if (connection != null) connection.disconnect(); }
            });
        }
    }

    private void showSavedFile(Uri file) {
        if (isFinishing() || isDestroyed()) { notifyOutcome("Файл сохранён"); return; }
        new AlertDialog.Builder(this).setTitle("Файл сохранён")
                .setMessage("Открыть файл или передать его в другое приложение?")
                .setPositiveButton("Открыть", (dialog, which) -> openSavedFile(file, false))
                .setNeutralButton("Поделиться", (dialog, which) -> openSavedFile(file, true))
                .setNegativeButton("Закрыть", null).show();
    }

    private void openSavedFile(Uri file, boolean share) {
        String mime = getContentResolver().getType(file);
        if (mime == null) mime = "application/octet-stream";
        Intent action = share ? new Intent(Intent.ACTION_SEND).setType(mime).putExtra(Intent.EXTRA_STREAM, file)
                : new Intent(Intent.ACTION_VIEW).setDataAndType(file, mime);
        action.setClipData(ClipData.newUri(getContentResolver(), "Ouroboros", file));
        action.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try { startActivity(Intent.createChooser(action, share ? "Поделиться файлом" : "Открыть файл")); }
        catch (ActivityNotFoundException error) { notifyOutcome("Нет приложения для этого файла"); }
    }

    @Override protected void onSaveInstanceState(Bundle saved) {
        super.onSaveInstanceState(saved);
        saved.putString("pending_download", pendingDownload);
    }

    @Override protected void onResume() { super.onResume(); visible = true; check(false); }
    @Override protected void onPause() { visible = false; handler.removeCallbacks(refresh); super.onPause(); }
    private void goBack() { if (web != null && web.canGoBack()) web.goBack(); else finish(); }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    private void notifyOutcome(String message) {
        runOnUiThread(() -> Toast.makeText(getApplicationContext(), message, Toast.LENGTH_LONG).show());
    }
    @Override public void onBackPressed() { goBack(); }
    @Override protected void onDestroy() {
        visible = false; handler.removeCallbacks(refresh);
        if (chooser != null) chooser.onReceiveValue(null);
        for (PermissionPrompt prompt : new ArrayList<>(mediaPrompts.values())) {
            prompt.cancelled = true; prompt.finish.accept(false);
        }
        if (geolocationPrompt != null) {
            geolocationPrompt.cancelled = true; geolocationPrompt.finish.accept(false);
        }
        permissionPrompts.clear();
        if (activePermissionPrompt != null) activePermissionPrompt.cancelled = true;
        activePermissionPrompt = null;
        if (web != null) web.destroy();
        io.shutdown(); super.onDestroy();
    }
}
