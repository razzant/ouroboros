package ai.ouroboros.android;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.pm.PackageInstaller;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import org.json.JSONObject;

/** Records PackageInstaller completion without claiming that commit submission is success. */
public final class PackageInstallReceiver extends BroadcastReceiver {
    static final String ACTION = "ai.ouroboros.android.PACKAGE_INSTALL_RESULT";
    static final String KEY = "idempotency_key";
    static final String SESSION = "session_id";
    private static final String CHANNEL = "ouroboros_package_installs";

    static void cancelNotification(Context context, int sessionId) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager != null) manager.cancel(CHANNEL, sessionId);
    }

    @Override public void onReceive(Context context, Intent intent) {
        if (!ACTION.equals(intent.getAction())) return;
        String key = intent.getStringExtra(KEY);
        if (key == null || key.isEmpty()) return;
        try {
            android.content.SharedPreferences prefs = context.getSharedPreferences(
                    "package_install_receipts", Context.MODE_PRIVATE);
            String previous = prefs.getString(key, null);
            if (previous == null) return;
            JSONObject receipt = new JSONObject(previous);
            int status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, Integer.MIN_VALUE);
            if (status == Integer.MIN_VALUE) return;
            int sessionId = receipt.getInt(SESSION);
            if (sessionId != intent.getIntExtra(SESSION, -1)) return;
            boolean pending = status == PackageInstaller.STATUS_PENDING_USER_ACTION;
            String statusName;
            if (status == PackageInstaller.STATUS_SUCCESS) statusName = "success";
            else if (pending) statusName = "pending_user_action";
            else statusName = "failure";
            receipt.put("status", statusName).put("status_code", status)
                    .put("completion_observed", !pending).put("outcome", statusName)
                    .put("status_message", intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE));
            String packageName = intent.getStringExtra(PackageInstaller.EXTRA_PACKAGE_NAME);
            if (packageName != null) receipt.put("package", packageName);
            // Record the platform result even when Android refuses notification delivery.
            prefs.edit().putString(key, receipt.toString()).apply();
            NotificationManager manager = context.getSystemService(NotificationManager.class);
            if (pending) {
                // Keep Android's confirmation alive in a system-owned PendingIntent. A click on
                // the notification opens it; a background receiver never forces an Activity open.
                Intent confirmation = intent.getParcelableExtra(Intent.EXTRA_INTENT);
                if (confirmation != null && manager != null) {
                    manager.createNotificationChannel(new NotificationChannel(CHANNEL,
                            "Ouroboros app installations", NotificationManager.IMPORTANCE_DEFAULT));
                    PendingIntent action = PendingIntent.getActivity(context, sessionId, confirmation,
                            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
                    Notification notification = new Notification.Builder(context, CHANNEL)
                            .setSmallIcon(android.R.drawable.stat_sys_download_done)
                            .setContentTitle("Confirm app installation")
                            .setContentText("Android needs your confirmation to continue.")
                            .setVisibility(Notification.VISIBILITY_PRIVATE)
                            .setContentIntent(action).setAutoCancel(true).build();
                    try {
                        manager.notify(CHANNEL, sessionId, notification);
                        receipt.put("confirmation_delivery", manager.areNotificationsEnabled()
                                ? "notification" : "notifications_disabled");
                    } catch (RuntimeException error) {
                        receipt.put("confirmation_delivery", "notification_failed");
                    }
                } else receipt.put("confirmation_delivery", "unavailable");
            } else cancelNotification(context, sessionId);
            prefs.edit().putString(key, receipt.toString()).apply();
        } catch (Exception error) {
            android.util.Log.e("OuroborosHost", "Package install result could not be recorded", error);
        }
    }
}
