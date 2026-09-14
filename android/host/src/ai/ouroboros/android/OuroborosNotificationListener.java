package ai.ouroboros.android;

import android.app.Notification;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;
import org.json.JSONArray;
import org.json.JSONObject;

/** Optional user-enabled notification readback and action surface. */
public final class OuroborosNotificationListener extends NotificationListenerService {
    private static volatile OuroborosNotificationListener instance;
    @Override public void onListenerConnected() { instance = this; }
    @Override public void onDestroy() { if (instance == this) instance = null; super.onDestroy(); }

    static JSONObject state() throws Exception {
        return new JSONObject().put("enabled", instance != null)
                .put("coverage", "user_enabled_notification_listener");
    }

    static JSONObject list(JSONObject params) throws Exception {
        OuroborosNotificationListener service = instance;
        JSONArray rows = new JSONArray();
        boolean includeText = params.optBoolean("include_text", false);
        if (service != null) for (StatusBarNotification item : service.getActiveNotifications()) {
            Notification notification = item.getNotification();
            JSONObject row = new JSONObject().put("package", item.getPackageName())
                    .put("key", item.getKey()).put("post_time", item.getPostTime());
            if (includeText && notification != null && notification.extras != null) {
                CharSequence title = notification.extras.getCharSequence(Notification.EXTRA_TITLE);
                CharSequence text = notification.extras.getCharSequence(Notification.EXTRA_TEXT);
                row.put("title", title == null ? JSONObject.NULL : title.toString())
                        .put("text", text == null ? JSONObject.NULL : text.toString());
            }
            rows.put(row);
        }
        return new JSONObject().put("enabled", service != null).put("rows", rows)
                .put("coverage", includeText ? "active_notifications_requested_text" : "active_notification_metadata");
    }
}
