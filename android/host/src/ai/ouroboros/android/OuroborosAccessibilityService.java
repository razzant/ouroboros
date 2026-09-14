package ai.ouroboros.android;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityWindowInfo;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONArray;
import org.json.JSONObject;

/** Optional, owner-enabled generic UI surface; Android consent remains required. */
public final class OuroborosAccessibilityService extends AccessibilityService {
    private static final int DEFAULT_MAX_WINDOWS = 32;
    private static final int DEFAULT_MAX_NODES = 256;
    private static final int MAX_MAX_NODES = 2048;
    private static final int DEFAULT_MAX_DEPTH = 16;
    private static final int MAX_MAX_DEPTH = 32;
    private static final int MAX_TEXT_CHARS = 512;
    private static final long GESTURE_TIMEOUT_MS = 10000;
    private static volatile OuroborosAccessibilityService instance;

    @Override public void onServiceConnected() {
        instance = this;
        super.onServiceConnected();
    }

    @Override public void onAccessibilityEvent(android.view.accessibility.AccessibilityEvent event) { }
    @Override public void onInterrupt() { }

    @Override public void onDestroy() {
        if (instance == this) instance = null;
        super.onDestroy();
    }

    static JSONObject state() throws Exception {
        return new JSONObject().put("enabled", instance != null)
                .put("coverage", "user_enabled_accessibility_service")
                .put("interactive_windows", true).put("node_actions", "click,set_text,scroll")
                .put("gestures", true);
    }

    /** Backward-compatible default snapshot for callers without request parameters. */
    static JSONObject windows() throws Exception { return windows(new JSONObject()); }

    /** Return a bounded point-in-time tree. node_address is valid only for this snapshot. */
    static JSONObject windows(JSONObject params) throws Exception {
        OuroborosAccessibilityService service = instance;
        if (service == null) return new JSONObject().put("enabled", false).put("windows", new JSONArray())
                .put("coverage", "bounded_interactive_window_tree")
                .put("reason", "accessibility_not_enabled");
        int maxWindows = bounded(params.optInt("max_windows", DEFAULT_MAX_WINDOWS), 1, DEFAULT_MAX_WINDOWS);
        int maxNodes = bounded(params.optInt("max_nodes", DEFAULT_MAX_NODES), 1, MAX_MAX_NODES);
        int maxDepth = bounded(params.optInt("max_depth", DEFAULT_MAX_DEPTH), 0, MAX_MAX_DEPTH);
        SnapshotBudget budget = new SnapshotBudget(maxNodes);
        JSONArray rows = new JSONArray();
        List<AccessibilityWindowInfo> windows = service.getWindows();
        if (windows == null) return new JSONObject().put("enabled", true).put("windows", rows)
                .put("coverage", "bounded_interactive_window_tree").put("reason", "no_windows");
        int index = 0;
        for (AccessibilityWindowInfo window : windows) {
            if (index >= maxWindows) break;
            if (window == null) { index++; continue; }
            JSONObject row = new JSONObject().put("id", window.getId()).put("index", index)
                    .put("active", window.isActive()).put("focused", window.isFocused());
            AccessibilityNodeInfo root = null;
            try {
                root = window.getRoot();
                if (root == null) {
                    row.put("reason", "window_root_unavailable");
                } else {
                    row.put("package", text(root.getPackageName())).put("class", text(root.getClassName()));
                    appendNode(row, root, "w" + window.getId() + "/0", 0, maxDepth, budget);
                }
            } catch (RuntimeException error) {
                row.put("reason", "window_snapshot_failed");
            } finally {
                if (root != null) root.recycle();
            }
            rows.put(row);
            index++;
            if (budget.truncated) break;
        }
        JSONObject result = new JSONObject().put("enabled", true).put("windows", rows)
                .put("coverage", "bounded_interactive_window_tree").put("max_nodes", maxNodes)
                .put("max_depth", maxDepth).put("max_windows", maxWindows).put("nodes", budget.count);
        if (budget.truncated) result.put("truncated", true).put("truncation_reason", budget.reason);
        else if (windows.size() > maxWindows) result.put("truncated", true).put("truncation_reason", "max_windows");
        return result;
    }

    static JSONObject perform(JSONObject params) throws Exception {
        OuroborosAccessibilityService service = instance;
        if (service == null) return new JSONObject().put("enabled", false).put("performed", false)
                .put("reason", "accessibility_not_enabled");
        String action = params.optString("action", "");
        boolean performed;
        switch (action) {
            case "back": performed = service.performGlobalAction(GLOBAL_ACTION_BACK); break;
            case "home": performed = service.performGlobalAction(GLOBAL_ACTION_HOME); break;
            case "notifications": performed = service.performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS); break;
            case "quick_settings": performed = service.performGlobalAction(GLOBAL_ACTION_QUICK_SETTINGS); break;
            case "click":
            case "set_text":
            case "scroll": {
                AccessibilityNodeInfo node = service.resolveNode(params.optString("node_address", ""));
                if (node == null) return stale(action);
                try {
                    if ("click".equals(action)) performed = node.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    else if ("set_text".equals(action)) {
                        Bundle args = new Bundle();
                        args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                                params.optString("text", ""));
                        performed = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
                    } else {
                        String direction = params.optString("direction", "forward");
                        int nodeAction = "backward".equals(direction)
                                ? AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                                : AccessibilityNodeInfo.ACTION_SCROLL_FORWARD;
                        performed = node.performAction(nodeAction);
                    }
                } finally { node.recycle(); }
                break;
            }
            case "click_text":
                performed = clickText(service.getRootInActiveWindow(), params.getString("text"));
                break;
            case "gesture":
                return service.gesture(params);
            default: throw new IllegalArgumentException("Unsupported accessibility action: " + action);
        }
        return new JSONObject().put("enabled", true).put("performed", performed)
                .put("action", action).put("coverage", "owner_requested_ui_action");
    }

    private static JSONObject stale(String action) throws Exception {
        return new JSONObject().put("enabled", true).put("performed", false).put("action", action)
                .put("reason", "stale_or_missing_node_address").put("outcome", "not_performed");
    }

    private AccessibilityNodeInfo resolveNode(String address) {
        if (address == null || !address.startsWith("w")) return null;
        int slash = address.indexOf('/');
        if (slash <= 1) return null;
        int windowId;
        try { windowId = Integer.parseInt(address.substring(1, slash)); }
        catch (NumberFormatException error) { return null; }
        AccessibilityNodeInfo current = null;
        List<AccessibilityWindowInfo> windows = getWindows();
        if (windows == null) return null;
        for (AccessibilityWindowInfo window : windows) {
            if (window != null && window.getId() == windowId) { current = window.getRoot(); break; }
        }
        if (current == null) return null;
        String path = address.substring(slash + 1);
        if (path.isEmpty() || "0".equals(path)) return current;
        String[] parts = path.split("/");
        // The snapshot's root is address .../0; consume it before walking children.
        int start = "0".equals(parts[0]) ? 1 : 0;
        try {
            for (int i = start; i < parts.length; i++) {
                int childIndex = Integer.parseInt(parts[i]);
                AccessibilityNodeInfo child = current.getChild(childIndex);
                current.recycle();
                current = child;
                if (current == null) return null;
            }
            return current;
        } catch (RuntimeException error) {
            if (current != null) current.recycle();
            return null;
        }
    }

    private JSONObject gesture(JSONObject params) throws Exception {
        String type = params.optString("gesture", "tap");
        if (!"tap".equals(type) && !"swipe".equals(type))
            throw new IllegalArgumentException("Unsupported accessibility gesture: " + type);
        float x = (float) params.getDouble("x");
        float y = (float) params.getDouble("y");
        float x2 = (float) params.optDouble("x2", x);
        float y2 = (float) params.optDouble("y2", y);
        long duration = Math.max(1, Math.min(5000, params.optLong("duration_ms", "swipe".equals(type) ? 350 : 1)));
        Path path = new Path();
        path.moveTo(x, y);
        path.lineTo("swipe".equals(type) ? x2 : x, "swipe".equals(type) ? y2 : y);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0, duration)).build();
        CountDownLatch done = new CountDownLatch(1);
        AtomicBoolean dispatched = new AtomicBoolean(false);
        AtomicReference<Boolean> result = new AtomicReference<>();
        Handler main = new Handler(Looper.getMainLooper());
        Runnable dispatch = () -> {
            boolean accepted = dispatchGesture(gesture, new GestureResult(done, dispatched, result), null);
            dispatched.set(accepted);
            if (!accepted) done.countDown();
        };
        main.post(dispatch);
        boolean completed = done.await(GESTURE_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        if (!completed && !dispatched.get()) main.removeCallbacks(dispatch);
        JSONObject response = new JSONObject().put("enabled", true).put("action", "gesture")
                .put("gesture", type).put("performed", completed && Boolean.TRUE.equals(result.get()))
                .put("coverage", "owner_requested_ui_action");
        if (!completed) response.put("reason", "gesture_result_timeout").put("outcome", "unknown");
        else if (!dispatched.get()) response.put("reason", "gesture_rejected");
        return response;
    }

    private static final class GestureResult extends AccessibilityService.GestureResultCallback {
        private final CountDownLatch done;
        private final AtomicBoolean dispatched;
        private final AtomicReference<Boolean> result;
        GestureResult(CountDownLatch done, AtomicBoolean dispatched, AtomicReference<Boolean> result) {
            this.done = done; this.dispatched = dispatched; this.result = result;
        }
        @Override public void onCompleted(GestureDescription gestureDescription) {
            result.set(true); done.countDown();
        }
        @Override public void onCancelled(GestureDescription gestureDescription) {
            result.set(false); done.countDown();
        }
    }

    private static void appendNode(JSONObject row, AccessibilityNodeInfo node, String address, int depth,
                                   int maxDepth, SnapshotBudget budget) throws Exception {
        if (budget.exhausted()) { budget.truncated = true; budget.reason = "max_nodes"; return; }
        budget.count++;
        row.put("node_address", address).put("depth", depth).put("class", text(node.getClassName()))
                .put("package", text(node.getPackageName())).put("text", boundedText(node.getText()))
                .put("content_description", boundedText(node.getContentDescription()))
                .put("view_id", boundedText(node.getViewIdResourceName()))
                .put("clickable", node.isClickable()).put("enabled", node.isEnabled())
                .put("focusable", node.isFocusable()).put("scrollable", node.isScrollable())
                .put("editable", node.isEditable()).put("visible", node.isVisibleToUser());
        Rect bounds = new Rect();
        node.getBoundsInScreen(bounds);
        row.put("bounds", new JSONObject().put("left", bounds.left).put("top", bounds.top)
                .put("right", bounds.right).put("bottom", bounds.bottom));
        if (depth >= maxDepth || budget.exhausted()) {
            if (node.getChildCount() > 0) {
                budget.truncated = true;
                budget.reason = budget.exhausted() ? "max_nodes" : "max_depth";
                row.put("children_truncated", true);
            }
            return;
        }
        JSONArray children = new JSONArray();
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child == null) continue;
            try {
                JSONObject childRow = new JSONObject();
                appendNode(childRow, child, address + "/" + i, depth + 1, maxDepth, budget);
                children.put(childRow);
            } finally { child.recycle(); }
            if (budget.exhausted()) {
                if (i + 1 < node.getChildCount()) {
                    budget.truncated = true;
                    budget.reason = "max_nodes";
                }
                break;
            }
        }
        if (children.length() > 0) row.put("children", children);
    }

    private static boolean clickText(AccessibilityNodeInfo node, String text) {
        if (node == null) return false;
        try {
            List<AccessibilityNodeInfo> matches = node.findAccessibilityNodeInfosByText(text);
            if (matches != null) for (AccessibilityNodeInfo match : matches) {
                try { if (match != null && match.isClickable() && match.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true; }
                finally { if (match != null) match.recycle(); }
            }
            return false;
        } finally { node.recycle(); }
    }

    private static int bounded(int value, int min, int max) { return Math.max(min, Math.min(max, value)); }
    private static String text(CharSequence value) { return value == null ? "" : value.toString(); }
    private static String boundedText(CharSequence value) {
        String result = text(value);
        return result.length() <= MAX_TEXT_CHARS ? result : result.substring(0, MAX_TEXT_CHARS);
    }
    private static final class SnapshotBudget {
        final int max; int count; boolean truncated; String reason = "max_nodes";
        SnapshotBudget(int max) { this.max = max; }
        boolean exhausted() { return count >= max; }
    }
}
