"""Execute the actual CoreService with deterministic work queues and Android stubs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.serial

STUBS = {
    "android/Manifest.java": "package android; public final class Manifest { public static final class permission { public static final String ACCESS_BACKGROUND_LOCATION=\"android.permission.ACCESS_BACKGROUND_LOCATION\", ACCESS_COARSE_LOCATION=\"android.permission.ACCESS_COARSE_LOCATION\", ACCESS_FINE_LOCATION=\"android.permission.ACCESS_FINE_LOCATION\"; }}",
    "android/R.java": "package android; public final class R { public static class drawable { public static final int ic_menu_manage=1; }}",
    "android/content/Intent.java": """package android.content; public class Intent {
        private String action; public Intent() {} public Intent(Object c, Class<?> cls) {}
        public String getAction() { return action; }
        public Intent setAction(String value) { action=value; return this; }}""",
    "android/os/IBinder.java": "package android.os; public interface IBinder {}",
    "android/os/Looper.java": "package android.os; public class Looper {}",
    "android/os/Handler.java": "package android.os; public class Handler { public Handler(Looper l) {} }",
    "android/os/Build.java": "package android.os; public class Build { public static class VERSION { public static int SDK_INT=36; }}",
    "android/content/pm/PackageManager.java": "package android.content.pm; public class PackageManager { public static final int PERMISSION_GRANTED=0; }",
    "android/content/pm/ServiceInfo.java": "package android.content.pm; public class ServiceInfo { public static final int FOREGROUND_SERVICE_TYPE_LOCATION=8, FOREGROUND_SERVICE_TYPE_SPECIAL_USE=1073741824; }",
    "android/os/SystemClock.java": """package android.os; public class SystemClock {
        private static long now; public static long elapsedRealtime() { now+=120000; return now; }}""",
    "android/net/Network.java": "package android.net; public class Network {}",
    "android/net/LinkProperties.java": """package android.net; public class LinkProperties {
        public java.util.List<java.net.InetAddress> getDnsServers() { return java.util.Collections.emptyList(); }
        public boolean isPrivateDnsActive() { return false; }}""",
    "android/net/ConnectivityManager.java": """package android.net; public class ConnectivityManager {
        public static class NetworkCallback {
            public void onAvailable(Network n) {} public void onLost(Network n) {}
            public void onLinkPropertiesChanged(Network n, LinkProperties p) {} }
        public void registerDefaultNetworkCallback(NetworkCallback c, android.os.Handler h) {}
        public void unregisterNetworkCallback(NetworkCallback c) {}
        public Network getActiveNetwork() { return null; }
        public LinkProperties getLinkProperties(Network n) { return null; }}""",
    "android/util/Log.java": "package android.util; public class Log { public static int e(String t,String m,Throwable e) { return 0; }}",
    "android/app/PendingIntent.java": """package android.app; public class PendingIntent {
        public static final int FLAG_UPDATE_CURRENT=1, FLAG_IMMUTABLE=2;
        public static PendingIntent getActivity(Object c,int n,android.content.Intent i,int f) { return new PendingIntent(); }
        public static PendingIntent getService(Object c,int n,android.content.Intent i,int f) { return new PendingIntent(); }}""",
    "android/app/Notification.java": """package android.app; public class Notification {
        public String text; public static class Action { public static class Builder {
            public Builder(Object icon,String text,PendingIntent intent) {} public Action build() { return new Action(); }}}
        public static class Builder { private Notification n=new Notification();
            public Builder(Object c,String channel) {} public Builder setSmallIcon(int i) { return this; }
            public Builder setContentTitle(String t) { return this; }
            public Builder setContentText(String t) { n.text=t; return this; }
            public Builder setContentIntent(PendingIntent i) { return this; }
            public Builder addAction(Action a) { return this; } public Builder setOngoing(boolean b) { return this; }
            public Notification build() { return n; }}}""",
    "android/app/NotificationChannel.java": """package android.app; public class NotificationChannel {
        public NotificationChannel(String id,String name,int importance) {} }""",
    "android/app/NotificationManager.java": """package android.app; public class NotificationManager {
        public static final int IMPORTANCE_LOW=2; public final java.util.List<String> messages=new java.util.ArrayList<>();
        public void createNotificationChannel(NotificationChannel c) {}
        public void notify(int id,Notification n) { messages.add(n.text); }}""",
    "android/app/Service.java": """package android.app; public class Service {
        public static final int START_STICKY=1, START_NOT_STICKY=2, STOP_FOREGROUND_REMOVE=1, STOP_FOREGROUND_DETACH=2;
        public final NotificationManager notifications=new NotificationManager();
        public final java.util.Set<String> grants=new java.util.HashSet<>(); public int foregroundType;
        public int checkSelfPermission(String permission) { return grants.contains(permission) ? 0 : -1; }
        public <T> T getSystemService(Class<T> cls) { return cls.cast(cls==NotificationManager.class
            ? notifications : new android.net.ConnectivityManager()); }
        public android.os.Looper getMainLooper() { return new android.os.Looper(); }
        public void onCreate() {} public void onDestroy() {}
        public int onStartCommand(android.content.Intent i,int flags,int id) { return 0; }
        public void startForeground(int id,Notification n) { foregroundType=0; notifications.messages.add(n.text); }
        public void startForeground(int id,Notification n,int type) { foregroundType=type; notifications.messages.add(n.text); }
        public void stopForeground(int flags) {} public void stopSelf(int id) {}
        public android.os.IBinder onBind(android.content.Intent i) { return null; }}""",
    "org/json/JSONObject.java": """package org.json; public class JSONObject {
        public JSONObject put(String key,Object value) { return this; }}""",
    "ai/ouroboros/android/MainActivity.java": "package ai.ouroboros.android; public class MainActivity {}",
    "ai/ouroboros/android/AndroidBridge.java": """package ai.ouroboros.android; class AndroidBridge {
        static AndroidBridge start(Object c) throws java.io.IOException { return new AndroidBridge(); }
        void close() {} }""",
    "ai/ouroboros/android/RuntimeClient.java": """package ai.ouroboros.android; class RuntimeClient {
        static final java.util.List<String> calls=new java.util.ArrayList<>();
        static boolean stopped=true, healthFailure=false, controlFailure=false;
        static Runnable onHealth;
        static String startResult="running";
        static String dnsConfiguration(java.util.List<java.net.InetAddress> values) { return ""; }
        static void updateDns(String value) {}
        static String control(String action,String intent) throws Exception {
            calls.add(action+":"+intent);
            if (action.equals("start")) {
                if (controlFailure) throw new java.io.IOException("control-failure");
                if (intent.equals("automatic") && stopped) return "stopped";
                stopped=false; return startResult;
            } return "8765";
        }
        static org.json.JSONObject request(String path,String method) throws Exception {
            return request(path,method,new org.json.JSONObject());
        }
        static org.json.JSONObject request(String path,String method,org.json.JSONObject body) throws Exception {
            calls.add(path);
            if (path.equals("/api/command")) stopped=true;
            if (path.equals("/api/health") && onHealth!=null) { Runnable hook=onHealth; onHealth=null; hook.run(); }
            if (path.equals("/api/health") && healthFailure) throw new java.io.IOException("health-failure");
            return new org.json.JSONObject();
        }}""",
}

HARNESS = """package ai.ouroboros.android;
import android.content.Intent;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.*;
public class LifecycleTest {
    static final String PENDING="Запуск запрошен. Готовность ядра пока не подтверждена; проверьте статус позже.";
    static class Queue extends AbstractExecutorService {
        final ArrayDeque<Runnable> items=new ArrayDeque<>(); boolean closed;
        public void execute(Runnable r) { items.add(r); }
        public void shutdown() { closed=true; }
        public List<Runnable> shutdownNow() { closed=true; items.clear(); return Collections.emptyList(); }
        public boolean isShutdown() { return closed; } public boolean isTerminated() { return closed; }
        public boolean awaitTermination(long t,TimeUnit u) { return closed; }
        void drain() { while (!items.isEmpty()) items.remove().run(); }
    }
    static void require(boolean value,String detail) {
        if (!value) throw new AssertionError(detail+"; calls="+RuntimeClient.calls);
    }
    static void inject(CoreService service,String name,Queue queue) throws Exception {
        Field field=CoreService.class.getDeclaredField(name); field.setAccessible(true);
        ((ExecutorService)field.get(service)).shutdownNow(); field.set(service,queue);
    }
    static void send(CoreService service,String action,int id) {
        int result=service.onStartCommand(action==null ? null : new Intent().setAction(action),0,id);
        require(result==("panic".equals(action) ? 2 : 1),"sticky return differs");
    }
    public static void main(String[] args) throws Exception {
        CoreService service=new CoreService(); Queue work=new Queue(), urgent=new Queue();
        inject(service,"work",work); inject(service,"urgent",urgent);
        service.onCreate(); work.drain(); service.notifications.messages.clear();
        String scenario=args[0];
        try {
            if (scenario.equals("panic_during_start_observation")) {
                RuntimeClient.startResult="starting"; RuntimeClient.healthFailure=true;
                RuntimeClient.onHealth=()-> { send(service,"panic",2); urgent.drain(); };
                send(service,"start",1); work.drain();
                require(RuntimeClient.stopped && RuntimeClient.calls.contains("/api/command"),"in-flight Panic lost");
                require(service.notifications.messages.stream().noneMatch(m->m.startsWith(PENDING)
                    || m.startsWith("Действие не выполнено")),"stale start outcome overwrote Panic");
            } else if (scenario.equals("starting_pending")) {
                RuntimeClient.startResult="starting"; RuntimeClient.healthFailure=true;
                send(service,"start",1); work.drain();
                require(service.notifications.messages.stream().anyMatch(m->m.startsWith(PENDING)),"missing unconfirmed readiness");
                require(service.notifications.messages.stream().noneMatch(m->m.startsWith("Действие не выполнено")),"start was falsely reported failed");
                require(Collections.frequency(RuntimeClient.calls,"start:owner")==1,"start was repeated");
            } else if (scenario.equals("health_failure") || scenario.equals("control_failure")) {
                RuntimeClient.healthFailure=scenario.equals("health_failure");
                RuntimeClient.controlFailure=scenario.equals("control_failure");
                send(service,RuntimeClient.healthFailure ? "status" : "start",1); work.drain();
                String cause=RuntimeClient.healthFailure ? "health-failure" : "control-failure";
                require(service.notifications.messages.stream().anyMatch(m->m.startsWith("Действие не выполнено") && m.contains(cause)),"ordinary failure hidden");
                require(service.notifications.messages.stream().noneMatch(m->m.startsWith(PENDING)),"ordinary failure became readiness pending");
            } else if (scenario.equals("sticky_status")) {
                send(service,null,1); work.drain();
                require(RuntimeClient.calls.stream().noneMatch(c->c.startsWith("start:")),"sticky restoration replayed Start");
            } else if (scenario.equals("foreground_special_only") || scenario.equals("foreground_location")) {
                if (scenario.equals("foreground_location")) {
                    service.grants.add("android.permission.ACCESS_BACKGROUND_LOCATION");
                    service.grants.add("android.permission.ACCESS_COARSE_LOCATION");
                }
                send(service,"status",1); work.drain();
                int special=android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE;
                require((service.foregroundType & special) != 0,"special-use foreground type missing");
                require(((service.foregroundType & android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION) != 0)
                    == scenario.equals("foreground_location"),"location foreground type did not follow background grant");
            } else {
                String[] actions=scenario.split("_"); send(service,actions[0],1); send(service,actions[1],2);
                urgent.drain(); work.drain();
                if (scenario.equals("start_boot") || scenario.equals("boot_start") || scenario.equals("panic_start")) {
                    require(Collections.frequency(RuntimeClient.calls,"start:owner")==1,"owner Start lost or duplicated");
                    require(!RuntimeClient.stopped,"owner Start did not resume");
                    require(!RuntimeClient.calls.contains("/api/command"),"superseded Panic still executed");
                } else {
                    require(RuntimeClient.stopped,"Panic allowed automatic resume");
                    require(!RuntimeClient.calls.contains("start:owner"),"queued Start survived Panic");
                    require(Collections.frequency(RuntimeClient.calls,"/api/command")==1,"Panic lost or duplicated");
                }
            }
            System.out.println("PASS "+scenario+" calls="+RuntimeClient.calls);
        } finally { service.onDestroy(); work.drain(); urgent.drain(); }
    }
}
"""


@pytest.fixture(scope="module")
def lifecycle_java(tmp_path_factory):
    output = tmp_path_factory.mktemp("android-lifecycle-java")
    java_home = os.environ.get("JAVA_HOME")
    javac = str(Path(java_home) / "bin/javac") if java_home else shutil.which("javac")
    java = str(Path(java_home) / "bin/java") if java_home else shutil.which("java")
    if not javac or not java:
        pytest.skip("The service execution check requires the build JDK")
    for name, text in STUBS.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    package = output / "ai/ouroboros/android"
    source = Path(__file__).resolve().parents[1] / "host/src/ai/ouroboros/android/CoreService.java"
    shutil.copy2(source, package / source.name)
    (package / "LifecycleTest.java").write_text(HARNESS, encoding="utf-8")
    compiled = subprocess.run([javac, "-encoding", "UTF-8", "-d", str(output),
        *map(str, output.rglob("*.java"))], capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    def invoke(scenario):
        result = subprocess.run([java, "-Dfile.encoding=UTF-8", "-cp", str(output),
            "ai.ouroboros.android.LifecycleTest", scenario], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.startswith("PASS " + scenario), result.stdout

    return invoke


@pytest.mark.parametrize("scenario", ["start_boot", "boot_start", "start_panic", "panic_boot",
                                    "panic_start", "sticky_status"])
def test_service_orders_owner_and_automatic_intents(lifecycle_java, scenario):
    lifecycle_java(scenario)


@pytest.mark.parametrize("scenario", ["starting_pending", "health_failure", "control_failure",
                                    "panic_during_start_observation"])
def test_start_observation_is_distinct_from_action_failure(lifecycle_java, scenario):
    lifecycle_java(scenario)


@pytest.mark.parametrize("scenario", ["foreground_special_only", "foreground_location"])
def test_foreground_service_type_discloses_background_location_grant(lifecycle_java, scenario):
    lifecycle_java(scenario)
