"""Execute PackageInstallReceiver against small Android API stubs.

These tests deliberately run the receiver's Java code.  They check the
boundary that is easy to get wrong in a manifest-only test: a pending Android
confirmation must remain a user-owned PendingIntent, while terminal callbacks
must close the old notification and mark the durable receipt complete.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.serial


STUBS = {
    "android/R.java": "package android; public final class R { public static class drawable { public static final int stat_sys_download_done=7; }}",
    "android/content/BroadcastReceiver.java": "package android.content; public abstract class BroadcastReceiver { public abstract void onReceive(Context c, Intent i); }",
    "android/content/Intent.java": r'''package android.content;
import java.util.HashMap; import java.util.Map;
public class Intent {
  public static final String EXTRA_INTENT = "android.intent.extra.INTENT";
  private String action; private final Map<String,Object> extras = new HashMap<>();
  public Intent() {} public Intent(Context c, Class<?> cls) {}
  public String getAction() { return action; }
  public Intent setAction(String value) { action=value; return this; }
  public Intent putExtra(String key, String value) { extras.put(key,value); return this; }
  public Intent putExtra(String key, int value) { extras.put(key,value); return this; }
  public Intent putExtra(String key, Intent value) { extras.put(key,value); return this; }
  public String getStringExtra(String key) { Object value=extras.get(key); return value instanceof String ? (String)value : null; }
  public int getIntExtra(String key, int fallback) { Object value=extras.get(key); return value instanceof Integer ? (Integer)value : fallback; }
  @SuppressWarnings("unchecked") public <T> T getParcelableExtra(String key) { return (T)extras.get(key); }
}''',
    "android/content/SharedPreferences.java": r'''package android.content;
public interface SharedPreferences {
  String getString(String key, String fallback);
  Editor edit();
  interface Editor { Editor putString(String key, String value); void apply(); }
}''',
    "android/content/Context.java": r'''package android.content;
import android.app.NotificationManager; import java.util.HashMap; import java.util.Map;
public class Context {
  public static final int MODE_PRIVATE=0; final Map<String,Prefs> prefs=new HashMap<>();
  public final NotificationManager notifications=new NotificationManager(); public int startedActivities;
  public <T> T getSystemService(Class<T> type) { return type.cast(type==NotificationManager.class ? notifications : null); }
  public SharedPreferences getSharedPreferences(String name,int mode) { return prefs.computeIfAbsent(name,k->new Prefs()); }
  static final class Prefs implements SharedPreferences {
    final Map<String,String> values=new HashMap<>(); public String getString(String k,String f){return values.getOrDefault(k,f);}
    public Editor edit(){ return new Editor(){ public Editor putString(String k,String v){values.put(k,v);return this;} public void apply(){} }; }
  }
}''',
    "android/app/BroadcastReceiver.java": "package android.app; public abstract class BroadcastReceiver extends android.content.BroadcastReceiver {}",
    "android/app/PendingIntent.java": r'''package android.app;
import android.content.Context; import android.content.Intent;
public class PendingIntent {
  public static final int FLAG_UPDATE_CURRENT=1, FLAG_IMMUTABLE=2; public final Intent intent; public final int requestCode;
  private PendingIntent(Intent i,int n){intent=i;requestCode=n;}
  public static PendingIntent getActivity(Context c,int n,Intent i,int f){return new PendingIntent(i,n);}
}''',
    "android/app/Notification.java": r'''package android.app;
public class Notification {
  public static final int VISIBILITY_PRIVATE=1; public String title,text; public PendingIntent contentIntent;
  public static class Builder { final Notification n=new Notification(); public Builder(android.content.Context c,String channel){}
    public Builder setSmallIcon(int i){return this;} public Builder setContentTitle(String s){n.title=s;return this;}
    public Builder setContentText(String s){n.text=s;return this;} public Builder setVisibility(int v){return this;}
    public Builder setContentIntent(PendingIntent p){n.contentIntent=p;return this;} public Builder setAutoCancel(boolean b){return this;}
    public Notification build(){return n;}
  }
}''',
    "android/app/NotificationChannel.java": "package android.app; public class NotificationChannel { public NotificationChannel(String id,String name,int importance){} }",
    "android/app/NotificationManager.java": r'''package android.app;
import java.util.HashMap; import java.util.Map;
public class NotificationManager {
  public static final int IMPORTANCE_DEFAULT=3; public final Map<String,Notification> active=new HashMap<>(); public int cancels; public boolean enabled=true; public int channels;
  public void createNotificationChannel(NotificationChannel c){channels++;}
  public void notify(String tag,int id,Notification n){active.put(tag+"/"+id,n);}
  public void cancel(String tag,int id){cancels++; active.remove(tag+"/"+id);}
  public boolean areNotificationsEnabled(){return enabled;}
}''',
    "android/content/pm/PackageInstaller.java": r'''package android.content.pm;
public final class PackageInstaller { public static final String EXTRA_STATUS="android.content.pm.extra.STATUS", EXTRA_STATUS_MESSAGE="android.content.pm.extra.STATUS_MESSAGE", EXTRA_PACKAGE_NAME="android.content.pm.extra.PACKAGE_NAME"; public static final int STATUS_SUCCESS=0, STATUS_PENDING_USER_ACTION=-1; }''',
    "android/util/Log.java": "package android.util; public class Log { public static int e(String t,String m,Throwable e){return 0;} }",
    "org/json/JSONObject.java": r'''package org.json;
import java.util.LinkedHashMap; import java.util.Map;
public class JSONObject {
  final Map<String,Object> values=new LinkedHashMap<>();
  public JSONObject(){}
  public JSONObject(String encoded){ for(String part:encoded.split("\\|",-1)){int i=part.indexOf('='); if(i>0) values.put(part.substring(0,i),part.substring(i+1));} }
  public JSONObject put(String k,Object v){values.put(k,v);return this;}
  public int getInt(String k){Object v=values.get(k); return v instanceof Integer ? (Integer)v : Integer.parseInt(String.valueOf(v));}
  public String toString(){StringBuilder b=new StringBuilder(); for(Map.Entry<String,Object> e:values.entrySet()){if(b.length()>0)b.append('|'); b.append(e.getKey()).append('=').append(e.getValue()==null?"null":e.getValue());} return b.toString();}
}''',
}


HARNESS = r'''package ai.ouroboros.android;
import android.app.*; import android.content.*; import android.content.pm.PackageInstaller; import java.util.*;
public class PackageInstallCallbackTest {
  static final String KEY="install-key"; static final int SESSION=41; static final String TAG="ouroboros_package_installs/41";
  static void require(boolean ok,String message){if(!ok) throw new AssertionError(message);}
  static Context context(String receipt){ Context c=new Context(); c.getSharedPreferences("package_install_receipts",0).edit().putString(KEY,receipt).apply(); return c; }
  static Intent callback(int status){ return callback(status,SESSION); }
  static Intent callback(int status,int session){ return new Intent().setAction(PackageInstallReceiver.ACTION).putExtra(PackageInstallReceiver.KEY,KEY).putExtra(PackageInstallReceiver.SESSION,session).putExtra(PackageInstaller.EXTRA_STATUS,status).putExtra(PackageInstaller.EXTRA_PACKAGE_NAME,"org.example.app"); }
  static String receipt(Context c){return c.getSharedPreferences("package_install_receipts",0).getString(KEY,null);}
  public static void main(String[] args){
    String scenario=args[0]; PackageInstallReceiver receiver=new PackageInstallReceiver();
    if(scenario.equals("pending_confirmation")){
      Context c=context("session_id=41"); Intent confirmation=new Intent().setAction("android.intent.action.CONFIRM_INSTALL");
      receiver.onReceive(c,callback(PackageInstaller.STATUS_PENDING_USER_ACTION).putExtra(Intent.EXTRA_INTENT,confirmation));
      Notification n=c.notifications.active.get(TAG);
      require(n!=null,"pending callback did not retain a notification"); require(n.contentIntent!=null,"notification lost confirmation PendingIntent");
      require(n.contentIntent.intent==confirmation,"notification PendingIntent did not carry Android confirmation Intent");
      require(c.startedActivities==0,"receiver launched UI in the background"); require(receipt(c).contains("completion_observed=false"),"pending was marked terminal");
      require(receipt(c).contains("confirmation_delivery=notification"),"pending delivery was not recorded");
    } else if(scenario.equals("terminal_success") || scenario.equals("terminal_failure")){
      Context c=context("session_id=41"); Intent pending=callback(PackageInstaller.STATUS_PENDING_USER_ACTION).putExtra(Intent.EXTRA_INTENT,new Intent()); receiver.onReceive(c,pending);
      require(c.notifications.active.containsKey(TAG),"setup pending notification missing");
      int status=scenario.equals("terminal_success") ? PackageInstaller.STATUS_SUCCESS : 9;
      receiver.onReceive(c,callback(status));
      require(!c.notifications.active.containsKey(TAG),"terminal callback left stale notification"); require(c.notifications.cancels==1,"terminal callback did not cancel exactly once");
      require(receipt(c).contains("completion_observed=true"),"terminal callback was not durable"); require(receipt(c).contains("outcome="+(status==0?"success":"failure")),"terminal outcome missing");
    } else if(scenario.equals("unknown_or_mismatch")){
      Context c=context("session_id=41"); receiver.onReceive(c,callback(PackageInstaller.STATUS_SUCCESS,99));
      require(receipt(c).equals("session_id=41"),"mismatched session mutated receipt"); require(c.notifications.active.isEmpty(),"mismatched callback published notification"); require(c.notifications.cancels==0,"mismatched callback cancelled notification");
      Intent unknown=callback(PackageInstaller.STATUS_SUCCESS).putExtra(PackageInstallReceiver.KEY,"missing-key"); receiver.onReceive(c,unknown);
      require(c.notifications.cancels==0,"unknown callback changed notification state");
    } else throw new AssertionError("unknown scenario "+scenario);
    System.out.println("PASS "+scenario);
  }
}'''


@pytest.fixture(scope="module")
def receiver_java(tmp_path_factory):
    output = tmp_path_factory.mktemp("android-package-install-java")
    java_home = os.environ.get("JAVA_HOME")
    javac = str(Path(java_home) / "bin/javac") if java_home else shutil.which("javac")
    java = str(Path(java_home) / "bin/java") if java_home else shutil.which("java")
    if not javac or not java:
        pytest.skip("The receiver execution check requires the build JDK")
    for name, text in STUBS.items():
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "host/src/ai/ouroboros/android/PackageInstallReceiver.java"
    target = output / "ai/ouroboros/android/PackageInstallReceiver.java"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    (output / "ai/ouroboros/android/PackageInstallCallbackTest.java").write_text(HARNESS, encoding="utf-8")
    compiled = subprocess.run([javac,"-encoding","UTF-8","-d",str(output),*map(str,output.rglob("*.java"))],capture_output=True,text=True,timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    def invoke(scenario):
        result=subprocess.run([java,"-Dfile.encoding=UTF-8","-cp",str(output),"ai.ouroboros.android.PackageInstallCallbackTest",scenario],capture_output=True,text=True,timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "PASS "+scenario, result.stdout
    return invoke


@pytest.mark.parametrize("scenario", ["pending_confirmation", "terminal_success", "terminal_failure", "unknown_or_mismatch"])
def test_package_install_receiver_callback_contract(receiver_java, scenario):
    receiver_java(scenario)
