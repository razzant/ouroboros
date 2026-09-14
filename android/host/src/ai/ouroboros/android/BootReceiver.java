package ai.ouroboros.android;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/** Credential-encrypted data is available after the first unlock. */
public final class BootReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        if (Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction()))
            context.startForegroundService(new Intent(context, CoreService.class).setAction("boot"));
        else if (Intent.ACTION_MY_PACKAGE_REPLACED.equals(intent.getAction()))
            context.startForegroundService(new Intent(context, CoreService.class).setAction("status"));
    }
}
