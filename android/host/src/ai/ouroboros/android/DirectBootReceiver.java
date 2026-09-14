package ai.ouroboros.android;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/** Direct Boot marker only; the credential-encrypted core starts after unlock. */
public final class DirectBootReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        Context direct = context.createDeviceProtectedStorageContext();
        direct.getSharedPreferences("boot", Context.MODE_PRIVATE).edit()
                .putLong("last_locked_boot_ms", System.currentTimeMillis()).apply();
    }
}
