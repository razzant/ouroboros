package ai.ouroboros.android;

import android.service.quicksettings.Tile;
import android.service.quicksettings.TileService;

/** Safe locked-screen entry point for a core status/action notification. */
public final class OuroborosQuickSettingsTile extends TileService {
    @Override public void onStartListening() { update(); }
    @Override public void onClick() {
        super.onClick();
        startForegroundService(new android.content.Intent(this, CoreService.class).setAction("status"));
        update();
    }
    private void update() {
        Tile tile = getQsTile();
        if (tile == null) return;
        tile.setLabel("Ouroboros"); tile.setState(Tile.STATE_ACTIVE); tile.updateTile();
    }
}
