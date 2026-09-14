package ai.ouroboros.android;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.service.wallpaper.WallpaperService;
import android.view.SurfaceHolder;

/** Minimal live wallpaper primitive; policy/content remains owned by the core. */
public final class OuroborosWallpaperService extends WallpaperService {
    @Override public Engine onCreateEngine() { return new Engine() {
        @Override public void onVisibilityChanged(boolean visible) { if (visible) draw(); }
        @Override public void onSurfaceChanged(SurfaceHolder holder, int format, int width, int height) { draw(); }
        private void draw() {
            Canvas canvas = null;
            try {
                canvas = getSurfaceHolder().lockCanvas();
                if (canvas == null) return;
                canvas.drawColor(Color.rgb(11, 18, 16));
                Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
                paint.setColor(Color.rgb(80, 220, 150)); paint.setTextSize(36);
                canvas.drawText("Ouroboros", 48, 96, paint);
            } finally { if (canvas != null) getSurfaceHolder().unlockCanvasAndPost(canvas); }
        }
    }; }
}
