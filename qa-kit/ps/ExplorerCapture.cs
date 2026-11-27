using System;
using System.Drawing;
using System.Runtime.InteropServices;

// Pixels come from ONE addressed window (PrintWindow), never from a rectangle of the screen: whatever lies over that
// window is not in them. C# 5 syntax: compiled by Windows PowerShell 5.1's Add-Type.
namespace ExplorerQa {
    public static class Native {
        [StructLayout(LayoutKind.Sequential)]
        private struct RECT { public int L, T, R, B; }

        [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
        [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out int pid);
        [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hwnd, out RECT r);
        [DllImport("user32.dll")] private static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
        [DllImport("dwmapi.dll")] private static extern int DwmGetWindowAttribute(IntPtr hwnd, int attribute, out RECT value, int size);

        public static int WindowProcess(IntPtr hwnd) {
            int pid;
            if (GetWindowThreadProcessId(hwnd, out pid) == 0) throw new InvalidOperationException("Explorer window disappeared.");
            return pid;
        }

        public static Rectangle WindowBounds(IntPtr hwnd) {
            RECT r;
            if (DwmGetWindowAttribute(hwnd, 9, out r, Marshal.SizeOf(typeof(RECT))) != 0)
                throw new InvalidOperationException("DWM did not return the window frame bounds.");
            return Rectangle.FromLTRB(r.L, r.T, r.R, r.B);
        }

        // One part of the window's own drawing, given in screen coordinates (its frame, its title bar, one row).
        public static Bitmap CaptureWindowPart(IntPtr hwnd, Rectangle part) {
            RECT r;
            if (!GetWindowRect(hwnd, out r)) throw new InvalidOperationException("Window disappeared.");
            int w = r.R - r.L, h = r.B - r.T;
            if (w <= 0 || h <= 0 || w > 8192 || h > 8192) throw new InvalidOperationException("Window size out of bounds.");
            Rectangle local = new Rectangle(part.X - r.L, part.Y - r.T, part.Width, part.Height);
            if (local.Width <= 0 || local.Height <= 0 || !new Rectangle(0, 0, w, h).Contains(local))
                throw new InvalidOperationException("The part asked for is not inside the window.");
            using (Bitmap whole = new Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format24bppRgb)) {
                bool drawn;
                using (Graphics g = Graphics.FromImage(whole)) {
                    IntPtr hdc = g.GetHdc();
                    try { drawn = PrintWindow(hwnd, hdc, 2); }  // PW_RENDERFULLCONTENT: composed content included
                    finally { g.ReleaseHdc(hdc); }
                }
                if (!drawn) throw new InvalidOperationException("The window did not draw itself.");
                return whole.Clone(local, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            }
        }

        // The target row only. frame: the window's bounds as read with the row; row: relative to them.
        public static Bitmap Capture(IntPtr hwnd, Rectangle frame, Rectangle row) {
            if (WindowBounds(hwnd) != frame) throw new InvalidOperationException("Window moved before capture.");
            Bitmap bitmap = CaptureWindowPart(hwnd, new Rectangle(frame.X + row.X, frame.Y + row.Y, row.Width, row.Height));
            try {
                if (WindowBounds(hwnd) != frame) throw new InvalidOperationException("Window moved during capture.");
                return bitmap;
            } catch { bitmap.Dispose(); throw; }
        }

        // Negative control only: a uniform row cannot show an icon and label.
        public static bool HasRowInk(Bitmap bitmap, Rectangle row) {
            if (row.Width <= 0 || row.Height <= 0 || !new Rectangle(Point.Empty, bitmap.Size).Contains(row)) return false;
            int first = bitmap.GetPixel(row.Left, row.Top).ToArgb();
            for (int y = row.Top; y < row.Bottom; y += 2)
                for (int x = row.Left; x < row.Right; x += 2)
                    if (bitmap.GetPixel(x, y).ToArgb() != first) return true;
            return false;
        }
    }
}
