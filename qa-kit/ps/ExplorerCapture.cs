using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

// Adapted from explorer-qa/ExplorerCapture.cs. Pixels come from ONE addressed window (PrintWindow), never from a screen DC.
// C# 5 syntax: compiled by Windows PowerShell 5.1's Add-Type.
namespace ExplorerQa {
    public static class Native {
        [StructLayout(LayoutKind.Sequential)]
        private struct RECT { public int L, T, R, B; }
        [StructLayout(LayoutKind.Sequential)]
        private struct POINT { public int X, Y; }

        [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
        [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out int pid);
        [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hwnd, out RECT r);
        [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hwnd);
        [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hwnd);
        [DllImport("user32.dll")] private static extern bool GetClientRect(IntPtr hwnd, out RECT r);
        [DllImport("user32.dll")] private static extern bool ClientToScreen(IntPtr hwnd, ref POINT point);
        [DllImport("user32.dll")] private static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
        [DllImport("dwmapi.dll")] private static extern int DwmGetWindowAttribute(IntPtr hwnd, int attribute, out RECT value, int size);

        public static int WindowProcess(IntPtr hwnd) {
            int pid;
            if (GetWindowThreadProcessId(hwnd, out pid) == 0) throw new InvalidOperationException("capture_window_gone");
            return pid;
        }

        public static Rectangle WindowBounds(IntPtr hwnd) {
            RECT r;
            if (DwmGetWindowAttribute(hwnd, 9, out r, Marshal.SizeOf(typeof(RECT))) != 0)
                throw new InvalidOperationException("capture_dwm_bounds_unavailable");
            return Rectangle.FromLTRB(r.L, r.T, r.R, r.B);
        }

        private static Rectangle BindWindow(IntPtr hwnd, int expectedPid, Rectangle frame) {
            if (WindowProcess(hwnd) != expectedPid || !IsWindowVisible(hwnd) || IsIconic(hwnd))
                throw new InvalidOperationException("capture_window_identity_or_visibility_changed");
            if (WindowBounds(hwnd) != frame) throw new InvalidOperationException("capture_frame_moved");
            RECT r;
            if (!GetWindowRect(hwnd, out r)) throw new InvalidOperationException("capture_win32_bounds_unavailable");
            return Rectangle.FromLTRB(r.L, r.T, r.R, r.B);
        }

        private static Rectangle ClientInScreen(IntPtr hwnd) {
            RECT r;
            POINT origin = new POINT();
            if (!GetClientRect(hwnd, out r) || !ClientToScreen(hwnd, ref origin))
                throw new InvalidOperationException("capture_client_bounds_unavailable");
            return new Rectangle(origin.X, origin.Y, r.R - r.L, r.B - r.T);
        }

        // Same verified HWND for both modes. The composed mode is the one already used for Explorer; ordinary PrintWindow
        // (flags = 0, whole-window drawing) is a fallback when the target cannot render usable pixels with that mode.
        public static Bitmap Capture(IntPtr hwnd, int expectedPid, Rectangle frame, Rectangle part, bool needClientInk, out uint usedFlags) {
            usedFlags = 0;
            Rectangle raw = BindWindow(hwnd, expectedPid, frame);
            if (raw.Width <= 0 || raw.Height <= 0 || raw.Width > 8192 || raw.Height > 8192)
                throw new InvalidOperationException("capture_unusable_window_size");
            if (part.Width <= 0 || part.Height <= 0 || !frame.Contains(part) || !raw.Contains(part))
                throw new InvalidOperationException("capture_crop_out_of_bounds");
            Rectangle local = new Rectangle(part.X - raw.X, part.Y - raw.Y, part.Width, part.Height);
            Rectangle client = needClientInk ? Rectangle.Intersect(part, ClientInScreen(hwnd)) : part;
            client.Offset(-raw.X, -raw.Y);
            string failure = "capture_printwindow";
            foreach (uint flags in new uint[] { 2, 0 }) { // PW_RENDERFULLCONTENT, then default
                if (BindWindow(hwnd, expectedPid, frame) != raw) throw new InvalidOperationException("capture_win32_bounds_changed");
                using (Bitmap whole = new Bitmap(raw.Width, raw.Height, PixelFormat.Format24bppRgb)) {
                    bool drawn;
                    using (Graphics g = Graphics.FromImage(whole)) {
                        IntPtr hdc = g.GetHdc();
                        try { drawn = PrintWindow(hwnd, hdc, flags); }
                        finally { g.ReleaseHdc(hdc); }
                    }
                    if (BindWindow(hwnd, expectedPid, frame) != raw) throw new InvalidOperationException("capture_win32_bounds_changed");
                    if (!drawn) { failure += "_" + flags + "_failed"; continue; }
                    if (!HasRowInk(whole, local) || (needClientInk && !HasRowInk(whole, client))) {
                        failure += "_" + flags + "_blank";
                        continue;
                    }
                    usedFlags = flags;
                    return whole.Clone(local, PixelFormat.Format24bppRgb);
                }
            }
            throw new InvalidOperationException(failure);
        }

        // Reopen the exact saved path; validate/decode its bytes and hash that same stream. A receipt is not a filename guess.
        public static string SaveCheckedPng(Bitmap bitmap, string path) {
            using (FileStream output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None)) {
                bitmap.Save(output, ImageFormat.Png);
            }
            using (FileStream input = File.OpenRead(path)) {
                using (Image image = Image.FromStream(input, false, true)) {
                    if (image.RawFormat.Guid != ImageFormat.Png.Guid || image.Width != bitmap.Width || image.Height != bitmap.Height)
                        throw new InvalidOperationException("capture_png_readback_mismatch");
                    using (Bitmap decoded = new Bitmap(image)) {
                        if (!HasRowInk(decoded, new Rectangle(Point.Empty, decoded.Size)))
                            throw new InvalidOperationException("capture_png_readback_blank");
                    }
                }
                input.Position = 0;
                using (SHA256 sha = SHA256.Create()) {
                    return BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
                }
            }
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
