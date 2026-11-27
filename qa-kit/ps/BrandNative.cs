using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;

// Documented Win32/Shell calls only (EnumWindows, GetWindowText, WM_GETICON, GetClassLongPtr, the shell property
// stores). No injection, no debugging, no reading of another process's memory. Pixels: ExplorerCapture.cs.
// C# 5 syntax: this is compiled by Windows PowerShell 5.1's Add-Type. x64 only (PROPVARIANT is 24 bytes here).
namespace BrandQa {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
    [StructLayout(LayoutKind.Sequential, Pack = 4)] public struct PROPERTYKEY { public Guid fmtid; public uint pid; }
    [StructLayout(LayoutKind.Explicit, Size = 24)] public struct PROPVARIANT {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr ptr;
        [FieldOffset(8)] public short boolVal;
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore {
        [PreserveSig] int GetCount(out uint cProps);
        [PreserveSig] int GetAt(uint iProp, out PROPERTYKEY pkey);
        [PreserveSig] int GetValue(ref PROPERTYKEY key, out PROPVARIANT pv);
        [PreserveSig] int SetValue(ref PROPERTYKEY key, ref PROPVARIANT pv);
        [PreserveSig] int Commit();
    }

    public sealed class WinInfo {
        public long Hwnd;
        public int Pid;
        public string Title;
        public string ClassName;
    }

    public static class Native {
        private delegate bool EnumProc(IntPtr hwnd, IntPtr lParam);
        private const string PropStoreIid = "886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99";

        [DllImport("user32.dll")] private static extern bool EnumWindows(EnumProc cb, IntPtr lParam);
        [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hwnd);
        [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hwnd);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern int GetWindowTextW(IntPtr hwnd, StringBuilder sb, int max);
        [DllImport("user32.dll")] private static extern int GetWindowTextLengthW(IntPtr hwnd);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern int GetClassNameW(IntPtr hwnd, StringBuilder sb, int max);
        [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out int pid);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern IntPtr FindWindowW(string cls, string title);
        [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hwnd, out RECT r);
        [DllImport("user32.dll", SetLastError = true)] private static extern IntPtr SendMessageTimeoutW(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam, uint flags, uint timeout, out IntPtr result);
        [DllImport("user32.dll", SetLastError = true)] private static extern IntPtr GetClassLongPtrW(IntPtr hwnd, int index);
        [DllImport("kernel32.dll")] private static extern void SetLastError(uint code);
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)] private static extern int SHGetPropertyStoreForWindow(IntPtr hwnd, ref Guid riid, out IPropertyStore ppv);
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)] private static extern int SHGetPropertyStoreFromParsingName(string path, IntPtr pbc, int flags, ref Guid riid, out IPropertyStore ppv);
        [DllImport("ole32.dll")] private static extern int PropVariantClear(ref PROPVARIANT pv);

        // Visible, non-minimised, titled top-level windows of ONE process; no other process's window is read.
        public static List<WinInfo> TopWindows(int pid) {
            List<WinInfo> found = new List<WinInfo>();
            EnumProc cb = delegate (IntPtr hwnd, IntPtr lParam) {
                int owner;
                GetWindowThreadProcessId(hwnd, out owner);
                if (owner != pid || !IsWindowVisible(hwnd) || IsIconic(hwnd)) return true;
                int n = GetWindowTextLengthW(hwnd);
                if (n <= 0) return true;
                StringBuilder title = new StringBuilder(n + 1);
                GetWindowTextW(hwnd, title, n + 1);
                StringBuilder cls = new StringBuilder(256);
                GetClassNameW(hwnd, cls, 256);
                WinInfo w = new WinInfo();
                w.Hwnd = hwnd.ToInt64();
                w.Pid = owner;
                w.Title = title.ToString();
                w.ClassName = cls.ToString();
                found.Add(w);
                return true;
            };
            EnumWindows(cb, IntPtr.Zero);
            GC.KeepAlive(cb);
            return found;
        }

        public static long FindTaskbar() { return FindWindowW("Shell_TrayWnd", null).ToInt64(); }

        public static int[] WindowRectArray(long hwnd) {
            RECT r;
            if (!GetWindowRect(new IntPtr(hwnd), out r)) return new int[] { 0, 0, 0, 0 };
            return new int[] { r.L, r.T, r.R, r.B };
        }

        // Negative control: a uniform frame cannot show an icon or a name.
        public static bool HasInk(Bitmap bmp) {
            int first = bmp.GetPixel(0, 0).ToArgb();
            for (int y = 0; y < bmp.Height; y += 2)
                for (int x = 0; x < bmp.Width; x += 2)
                    if (bmp.GetPixel(x, y).ToArgb() != first) return true;
            return false;
        }

        // WM_GETICON (documented): the icon the window ITSELF supplies. which: 0 small, 1 big, 2 small2.
        // 0 = it answered "none"; long.MinValue = it did not answer, which is not the same as none.
        public static long WindowIcon(long hwnd, int which) {
            IntPtr result;
            IntPtr sent = SendMessageTimeoutW(new IntPtr(hwnd), 0x007F, new IntPtr(which), IntPtr.Zero, 0x0002, 1000, out result);
            return sent == IntPtr.Zero ? long.MinValue : result.ToInt64();
        }

        // The window class's icon (GCLP_HICON -14, GCLP_HICONSM -34): what Windows falls back to. 0 = none. A failed
        // GetClassLongPtrW also returns 0, with a last error: long.MinValue = it was not read, which is not the same as none.
        public static long ClassIcon(long hwnd, bool small) {
            SetLastError(0);
            long value = GetClassLongPtrW(new IntPtr(hwnd), small ? -34 : -14).ToInt64();
            return value == 0 && Marshal.GetLastWin32Error() != 0 ? long.MinValue : value;
        }

        public static Bitmap IconBitmap(long hicon) {
            using (Icon icon = Icon.FromHandle(new IntPtr(hicon))) {
                return icon.ToBitmap();
            }
        }

        private static string ReadValue(IPropertyStore store, string fmtid, uint pid) {
            PROPERTYKEY key = new PROPERTYKEY();
            key.fmtid = new Guid(fmtid);
            key.pid = pid;
            PROPVARIANT pv;
            int hr = store.GetValue(ref key, out pv);
            if (hr != 0) return "<hr:0x" + hr.ToString("X8") + ">";
            try {
                if (pv.vt == 0) return null;                                    // VT_EMPTY: not set
                if (pv.vt == 31) return Marshal.PtrToStringUni(pv.ptr);         // VT_LPWSTR
                if (pv.vt == 11) return pv.boolVal != 0 ? "true" : "false";     // VT_BOOL
                return "<vt:" + pv.vt + ">";
            } finally {
                PropVariantClear(ref pv);
            }
        }

        // A property the window's own property store holds (System.AppUserModel.*). null = not set at window level.
        public static string WindowProp(long hwnd, string fmtid, uint pid) {
            Guid iid = new Guid(PropStoreIid);
            IPropertyStore store;
            int hr = SHGetPropertyStoreForWindow(new IntPtr(hwnd), ref iid, out store);
            if (hr != 0 || store == null) return "<store-hr:0x" + hr.ToString("X8") + ">";
            try { return ReadValue(store, fmtid, pid); } finally { Marshal.ReleaseComObject(store); }
        }

        // A property of a file, read through the shell's own property handler (a .lnk carries System.AppUserModel.ID).
        public static string FileProp(string path, string fmtid, uint pid) {
            Guid iid = new Guid(PropStoreIid);
            IPropertyStore store;
            int hr = SHGetPropertyStoreFromParsingName(path, IntPtr.Zero, 0, ref iid, out store);
            if (hr != 0 || store == null) return "<store-hr:0x" + hr.ToString("X8") + ">";
            try { return ReadValue(store, fmtid, pid); } finally { Marshal.ReleaseComObject(store); }
        }
    }
}
