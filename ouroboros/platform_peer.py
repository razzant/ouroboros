"""Bounded diagnostic peer lookup, accessed through platform_layer only."""

def local_connection_peer(address) -> dict:
    """Bounded, diagnostic-only loopback source-port lookup; never kill authority."""
    import ipaddress
    try:
        host, port = address
        ip = ipaddress.ip_address(str(host).split("%", 1)[0])
        if not (ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)):
            return {}
        if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
            return {}
    except (ValueError, TypeError):
        return {}  # A remote source port cannot identify a local process.
    import json
    import subprocess
    import sys

    if getattr(sys, "frozen", False):
        return {}  # The app executable is not a Python CLI; retain unknown PID.

    # A disposable helper bounds psutil/proc/lsof enumeration, including denied
    # kernel queries. No identity or command match changes stop authority.
    script = r'''
import json, os, sys, subprocess
host, port = json.loads(sys.argv[1]); pid = None
if sys.platform == 'darwin':
    r = subprocess.run(['/usr/sbin/lsof', '-nP', '-iTCP:'+str(port), '-Fpn'], capture_output=True, text=True, timeout=.3)
    current = None
    for line in r.stdout.splitlines():
        if line.startswith('p'): current = int(line[1:])
        if line.startswith('n') and (':'+str(port)+'->') in line: pid = current; break
elif sys.platform.startswith('linux'):
    from pathlib import Path
    inodes = set()
    for table in ('tcp', 'tcp6'):
        for line in Path('/proc/net/'+table).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].rsplit(':',1)[1],16) == port: inodes.add(fields[9])
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try:
            if any(os.readlink(fd) in {'socket:['+i+']' for i in inodes} for fd in (process/'fd').iterdir()):
                pid = int(process.name); break
        except OSError: continue
else:
    import psutil
    for c in psutil.net_connections(kind='tcp'):
        if c.laddr and c.laddr.port == port and c.raddr: pid = c.pid; break
if pid:
    try:
        import psutil
        p = psutil.Process(pid)
        print(json.dumps({'pid':pid,'ppid':p.ppid(),'executable':p.exe(),'command':p.cmdline()}))
    except ImportError:
        r = subprocess.run(['ps','-p',str(pid),'-o','ppid=,comm=,args='],capture_output=True,text=True,timeout=.2)
        parts = r.stdout.strip().split(None,2)
        print(json.dumps({'pid':pid,'ppid':int(parts[0]),'executable':parts[1],'command':parts[-1]}))
'''
    try:
        result = subprocess.run([sys.executable, "-c", script, json.dumps(address)],
                                capture_output=True, text=True, timeout=.7)
        return json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
