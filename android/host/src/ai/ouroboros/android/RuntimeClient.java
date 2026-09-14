package ai.ouroboros.android;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import java.util.List;
import java.util.LinkedHashSet;
import org.json.JSONObject;

/** Native host operations. The shared launcher owns every core generation. */
final class RuntimeClient {
    static final String CONTROL = "/data/local/ouroboros-phone/bin/core-control";
    static volatile int port = 8765;

    static String baseUrl() { return "http://localhost:" + port; }

    static JSONObject request(String path, String method) throws Exception {
        return request(path, method, new JSONObject());
    }

    static JSONObject request(String path, String method, JSONObject payload) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(baseUrl() + path).openConnection();
        connection.setConnectTimeout(3000);
        connection.setReadTimeout(5000);
        connection.setRequestMethod(method);
        try {
            if (!method.equals("GET")) {
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json");
                try (java.io.OutputStream stream = connection.getOutputStream()) {
                    stream.write(payload.toString().getBytes(StandardCharsets.UTF_8));
                }
            }
            int status = connection.getResponseCode();
            String body = read(connection.getErrorStream() != null
                    ? connection.getErrorStream() : connection.getInputStream());
            if (status < 200 || status >= 300) throw new java.io.IOException("HTTP " + status + ": " + body);
            return new JSONObject(body);
        } finally { connection.disconnect(); }
    }

    static String control(String action, String intent) throws Exception {
        // Arguments come only from native Start/boot/status actions, never WebView JS.
        String output = rootCommand(CONTROL + " " + action + " " + intent, null);
        if (action.equals("port")) {
            int discovered = Integer.parseInt(output);
            if (discovered < 1 || discovered > 65535) throw new java.io.IOException("Invalid runtime port");
            port = discovered;
        }
        return output;
    }

    static String dnsConfiguration(List<InetAddress> servers) {
        LinkedHashSet<String> addresses = new LinkedHashSet<>();
        for (InetAddress server : servers) addresses.add(server.getHostAddress());
        StringBuilder text = new StringBuilder();
        for (String address : addresses) text.append("nameserver ").append(address).append('\n');
        return text.toString();
    }

    static String dnsWriteCommand(String path) {
        // path is a native constant; DNS bytes enter through stdin, never shell syntax.
        String target = "'" + path.replace("'", "'\\''") + "'";
        return "set -e; target=" + target + "; temporary=$(mktemp \"${target}.ouroboros.XXXXXX\"); "
                + "trap 'rm -f \"$temporary\"' EXIT; cat > \"$temporary\"; "
                + "test -s \"$temporary\" || exit 0; "
                + "if ! cmp -s \"$temporary\" \"$target\"; then chmod 644 \"$temporary\"; "
                + "mv \"$temporary\" \"$target\"; fi";
    }

    static void updateDns(String configuration) throws Exception {
        if (configuration.isEmpty()) return;
        rootCommand(dnsWriteCommand("/data/local/ouroboros-phone/rootfs/etc/resolv.conf"), configuration);
    }

    private static String rootCommand(String command, String input) throws Exception {
        Process process = new ProcessBuilder("su", "-c", command)
                .redirectErrorStream(true).start();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Thread reader = new Thread(() -> {
            try (InputStream stream = process.getInputStream()) {
                byte[] buffer = new byte[4096]; int count;
                while ((count = stream.read(buffer)) >= 0) bytes.write(buffer, 0, count);
            } catch (Exception ignored) { /* Process exit and status remain authoritative. */ }
        }, "ouroboros-control-output");
        reader.start();
        try {
            try (java.io.OutputStream stream = process.getOutputStream()) {
                if (input != null) stream.write(input.getBytes(StandardCharsets.UTF_8));
            }
            if (!process.waitFor(30, TimeUnit.SECONDS))
                throw new java.io.IOException("Действие ещё не подтверждено. Проверьте статус перед повтором.");
            reader.join(1000);
            String output = bytes.toString("UTF-8").trim();
            if (process.exitValue() != 0) throw new java.io.IOException(output);
            return output;
        } finally {
            if (process.isAlive()) process.destroy();
        }
    }

    static String read(InputStream stream) throws Exception {
        try (InputStream input = stream; ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192]; int count;
            while ((count = input.read(buffer)) >= 0) bytes.write(buffer, 0, count);
            return bytes.toString("UTF-8");
        }
    }
}
