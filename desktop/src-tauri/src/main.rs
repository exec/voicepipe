// voicepipe desktop shell.
//
// On startup: pick a free loopback port, spawn `voicepipe serve --port <p> --no-browser --no-auth`
// as a sidecar (the Python engine), wait until /v1/health answers, then create the window
// pointing at it. On exit (window closed, Cmd-Q, or even a force-kill of this process — the engine
// watches its parent pid): the sidecar is killed. The native app is trusted — no auth.
//
// FINDING THE ENGINE: a Finder-launched .app does NOT inherit the shell PATH, so we can't just
// rely on `voicepipe` being on PATH. Release-build order: bundled Resources/voicepipe-serve ;
// $VOICEPIPE_BIN ; `voicepipe` on PATH. Debug builds additionally probe a list of common dev
// venv locations and `python3 -m pipeline` (gated on cfg!(debug_assertions) — those paths must
// NOT ship in release). A shipped build bundles a standalone Python + voicepipe[gui] under
// Resources/ — see ../README.md "Packaging". (The "bind a Unix socket, no TCP port at all"
// refinement is also TODO there.)

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

struct Sidecar(Mutex<Option<Child>>);

impl Sidecar {
    fn kill(&self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
            }
        }
    }
}

fn free_loopback_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8765)
}

/// Minimal HTTP/1.1 GET over a TcpStream — avoids shelling out to `curl`, which isn't
/// guaranteed on minimal Windows / Linux container hosts. Returns true on a 2xx status.
fn http_get_ok(host: &str, port: u16, path: &str) -> bool {
    let addr = format!("{host}:{port}");
    let mut stream = match TcpStream::connect_timeout(
        &match addr.parse() {
            Ok(a) => a,
            Err(_) => return false,
        },
        Duration::from_millis(500),
    ) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(1500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: */*\r\nUser-Agent: voicepipe-desktop/0.1\r\n\r\n"
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    // Read just enough to parse the status line.
    let mut buf = [0u8; 64];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return false,
    };
    // Expected: "HTTP/1.1 2xx ..."
    let head = &buf[..n];
    if head.len() < 12 || !head.starts_with(b"HTTP/") {
        return false;
    }
    matches!(head.get(9), Some(b'2'))
}

fn wait_for_health(base_host: &str, base_port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if http_get_ok(base_host, base_port, "/v1/health") {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

/// (program, leading_args) for launching the engine. Prefers the engine bundled inside the .app
/// (Resources/voicepipe-serve/voicepipe-serve); falls back to a dev install for `cargo tauri dev`.
fn locate_engine(app: &tauri::AppHandle) -> Option<(PathBuf, Vec<String>)> {
    // 1. the bundled, self-contained engine (PyInstaller --onedir under Resources/)
    if let Ok(res) = app.path().resource_dir() {
        let exe = if cfg!(windows) { "voicepipe-serve.exe" } else { "voicepipe-serve" };
        let bundled = res.join("voicepipe-serve").join(exe);
        if bundled.is_file() {
            return Some((bundled, vec![]));
        }
    }
    // 2. an explicit override
    if let Ok(p) = std::env::var("VOICEPIPE_BIN") {
        if !p.trim().is_empty() && PathBuf::from(&p).exists() {
            return Some((PathBuf::from(p), vec![]));
        }
    }
    // 3. `voicepipe` on PATH
    if Command::new("voicepipe").arg("--help").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok() {
        return Some((PathBuf::from("voicepipe"), vec![]));
    }
    // 4 + 5: dev-only fallbacks (a developer's local venv layout). A release build must rely
    // on the bundled Resources/ engine (#1), $VOICEPIPE_BIN (#2), or a PATH `voicepipe` (#3).
    if cfg!(debug_assertions) {
        let home = std::env::var("HOME").unwrap_or_default();
        for c in [
            format!("{home}/Developer/dec-bot/.venv/bin/voicepipe"),
            format!("{home}/dec-bot/.venv/bin/voicepipe"),
            format!("{home}/voicepipe/.venv/bin/voicepipe"),
            format!("{home}/.local/bin/voicepipe"),
            "/opt/homebrew/bin/voicepipe".to_string(),
            "/usr/local/bin/voicepipe".to_string(),
        ] {
            if PathBuf::from(&c).exists() {
                return Some((PathBuf::from(c), vec![]));
            }
        }
        for venv_py in [
            format!("{home}/Developer/dec-bot/.venv/bin/python"),
            format!("{home}/dec-bot/.venv/bin/python"),
        ] {
            if PathBuf::from(&venv_py).exists() {
                return Some((PathBuf::from(venv_py), vec!["-m".into(), "pipeline".into()]));
            }
        }
    }
    None
}

/// A small rolling buffer of the sidecar's stderr — only used to surface a diagnostic in the
/// "health never came up" dialog. Bounded so a chatty engine doesn't accumulate forever.
const STDERR_TAIL_BYTES: usize = 8192;

#[derive(Clone, Default)]
struct StderrTail(Arc<Mutex<String>>);

impl StderrTail {
    fn snapshot(&self) -> String {
        self.0.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

fn spawn_engine(app: &tauri::AppHandle, port: u16) -> Result<(Child, StderrTail), String> {
    let (program, mut args) = locate_engine(app).ok_or_else(|| {
        "Could not find the voicepipe engine. (In a packaged build it ships inside the app; for a \
         dev checkout, `pip install -e \".[gui]\"` so `voicepipe` is on PATH, or set $VOICEPIPE_BIN.)".to_string()
    })?;
    // --no-auth: the native app is trusted; never prompt for a token regardless of ambient env.
    args.extend(["serve".into(), "--host".into(), "127.0.0.1".into(),
                 "--port".into(), port.to_string(), "--no-browser".into(), "--no-auth".into()]);
    let mut child = Command::new(&program)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .env_remove("VOICEPIPE_AUTH_TOKEN")                                  // belt + suspenders with --no-auth
        .env("VOICEPIPE_PARENT_PID", std::process::id().to_string())         // so the engine exits if we get killed
        .spawn()
        .map_err(|e| format!("Failed to launch the engine ({}): {e}", program.display()))?;

    let tail = StderrTail::default();
    if let Some(stderr) = child.stderr.take() {
        let buf = tail.0.clone();
        std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                // Pass it through so devs running `cargo tauri dev` still see engine logs,
                // and keep a bounded copy for the health-failure dialog.
                eprintln!("{line}");
                if let Ok(mut g) = buf.lock() {
                    g.push_str(&line);
                    g.push('\n');
                    if g.len() > STDERR_TAIL_BYTES {
                        let drop_to = g.len() - STDERR_TAIL_BYTES;
                        // Drop a prefix; align to a char boundary so we don't panic on multibyte.
                        let mut idx = drop_to;
                        while idx < g.len() && !g.is_char_boundary(idx) {
                            idx += 1;
                        }
                        g.replace_range(..idx, "");
                    }
                }
            }
        });
    }
    Ok((child, tail))
}

fn main() {
    let port = free_loopback_port();
    let base = format!("http://127.0.0.1:{port}");

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup({
            let base = base.clone();
            move |app| {
                match spawn_engine(app.handle(), port) {
                    Ok((child, stderr_tail)) => {
                        app.manage(Sidecar(Mutex::new(Some(child))));
                        if !wait_for_health("127.0.0.1", port, Duration::from_secs(30)) {
                            // Sidecar spawned but /v1/health never came up — surface the captured
                            // stderr instead of letting the webview open against a dead port.
                            let tail = stderr_tail.snapshot();
                            let mut msg = format!(
                                "The voicepipe engine launched but never reported healthy on {base}/v1/health.\n\n"
                            );
                            if tail.trim().is_empty() {
                                msg.push_str("(No stderr output was captured from the engine.)");
                            } else {
                                msg.push_str("Last engine stderr:\n\n");
                                msg.push_str(tail.trim_end());
                            }
                            eprintln!("voicepipe: engine health check failed.\n{msg}");
                            app.dialog()
                                .message(&msg)
                                .title("voicepipe — engine failed to start")
                                .kind(MessageDialogKind::Error)
                                .blocking_show();
                            if let Some(s) = app.handle().try_state::<Sidecar>() {
                                s.kill();
                            }
                            app.handle().exit(1);
                            return Ok(());
                        }
                        #[allow(unused_mut)]
                        let mut win = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(base.parse().expect("url")))
                            .title("voicepipe")
                            .inner_size(1200.0, 800.0)
                            .min_inner_size(880.0, 560.0)
                            .resizable(true);
                        // macOS: hide the title bar, keep the traffic lights overlaying the content.
                        #[cfg(target_os = "macos")]
                        {
                            win = win.title_bar_style(tauri::TitleBarStyle::Overlay).hidden_title(true);
                        }
                        // Windows/Linux: drop the OS frame entirely; the webui draws its own
                        // min/max/close controls in the topbar (see app.js, gated on window.__TAURI__).
                        #[cfg(not(target_os = "macos"))]
                        {
                            win = win.decorations(false);
                        }
                        win.build()?;
                    }
                    Err(msg) => {
                        app.dialog()
                            .message(msg)
                            .title("voicepipe — engine not found")
                            .kind(MessageDialogKind::Error)
                            .blocking_show();
                        app.handle().exit(1);
                    }
                }
                Ok(())
            }
        })
        .on_window_event(|window, event| {
            if let WindowEvent::Destroyed = event {
                if let Some(s) = window.app_handle().try_state::<Sidecar>() {
                    s.kill();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error building voicepipe desktop");

    app.run(|handle, event| match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => {
            if let Some(s) = handle.try_state::<Sidecar>() {
                s.kill();
            }
        }
        _ => {}
    });
}
