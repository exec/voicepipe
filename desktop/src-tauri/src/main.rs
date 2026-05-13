// voicepipe desktop shell.
//
// On startup: pick a free loopback port, spawn `voicepipe serve --port <p> --no-browser --no-auth`
// as a sidecar (the Python engine), wait until /v1/health answers, then create the window
// pointing at it. On exit (window closed, Cmd-Q, or even a force-kill of this process — the engine
// watches its parent pid): the sidecar is killed. The native app is trusted — no auth.
//
// FINDING THE ENGINE: a Finder-launched .app does NOT inherit the shell PATH, so we can't just
// rely on `voicepipe` being on PATH. Order: $VOICEPIPE_BIN ; `voicepipe` on PATH ; a list of
// common dev/install locations (incl. this repo's .venv) ; finally `python3 -m pipeline`. A
// shipped build should instead bundle a standalone Python + voicepipe[gui] under Resources/ and
// point here at that — see ../README.md "Packaging". (The "bind a Unix socket, no TCP port at
// all" refinement is also TODO there.)

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
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

fn wait_for_health(base: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let url = format!("{base}/v1/health");
    while Instant::now() < deadline {
        if Command::new("curl")
            .args(["-sf", "-o", "/dev/null", &url])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
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
    // 4. common dev/install locations for the console script
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
    // 5. last resort: a python that can `-m pipeline` (the repo must be importable)
    for venv_py in [
        format!("{home}/Developer/dec-bot/.venv/bin/python"),
        format!("{home}/dec-bot/.venv/bin/python"),
    ] {
        if PathBuf::from(&venv_py).exists() {
            return Some((PathBuf::from(venv_py), vec!["-m".into(), "pipeline".into()]));
        }
    }
    None
}

fn spawn_engine(app: &tauri::AppHandle, port: u16) -> Result<Child, String> {
    let (program, mut args) = locate_engine(app).ok_or_else(|| {
        "Could not find the voicepipe engine. (In a packaged build it ships inside the app; for a \
         dev checkout, `pip install -e \".[gui]\"` so `voicepipe` is on PATH, or set $VOICEPIPE_BIN.)".to_string()
    })?;
    // --no-auth: the native app is trusted; never prompt for a token regardless of ambient env.
    args.extend(["serve".into(), "--host".into(), "127.0.0.1".into(),
                 "--port".into(), port.to_string(), "--no-browser".into(), "--no-auth".into()]);
    Command::new(&program)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env_remove("VOICEPIPE_AUTH_TOKEN")                                  // belt + suspenders with --no-auth
        .env("VOICEPIPE_PARENT_PID", std::process::id().to_string())         // so the engine exits if we get killed
        .spawn()
        .map_err(|e| format!("Failed to launch the engine ({}): {e}", program.display()))
}

fn main() {
    let port = free_loopback_port();
    let base = format!("http://127.0.0.1:{port}");

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup({
            let base = base.clone();
            move |app| {
                match spawn_engine(app.handle(), port) {
                    Ok(child) => {
                        app.manage(Sidecar(Mutex::new(Some(child))));
                        let _ = wait_for_health(&base, Duration::from_secs(30));
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
