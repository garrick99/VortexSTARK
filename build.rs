use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";

    // Find CUDA
    let cuda_dir = if is_windows {
        // Try v13.x versions in preference order, fall back to v13.0
        let cuda_base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        let candidate = std::fs::read_dir(&cuda_base)
            .ok()
            .and_then(|entries| {
                let mut versions: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .filter_map(|e| {
                        let name = e.file_name().into_string().ok()?;
                        if name.starts_with("v13.") { Some((name, e.path())) } else { None }
                    })
                    .collect();
                versions.sort_by(|a, b| a.0.cmp(&b.0)); // sort ascending, take highest
                versions.into_iter().last().map(|(_, p)| p)
            });
        candidate.unwrap_or_else(|| cuda_base.join("v13.0"))
    } else {
        // Linux: check CUDA_PATH, then common locations
        env::var("CUDA_PATH").map(PathBuf::from).unwrap_or_else(|_| {
            for path in &["/usr/local/cuda", "/usr/local/cuda-13.2", "/usr/local/cuda-12.8"] {
                if std::fs::metadata(path).is_ok() {
                    return PathBuf::from(path);
                }
            }
            PathBuf::from("/usr/local/cuda")
        })
    };

    let nvcc = if is_windows {
        cuda_dir.join("bin").join("nvcc.exe")
    } else {
        cuda_dir.join("bin").join("nvcc")
    };

    // Windows: ensure MSVC lib.exe is on PATH
    if is_windows {
        let path = env::var("PATH").unwrap_or_default();
        if !path.contains("MSVC") {
            // Use vswhere to find the MSVC toolchain dynamically
            let vswhere_paths = [
                r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
                r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
            ];
            let msvc_bin = vswhere_paths.iter()
                .find(|p| std::fs::metadata(p).is_ok())
                .and_then(|vswhere| {
                    Command::new(vswhere)
                        .args(["-latest", "-products", "*",
                               "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                               "-find", r"VC\Tools\MSVC\*\bin\Hostx64\x64"])
                        .output().ok()
                })
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.lines().last().unwrap_or("").trim().to_string())
                .filter(|s| !s.is_empty());
            if let Some(bin) = msvc_bin {
                unsafe { env::set_var("PATH", format!("{bin};{path}")) };
            }
        }
    }

    // Collect all .cu files
    let cuda_sources: Vec<PathBuf> = std::fs::read_dir("cuda")
        .expect("cuda/ directory not found")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
        .collect();

    let obj_ext = if is_windows { "obj" } else { "o" };
    let mut objects = Vec::new();

    for src in &cuda_sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.{obj_ext}"));

        let mut cmd = Command::new(&nvcc);
        cmd.args(["-c", "-O3"]);

        // On Windows, nvcc may reject newer MSVC/SDK versions as an unsupported host compiler.
        // Use an older MSVC (14.44 / VS 2022) and an older Windows SDK (22621) for CUDA
        // host compilation, which CUDA 13.x is known to support.
        if is_windows {
            // Find the MSVC toolset: scan VS install dirs for any BuildTools\VC\Tools\MSVC\<ver>
            // and pick the first (lowest = most CUDA-compatible).
            let msvc_bin = (|| {
                for vs_ver in &["2022", "2019", "18"] {
                    for edition in &["BuildTools", "Community", "Professional", "Enterprise"] {
                        let msvc_root = PathBuf::from(format!(
                            r"C:\Program Files (x86)\Microsoft Visual Studio\{}\{}\VC\Tools\MSVC",
                            vs_ver, edition
                        ));
                        if let Ok(entries) = std::fs::read_dir(&msvc_root) {
                            let mut versions: Vec<PathBuf> = entries
                                .filter_map(|e| e.ok().map(|e| e.path()))
                                .filter(|p| p.join("bin").join("Hostx64").join("x64").join("cl.exe").exists())
                                .collect();
                            versions.sort();
                            if let Some(v) = versions.first() {
                                return Some(v.join("bin").join("Hostx64").join("x64"));
                            }
                        }
                    }
                }
                None
            })();

            if let Some(ccbin) = &msvc_bin {
                cmd.arg("-ccbin").arg(ccbin);

                // Set INCLUDE for CUDA host compilation using the found MSVC
                let sdk_base = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\Include");
                let msvc_include = ccbin.parent().and_then(|p| p.parent())
                    .and_then(|p| p.parent()).and_then(|p| p.parent())
                    .map(|p| p.join("include"));
                for sdk_ver in &["10.0.22621.0", "10.0.22000.0", "10.0.26100.0"] {
                    let sdk_inc = sdk_base.join(sdk_ver);
                    if sdk_inc.exists() {
                        if let Some(msvc_inc) = &msvc_include {
                            let include = format!(
                                "{};{};{};{}",
                                msvc_inc.display(),
                                sdk_inc.join("ucrt").display(),
                                sdk_inc.join("um").display(),
                                sdk_inc.join("shared").display(),
                            );
                            cmd.env("INCLUDE", &include);
                        }
                        break;
                    }
                }
            }
        }

        // GPU architectures
        cmd.args(["-gencode", "arch=compute_89,code=sm_89"]);
        // sm_120 (Blackwell) requires CUDA 13+; detect via nvcc version
        let nvcc_version = Command::new(&nvcc).arg("--version").output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        if nvcc_version.contains("cuda_13") || nvcc_version.contains("cuda_14") {
            cmd.args(["-gencode", "arch=compute_120,code=sm_120"]);
        }
        cmd.args(["-gencode", "arch=compute_89,code=compute_89"]);

        if is_windows {
            cmd.arg("--allow-unsupported-compiler");
        }

        cmd.args(["-Icuda/include", "-o"]);
        cmd.arg(&obj).arg(src);

        let status = cmd.status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc at {}: {e}", nvcc.display()));
        assert!(status.success(), "nvcc failed on {}", src.display());
        objects.push(obj);
    }

    // Link objects into static library
    let lib_name = "vortex_cuda";
    if is_windows {
        let lib_path = out_dir.join(format!("{lib_name}.lib"));
        let mut args = vec![format!("/OUT:{}", lib_path.display())];
        for obj in &objects {
            args.push(obj.to_str().unwrap().to_string());
        }
        let status = Command::new("lib.exe")
            .args(&args)
            .status()
            .expect("Failed to run lib.exe");
        assert!(status.success(), "lib.exe failed");
    } else {
        let lib_path = out_dir.join(format!("lib{lib_name}.a"));
        let mut cmd = Command::new("ar");
        cmd.arg("rcs").arg(&lib_path);
        for obj in &objects {
            cmd.arg(obj);
        }
        let status = cmd.status().expect("Failed to run ar");
        assert!(status.success(), "ar failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");

    // Link CUDA runtime
    if is_windows {
        println!("cargo:rustc-link-search=native={}", cuda_dir.join("lib").join("x64").display());
    } else {
        println!("cargo:rustc-link-search=native={}", cuda_dir.join("lib64").display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Also link stdc++ on Linux (CUDA runtime depends on it)
    if !is_windows {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Emit per-file rerun-if-changed so cargo recompiles CUDA when any .cu file changes.
    println!("cargo:rerun-if-changed=build.rs");
    for src in &cuda_sources {
        println!("cargo:rerun-if-changed={}", src.display());
    }
    // Also track the cuda include directory.
    for entry in std::fs::read_dir("cuda/include").into_iter().flatten().filter_map(|e| e.ok()) {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
}
