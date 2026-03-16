use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";

    // Find CUDA
    let cuda_dir = if is_windows {
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
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

    // Windows: ensure MSVC on PATH
    if is_windows {
        let msvc_bin = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64";
        let path = env::var("PATH").unwrap_or_default();
        if !path.contains("MSVC") {
            unsafe { env::set_var("PATH", format!("{msvc_bin};{path}")) };
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
    let lib_name = "kraken_cuda";
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

    println!("cargo:rerun-if-changed=cuda/");
}
