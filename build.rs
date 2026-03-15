use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_dir = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0");
    let nvcc = cuda_dir.join("bin").join("nvcc.exe");

    // Ensure MSVC cl.exe and lib.exe are on PATH for nvcc
    let msvc_bin = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64";
    let path = env::var("PATH").unwrap_or_default();
    if !path.contains("MSVC") {
        unsafe { env::set_var("PATH", format!("{msvc_bin};{path}")) };
    }

    // Collect all .cu files from cuda/
    let cuda_sources: Vec<PathBuf> = std::fs::read_dir("cuda")
        .expect("cuda/ directory not found")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
        .collect();

    let mut objects = Vec::new();
    for src in &cuda_sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.obj"));

        let status = Command::new(&nvcc)
            .args([
                "-c",
                "-O3",
                "-gencode", "arch=compute_89,code=sm_89",
                "-gencode", "arch=compute_120,code=sm_120",
                "-gencode", "arch=compute_89,code=compute_89",
                "--allow-unsupported-compiler",
                "-Icuda/include",
                "-o",
            ])
            .arg(&obj)
            .arg(src)
            .status()
            .expect("Failed to run nvcc");

        assert!(status.success(), "nvcc failed on {}", src.display());
        objects.push(obj);
    }

    // Link all .obj into a static lib
    let lib_path = out_dir.join("kraken_cuda.lib");
    let mut args = vec!["/OUT:".to_string() + lib_path.to_str().unwrap()];
    for obj in &objects {
        args.push(obj.to_str().unwrap().to_string());
    }

    let status = Command::new("lib.exe")
        .args(&args)
        .status()
        .expect("Failed to run lib.exe");

    assert!(status.success(), "lib.exe failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kraken_cuda");

    // Link CUDA runtime
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_dir.join("lib").join("x64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Rebuild if CUDA sources change
    println!("cargo:rerun-if-changed=cuda/");
}
