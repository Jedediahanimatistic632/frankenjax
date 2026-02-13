#![forbid(unsafe_code)]

use fj_conformance::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, scrub_sidecar,
};
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(usage());
    }

    let command = args.remove(0);
    match command.as_str() {
        "generate" => cmd_generate(args),
        "scrub" => cmd_scrub(args),
        "proof" => cmd_proof(args),
        "pipeline" => cmd_pipeline(args),
        _ => Err(usage()),
    }
}

fn cmd_generate(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;

    let symbol_size = optional_u16_flag(&args, "--symbol-size")?.unwrap_or(256);
    let max_block_size = optional_usize_flag(&args, "--max-block-size")?.unwrap_or(1024 * 1024);
    let repair_overhead = optional_f64_flag(&args, "--repair-overhead")?.unwrap_or(1.1);

    let manifest = encode_artifact_to_sidecar(
        &artifact,
        &sidecar,
        &SidecarConfig {
            symbol_size,
            max_block_size,
            repair_overhead,
        },
    )
    .map_err(|err| err.to_string())?;

    println!(
        "generated sidecar: {} (total symbols={}, source={}, repair={})",
        sidecar.display(),
        manifest.total_symbols,
        manifest.source_symbols,
        manifest.repair_symbols
    );

    Ok(())
}

fn cmd_scrub(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;
    let report = required_path_flag(&args, "--report")?;

    let scrub = scrub_sidecar(&sidecar, &artifact, &report).map_err(|err| err.to_string())?;
    println!(
        "scrub report: {} (match={})",
        report.display(),
        scrub.decoded_matches_expected
    );
    Ok(())
}

fn cmd_proof(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;
    let proof = required_path_flag(&args, "--proof")?;
    let drop_source_count = optional_usize_flag(&args, "--drop-source")?.unwrap_or(1);

    let result = generate_decode_proof(&sidecar, &artifact, &proof, drop_source_count)
        .map_err(|err| err.to_string())?;
    println!(
        "decode proof: {} (recovered={}, dropped={})",
        proof.display(),
        result.recovered,
        result.dropped_symbols.len()
    );
    Ok(())
}

fn cmd_pipeline(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;
    let report = required_path_flag(&args, "--report")?;
    let proof = required_path_flag(&args, "--proof")?;

    let symbol_size = optional_u16_flag(&args, "--symbol-size")?.unwrap_or(256);
    let max_block_size = optional_usize_flag(&args, "--max-block-size")?.unwrap_or(1024 * 1024);
    let repair_overhead = optional_f64_flag(&args, "--repair-overhead")?.unwrap_or(1.1);
    let drop_source_count = optional_usize_flag(&args, "--drop-source")?.unwrap_or(1);

    encode_artifact_to_sidecar(
        &artifact,
        &sidecar,
        &SidecarConfig {
            symbol_size,
            max_block_size,
            repair_overhead,
        },
    )
    .map_err(|err| err.to_string())?;

    scrub_sidecar(&sidecar, &artifact, &report).map_err(|err| err.to_string())?;
    generate_decode_proof(&sidecar, &artifact, &proof, drop_source_count)
        .map_err(|err| err.to_string())?;

    println!(
        "pipeline complete: sidecar={}, report={}, proof={}",
        sidecar.display(),
        report.display(),
        proof.display()
    );
    Ok(())
}

fn required_path_flag(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let value = required_string_flag(args, flag)?;
    Ok(PathBuf::from(value))
}

fn required_string_flag(args: &[String], flag: &str) -> Result<String, String> {
    for idx in 0..args.len() {
        if args[idx] == flag {
            if let Some(value) = args.get(idx + 1) {
                return Ok(value.clone());
            }
            return Err(format!("missing value for {flag}"));
        }
    }
    Err(format!("missing required flag {flag}"))
}

fn optional_usize_flag(args: &[String], flag: &str) -> Result<Option<usize>, String> {
    optional_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|err| format!("invalid {flag}: {err}"))
        })
        .transpose()
}

fn optional_u16_flag(args: &[String], flag: &str) -> Result<Option<u16>, String> {
    optional_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<u16>()
                .map_err(|err| format!("invalid {flag}: {err}"))
        })
        .transpose()
}

fn optional_f64_flag(args: &[String], flag: &str) -> Result<Option<f64>, String> {
    optional_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("invalid {flag}: {err}"))
        })
        .transpose()
}

fn optional_string_flag(args: &[String], flag: &str) -> Result<Option<String>, String> {
    for idx in 0..args.len() {
        if args[idx] == flag {
            if let Some(value) = args.get(idx + 1) {
                return Ok(Some(value.clone()));
            }
            return Err(format!("missing value for {flag}"));
        }
    }
    Ok(None)
}

fn usage() -> String {
    [
        "usage:",
        "  fj_durability generate --artifact <path> --sidecar <path> [--symbol-size <u16>] [--max-block-size <usize>] [--repair-overhead <f64>]",
        "  fj_durability scrub --artifact <path> --sidecar <path> --report <path>",
        "  fj_durability proof --artifact <path> --sidecar <path> --proof <path> [--drop-source <usize>]",
        "  fj_durability pipeline --artifact <path> --sidecar <path> --report <path> --proof <path> [--symbol-size <u16>] [--max-block-size <usize>] [--repair-overhead <f64>] [--drop-source <usize>]",
    ]
    .join("\n")
}
