open Configurator.V1

module Process = Configurator.V1.Process

let concat_map f xs = List.concat (List.map f xs)

let default_candidates =
  [
    "/opt/homebrew/opt/llvm/lib";
    "/usr/local/opt/llvm/lib";
    "/usr/lib/llvm-19/lib";
    "/usr/lib/llvm-18/lib";
    "/usr/lib/llvm-17/lib";
    "/usr/lib/llvm/lib";
  ]

let capi_archives =
  [
    "libMLIRCAPIIR.a";
    "libMLIRCAPIFunc.a";
    "libMLIRCAPIArith.a";
    "libMLIRCAPIControlFlow.a";
    "libMLIRCAPIRegisterEverything.a";
  ]

let dir_exists path =
  try
    let stats = Unix.stat path in
    stats.Unix.st_kind = Unix.S_DIR
  with
  | Unix.Unix_error _ -> false

let detect_libdir cfg =
  match Sys.getenv_opt "LLVM_LIBDIR" with
  | Some path when dir_exists path -> path
  | Some path ->
      die "%s is set but %s does not exist" "LLVM_LIBDIR" path
  | None -> (
      let from_llvm_config =
        match Process.run cfg "llvm-config" ["--libdir"] with
        | { Process.exit_code = 0; stdout; _ } ->
            let candidate = String.trim stdout in
            if dir_exists candidate then Some candidate else None
        | _ -> None
      in
      match from_llvm_config with
      | Some path -> path
      | None -> (
          match List.find_opt dir_exists default_candidates with
          | Some path -> path
          | None ->
              die
                "Unable to locate LLVM's lib directory. Set LLVM_LIBDIR or \
                 ensure llvm-config is on PATH." ))

type host =
  | Mac
  | Linux
  | Unknown of string

let detect_host cfg =
  match ocaml_config_var cfg "system" with
  | Some ("macosx" | "darwin") -> Mac
  | Some "linux" -> Linux
  | Some other -> Unknown other
  | None -> Unknown "unknown"

let mac_force_loads libdir =
  concat_map
    (fun archive ->
      let full = Filename.concat libdir archive in
      ["-cclib"; Printf.sprintf "-Wl,-force_load,%s" full])
    capi_archives

let linux_whole_archive libdir =
  ["-cclib"; "-Wl,--whole-archive"]
  @ concat_map
      (fun archive ->
        let full = Filename.concat libdir archive in
        ["-cclib"; full])
      capi_archives
  @ ["-cclib"; "-Wl,--no-whole-archive"]

let mac_runtime_libs libdir =
  concat_map
    (fun name ->
      let full = Filename.concat libdir name in
      ["-cclib"; full])
    ["libMLIR.dylib"; "libLLVM-C.dylib"; "libLLVM.dylib"]

let linux_runtime_libs =
  ["-cclib"; "-lMLIR"; "-cclib"; "-lLLVM-C"; "-cclib"; "-lLLVM"]

let mac_cxx_runtime = ["-cclib"; "-lc++"; "-cclib"; "-lc++abi"]
let linux_cxx_runtime = ["-cclib"; "-lstdc++"; "-cclib"; "-lm"]

let emit_flags host libdir =
  let common =
    [
      "-cclib";
      Printf.sprintf "-L%s" libdir;
      "-cclib";
      Printf.sprintf "-Wl,-rpath,%s" libdir;
      "-cclib";
      "-Wl,-export_dynamic";
    ]
  in
  match host with
  | Mac ->
      common @ mac_force_loads libdir @ mac_cxx_runtime @ mac_runtime_libs libdir
  | Linux ->
      common @ linux_whole_archive libdir @ linux_cxx_runtime
      @ linux_runtime_libs
  | Unknown system ->
      die "Unsupported host platform: %s. Set LLVM_LIBDIR and edit detect_llvm."
        system

let output_file = ref None

let () =
  let set_output path = output_file := Some path in
  main
    ~name:"detect_llvm"
    ~args:
      [
        ( "-o"
        , Arg.String set_output
        , "Path to the generated mlir_link_flags.sexp file" );
      ]
    (fun cfg ->
      let target =
        match !output_file with
        | Some path -> path
        | None -> die "detect_llvm requires -o <output>"
      in
      let libdir = detect_libdir cfg in
      let host = detect_host cfg in
      let flags = emit_flags host libdir in
      Flags.write_lines target flags)
