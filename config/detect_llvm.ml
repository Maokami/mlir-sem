open Configurator.V1

module Process = Configurator.V1.Process

let concat_map f xs = List.concat (List.map f xs)

let known_versions = ["21"; "20"; "19"; "18"; "17"; "16"; "15"; "14"]

let llvm_config_candidates =
  "llvm-config" :: List.map (fun v -> "llvm-config-" ^ v) known_versions

let default_candidates =
  [
    "/opt/homebrew/opt/llvm/lib";
    "/usr/local/opt/llvm/lib";
    "/usr/lib/llvm/lib";
  ]
  @ List.map (fun v -> Printf.sprintf "/usr/lib/llvm-%s/lib" v) known_versions

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

let rec find_with_llvm_config cfg = function
  | [] -> None
  | prog :: rest -> (
      match Process.run cfg prog ["--libdir"] with
      | { Process.exit_code = 0; stdout; _ } ->
          let candidate = String.trim stdout in
          if dir_exists candidate then Some candidate
          else find_with_llvm_config cfg rest
      | _ -> find_with_llvm_config cfg rest )

let detect_libdir cfg =
  match Sys.getenv_opt "LLVM_LIBDIR" with
  | Some path when dir_exists path -> path
  | Some path ->
      die "%s is set but %s does not exist" "LLVM_LIBDIR" path
  | None -> (
      let from_llvm_config = find_with_llvm_config cfg llvm_config_candidates in
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

let mac_cxx_runtime = ["-cclib"; "-lc++"; "-cclib"; "-lc++abi"]
let linux_cxx_runtime = ["-cclib"; "-lstdc++"; "-cclib"; "-lm"]

let contains_sub str sub =
  let len_s = String.length sub in
  let len = String.length str in
  let rec aux i =
    if i + len_s > len then false
    else if String.sub str i len_s = sub then true
    else aux (i + 1)
  in
  aux 0

let is_library_file name =
  Filename.check_suffix name ".a"
  || Filename.check_suffix name ".so"
  || contains_sub name ".a."
  || contains_sub name ".so."

let starts_with ~prefix s =
  let len_p = String.length prefix in
  String.length s >= len_p && String.sub s 0 len_p = prefix

let candidate_libdirs libdir =
  let dirs =
    [
      libdir;
      Filename.concat libdir "x86_64-unknown-linux-gnu";
      Filename.concat libdir "x86_64-linux-gnu";
      "/usr/lib/x86_64-linux-gnu";
      "/usr/lib";
    ]
  in
  dirs |> List.filter dir_exists |> List.sort_uniq String.compare

let find_library_paths libdir base =
  let is_candidate name =
    is_library_file name
    && starts_with ~prefix:base name
    &&
    (String.length name = String.length base
    ||
    let c = name.[String.length base] in
    Char.equal c '.' || Char.equal c '-')
  in
  let rec search acc = function
    | [] -> List.rev acc
    | dir :: rest -> (
        let entries =
          try Array.to_list (Sys.readdir dir) with Sys_error _ -> []
        in
        let matches =
          List.filter is_candidate entries
          |> List.map (fun name -> Filename.concat dir name)
        in
        search (matches @ acc) rest )
  in
  search [] (candidate_libdirs libdir)

let linux_runtime_libs libdir =
  let resolve base ~required =
    match find_library_paths libdir base with
    | path :: _ -> Some path
    | [] ->
        if required then
          die "Unable to locate %s in %s or common locations" base libdir
        else None
  in
  ["libMLIR", true; "libLLVM-C", false; "libLLVM", true]
  |> concat_map (fun (base, required) ->
         match resolve base ~required with
         | Some path -> ["-cclib"; path]
         | None -> [])

let emit_flags host libdir =
  let export_flag =
    match host with Mac -> "-Wl,-export_dynamic" | _ -> "-Wl,-export-dynamic"
  in
  let common =
    [
      "-cclib";
      Printf.sprintf "-L%s" libdir;
      "-cclib";
      Printf.sprintf "-Wl,-rpath,%s" libdir;
      "-cclib";
      export_flag;
    ]
  in
  match host with
  | Mac ->
      common @ mac_force_loads libdir @ mac_cxx_runtime @ mac_runtime_libs libdir
  | Linux ->
      common @ linux_whole_archive libdir @ linux_cxx_runtime
      @ linux_runtime_libs libdir
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
