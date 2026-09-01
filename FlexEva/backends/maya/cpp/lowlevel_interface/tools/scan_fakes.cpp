#include <clang-c/Index.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

struct Parameter {
  std::string name;
  std::string type;
};

struct FunctionSignature {
  std::string name;
  std::string return_type;
  std::vector<Parameter> parameters;
};

struct FakeMetadata {
  std::string prefix;
  fs::path source_path;
  std::vector<FunctionSignature> functions;
};

std::string normalize_exported_name(std::string prefix, std::string name) {
  if (prefix != "cublas") {
    return name;
  }

  if (name == "cublasCreate" || name == "cublasDestroy" ||
      name == "cublasGetStream" || name == "cublasSetStream" ||
      name == "cublasSgemm" || name == "cublasDgemm") {
    return name + "_v2";
  }
  return name;
}

struct Options {
  fs::path fakes_dir;
  fs::path config_dir;
  fs::path compile_commands;
  bool overwrite{false};
};

struct CompileCommandEntry {
  fs::path file;
  std::vector<std::string> arguments;
};

std::string to_string(CXString str) {
  const char *c = clang_getCString(str);
  std::string result = c != nullptr ? c : "";
  clang_disposeString(str);
  return result;
}

fs::path normalize_path(const fs::path &path) {
  std::error_code ec;
  auto absolute = fs::absolute(path, ec);
  if (ec) {
    return path;
  }
  return absolute.lexically_normal();
}

std::vector<CompileCommandEntry>
parse_compile_commands(const fs::path &compile_commands_path) {
  if (!fs::exists(compile_commands_path)) {
    std::cerr << "warning: compile_commands.json not found at "
              << compile_commands_path
              << "; falling back to default parser arguments\n";
    return {};
  }
  std::ifstream input(compile_commands_path);
  if (!input) {
    throw std::runtime_error("failed to open compile_commands.json at " +
                             compile_commands_path.string());
  }
  nlohmann::json doc = nlohmann::json::parse(input, nullptr, true, true);
  if (!doc.is_array()) {
    throw std::runtime_error("compile_commands.json must be an array");
  }

  std::vector<CompileCommandEntry> entries;
  for (const auto &entry : doc) {
    if (!entry.contains("file")) {
      continue;
    }
    fs::path file_path = entry.at("file").get<std::string>();
    std::vector<std::string> args;
    if (entry.contains("arguments") && entry.at("arguments").is_array()) {
      for (const auto &arg : entry.at("arguments")) {
        args.push_back(arg.get<std::string>());
      }
    } else if (entry.contains("command")) {
      const auto &command_string =
          entry.at("command").get_ref<const std::string &>();
      std::vector<std::string> tokens;
      tokens.reserve(32);
      std::string current;
      bool in_single = false;
      bool in_double = false;
      bool escape = false;
      for (char ch : command_string) {
        if (escape) {
          current.push_back(ch);
          escape = false;
          continue;
        }
        if (ch == '\\') {
          escape = true;
          continue;
        }
        if (ch == '\'' && !in_double) {
          in_single = !in_single;
          continue;
        }
        if (ch == '"' && !in_single) {
          in_double = !in_double;
          continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch)) && !in_single &&
            !in_double) {
          if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
          }
          continue;
        }
        current.push_back(ch);
      }
      if (!current.empty()) {
        tokens.push_back(current);
      }
      args = std::move(tokens);
    }

    if (args.empty()) {
      continue;
    }

    entries.push_back(
        CompileCommandEntry{normalize_path(file_path), std::move(args)});
  }
  return entries;
}

std::vector<std::string>
compile_arguments_for(const fs::path &source_file,
                      const std::vector<CompileCommandEntry> &entries) {
  const auto normalized_source = normalize_path(source_file);
  for (const auto &entry : entries) {
    if (entry.file != normalized_source) {
      continue;
    }
    std::vector<std::string> filtered;
    const auto &args = entry.arguments;
    if (args.empty()) {
      continue;
    }
    // Skip compiler binary
    for (std::size_t i = 1; i < args.size(); ++i) {
      const auto &token = args[i];
      if (token == "-c" || token == "-o") {
        ++i;
        continue;
      }
      if (token.rfind("-o", 0) == 0 && token.size() > 2) {
        continue;
      }
      filtered.push_back(token);
    }
    if (!filtered.empty()) {
      return filtered;
    }
  }
  return {"-std=c++20"};
}

std::string prefix_from_filename(const fs::path &path) {
  const auto stem = path.stem().string();
  const auto pos = stem.find('_');
  if (pos == std::string::npos || pos == 0) {
    throw std::runtime_error("unable to derive prefix from filename: " +
                             path.string());
  }
  return stem.substr(0, pos);
}

struct CursorCollector {
  std::string prefix;
  std::vector<FunctionSignature> *functions;
};

CXChildVisitResult collect_functions(CXCursor cursor, CXCursor parent,
                                     CXClientData client_data) {
  (void)parent;
  auto *collector = static_cast<CursorCollector *>(client_data);

  if (clang_Location_isFromMainFile(clang_getCursorLocation(cursor)) == 0) {
    return CXChildVisit_Recurse;
  }

  if (clang_getCursorKind(cursor) == CXCursor_FunctionDecl &&
      clang_isCursorDefinition(cursor) &&
      clang_getCursorLinkage(cursor) == CXLinkage_External) {
    const std::string name = normalize_exported_name(
        collector->prefix, to_string(clang_getCursorSpelling(cursor)));
    if (name.rfind(collector->prefix, 0) != 0) {
      return CXChildVisit_Recurse;
    }

    FunctionSignature signature;
    signature.name = name;
    signature.return_type =
        to_string(clang_getTypeSpelling(clang_getCursorResultType(cursor)));

    const int param_count = clang_Cursor_getNumArguments(cursor);
    for (int idx = 0; idx < param_count; ++idx) {
      CXCursor arg_cursor = clang_Cursor_getArgument(cursor, idx);
      std::string param_name = to_string(clang_getCursorSpelling(arg_cursor));
      if (param_name.empty()) {
        param_name = "param" + std::to_string(idx);
      }
      std::string param_type =
          to_string(clang_getTypeSpelling(clang_getCursorType(arg_cursor)));
      signature.parameters.push_back(Parameter{param_name, param_type});
    }

    collector->functions->push_back(std::move(signature));
    return CXChildVisit_Continue;
  }

  return CXChildVisit_Recurse;
}

FakeMetadata
scan_fake_source(const fs::path &source_path,
                 const std::vector<CompileCommandEntry> &commands) {
  const auto args = compile_arguments_for(source_path, commands);

  std::vector<const char *> c_args;
  c_args.reserve(args.size());
  for (const auto &arg : args) {
    c_args.push_back(arg.c_str());
  }

  CXIndex index = clang_createIndex(0, 0);
  CXTranslationUnit tu = clang_parseTranslationUnit(
      index, source_path.c_str(), c_args.data(),
      static_cast<int>(c_args.size()), nullptr, 0, CXTranslationUnit_None);

  if (tu == nullptr) {
    clang_disposeIndex(index);
    throw std::runtime_error("failed to parse " + source_path.string());
  }

  // Ensure diagnostics are non-fatal
  const unsigned diag_count = clang_getNumDiagnostics(tu);
  for (unsigned i = 0; i < diag_count; ++i) {
    CXDiagnostic diag = clang_getDiagnostic(tu, i);
    const CXDiagnosticSeverity severity = clang_getDiagnosticSeverity(diag);
    if (severity >= CXDiagnostic_Error) {
      std::string message = to_string(clang_formatDiagnostic(
          diag, clang_defaultDiagnosticDisplayOptions()));
      clang_disposeDiagnostic(diag);
      clang_disposeTranslationUnit(tu);
      clang_disposeIndex(index);
      throw std::runtime_error("clang error while parsing " +
                               source_path.string() + ":\n" + message);
    }
    clang_disposeDiagnostic(diag);
  }

  std::vector<FunctionSignature> functions;
  CursorCollector collector{prefix_from_filename(source_path), &functions};
  clang_visitChildren(clang_getTranslationUnitCursor(tu), collect_functions,
                      &collector);
  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);

  std::sort(functions.begin(), functions.end(),
            [](const FunctionSignature &lhs, const FunctionSignature &rhs) {
              return lhs.name < rhs.name;
            });

  if (functions.empty()) {
    throw std::runtime_error("no matching functions discovered in " +
                             source_path.string());
  }

  return FakeMetadata{collector.prefix, source_path, std::move(functions)};
}

nlohmann::json load_existing_config(const fs::path &path) {
  if (!fs::exists(path)) {
    return nlohmann::json::object();
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config " + path.string());
  }
  return nlohmann::json::parse(input, nullptr, true, true);
}

std::unordered_map<std::string, std::string>
event_kind_lookup(const nlohmann::json &existing) {
  std::unordered_map<std::string, std::string> mapping;
  if (!existing.contains("apis") || !existing["apis"].is_array()) {
    return mapping;
  }
  for (const auto &entry : existing["apis"]) {
    if (!entry.contains("name") || !entry.contains("event_kind")) {
      continue;
    }
    mapping.emplace(entry["name"].get<std::string>(),
                    entry["event_kind"].get<std::string>());
  }
  return mapping;
}

std::vector<std::string> default_includes(const std::string &prefix) {
  return {"\"lowlevel/interface/wrapper_template.hpp\"", "<" + prefix + ".h>"};
}

std::string make_declaration(const std::string &type, const std::string &name) {
  if (const auto star_paren = type.find("(*)");
      star_paren != std::string::npos) {
    auto result = type;
    result.replace(star_paren, 3, "(*" + name + ")");
    return result;
  }
  const auto pos = type.find("(*");
  if (pos != std::string::npos) {
    auto result = type;
    result.insert(pos + 2, name);
    return result;
  }
  if (const auto open = type.find('('); open != std::string::npos) {
    const auto close = type.find(')', open);
    const auto star = type.find('*', open);
    if (star != std::string::npos && close != std::string::npos &&
        star < close) {
      auto result = type;
      result.insert(star + 1, name);
      return result;
    }
  }
  return type + " " + name;
}

nlohmann::json render_config(const FakeMetadata &metadata,
                             const nlohmann::json &existing) {
  const auto kind_lookup = event_kind_lookup(existing);

  nlohmann::json apis = nlohmann::json::array();
  for (const auto &fn : metadata.functions) {
    nlohmann::json params = nlohmann::json::array();
    for (const auto &param : fn.parameters) {
      params.push_back(
          {{"name", param.name},
           {"declaration", make_declaration(param.type, param.name)},
           {"type", param.type}});
    }
    const auto kind_it = kind_lookup.find(fn.name);
    const std::string &event_kind =
        kind_it != kind_lookup.end() ? kind_it->second : "Unknown";
    apis.push_back({{"name", fn.name},
                    {"return_type", fn.return_type},
                    {"event_kind", event_kind},
                    {"parameters", params}});
  }

  nlohmann::json config = nlohmann::json::object();
  config["api_set"] = metadata.prefix;
  config["output"] = existing.contains("output")
                         ? existing["output"].get<std::string>()
                         : ("generated/" + metadata.prefix + "_wrappers.cpp");
  if (existing.contains("includes") && existing["includes"].is_array()) {
    config["includes"] = existing["includes"];
  } else {
    config["includes"] = default_includes(metadata.prefix);
  }
  config["apis"] = std::move(apis);
  return config;
}

void write_config(const fs::path &path, const nlohmann::json &payload) {
  if (path.has_parent_path()) {
    fs::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write config " + path.string());
  }
  output << payload.dump(4) << '\n';
}

Options parse_arguments(int argc, char **argv) {
  Options opts;
  const auto cwd = fs::current_path();
  opts.fakes_dir = normalize_path(cwd / "lowlevel-interface" / "fakes");
  opts.config_dir = normalize_path(cwd / "lowlevel-interface" / "config");
  opts.compile_commands = normalize_path(cwd / "compile_commands.json");

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--fakes-dir" && i + 1 < argc) {
      opts.fakes_dir = normalize_path(argv[++i]);
    } else if (arg == "--config-dir" && i + 1 < argc) {
      opts.config_dir = normalize_path(argv[++i]);
    } else if (arg == "--compile-commands" && i + 1 < argc) {
      opts.compile_commands = normalize_path(argv[++i]);
    } else if (arg == "--overwrite") {
      opts.overwrite = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: scan_fakes [--fakes-dir DIR] [--config-dir DIR] "
                   "[--compile-commands PATH] [--overwrite]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      std::cerr << "Use --help for usage information.\n";
      std::exit(1);
    }
  }
  return opts;
}

void run(const Options &opts) {
  if (!fs::exists(opts.fakes_dir)) {
    throw std::runtime_error("fakes directory not found: " +
                             opts.fakes_dir.string());
  }

  const auto compile_commands = parse_compile_commands(opts.compile_commands);

  std::vector<FakeMetadata> metadata;
  for (const auto &entry : fs::directory_iterator(
           opts.fakes_dir, fs::directory_options::skip_permission_denied)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() != ".cpp") {
      continue;
    }
    metadata.push_back(scan_fake_source(entry.path(), compile_commands));
  }

  if (metadata.empty()) {
    throw std::runtime_error("no fake sources discovered under " +
                             opts.fakes_dir.string());
  }

  for (const auto &fake : metadata) {
    const fs::path config_path =
        opts.config_dir / (fake.prefix + "_wrappers.json");
    const auto existing = opts.overwrite ? nlohmann::json::object()
                                         : load_existing_config(config_path);
    const auto payload = render_config(fake, existing);
    if (!opts.overwrite && !existing.empty() && payload == existing) {
      continue;
    }
    write_config(config_path, payload);
    std::cout << "Wrote " << config_path << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opts = parse_arguments(argc, argv);
    run(opts);
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "error: " << exc.what() << '\n';
    return 1;
  }
}
