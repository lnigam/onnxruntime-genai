// -----------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// C++ API Example: Model Compile
// Runs the same model under different execution provider and compile configurations:
// 1) CPU only (no compile overlay)
// 2) CPU with compile config via config_overlay
// 3) NvTensorRtRtx EP without compile options
// 4) NvTensorRtRtx EP with 4 compile options (enable_ep_context, ep_context_embed_mode,
//    force_compile_if_needed, graph_optimization_level)
// 5) NvTensorRtRtx EP with all compile options
//
// ONNX Runtime log level:
//  - Global: use -d/--debug (sets ORTGENAI_ORT_VERBOSE_LOGGING=1) or set that env var before
//    launching. The GenAI library uses it when creating the ORT environment.
//  - Per-session: overlay config with "model": {"decoder": {"session_options": {"log_severity_level": 0}}}
//    (0=Verbose, 1=Info, 2=Warning, 3=Error).
// -----------------------------------------------------------------------------------------------

#include <chrono>
#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.h"

namespace fs = std::filesystem;

// Enable ONNX Runtime verbose logging. Must be set before any Oga/ORT API use.
// Alternatively set env ORTGENAI_ORT_VERBOSE_LOGGING=1 before launching.
static void SetOrtVerboseLogging() {
#ifdef _WIN32
  _putenv("ORTGENAI_ORT_VERBOSE_LOGGING=1");
#else
  setenv("ORTGENAI_ORT_VERBOSE_LOGGING", "1", 1);
#endif
}

static const char* kCpuEp = "cpu";
static const char* kNvTensorRtRtxEp = "NvTensorRtRtx";

static const char* kDefaultPrompt = "Tell me about AI and ML";

// Runs one short generation to verify the model works. Prints prompt and inference output. Returns inference time in seconds.
static double RunOneGeneration(OgaModel& model, OgaTokenizer& tokenizer, bool verbose) {
  auto stream = OgaTokenizerStream::Create(tokenizer);
  auto sequences = OgaSequences::Create();
  tokenizer.Encode(kDefaultPrompt, *sequences);

  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("max_length", 128);
  params->SetSearchOption("batch_size", 1);

  auto generator = OgaGenerator::Create(model, *params);
  generator->AppendTokenSequences(*sequences);

  if (verbose) std::cout << "Prompt: " << kDefaultPrompt << std::endl;
  std::cout << "Output: " << std::flush;
  auto t0 = Clock::now();
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    const auto new_token = generator->GetNextTokens()[0];
    std::cout << stream->Decode(new_token) << std::flush;
  }
  auto t1 = Clock::now();
  std::cout << std::endl;
  if (verbose) std::cout << "Generating done." << std::endl;
  return std::chrono::duration<double>(t1 - t0).count();
}

static void PrintTimings(const char* label, double load_time_sec, double inference_time_sec) {
  const auto default_precision = std::cout.precision();
  std::cout << "  " << label << ": "
            << std::fixed << std::setprecision(3)
            << "model load " << load_time_sec << "s, "
            << "inference " << inference_time_sec << "s"
            << std::setprecision(default_precision) << std::endl;
}

// 1) Run model with CPU execution provider only (no compile overlay).
void RunWithCpu(const std::string& model_path, const std::string& ep_path, bool verbose) {
  (void)ep_path;
  if (verbose) std::cout << "[RunWithCpu] Creating config (CPU, no compile overlay)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kCpuEp, ep_options, search_options);
  if (verbose) std::cout << "[RunWithCpu] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithCpu] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithCpu (CPU, no overlay)", load_time, inference_time);
  if (verbose) std::cout << "[RunWithCpu] OK." << std::endl;
}

// 2) Run model with CPU execution provider and compile config passed via config_overlay.
void RunWithCpuAndCompileOverlay(const std::string& model_path, const std::string& ep_path, bool verbose) {
  (void)ep_path;
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] Creating config (CPU + compile overlay)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kCpuEp, ep_options, search_options);
  config->Overlay(R"({
    "model": {
      "decoder": {
        "compile_options": {
          "enable_ep_context": true,
          "ep_context_embed_mode": false,
          "force_compile_if_needed": true,
          "graph_optimization_level": 99
        }
      }
    }
  })");
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithCpuAndCompileOverlay (CPU + overlay)", load_time, inference_time);
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] OK." << std::endl;
}

// 3) Run model with NvTensorRtRtx EP without compile options.
void RunWithNvTensorRtRtxNoCompile(const std::string& model_path, const std::string& ep_path, bool verbose) {
  if (ep_path.empty() && verbose) {
    std::cout << "Warning: --ep_path not set; NvTensorRTRTX may not be available (only CPU)." << std::endl;
  }
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] Creating config (NvTensorRtRtx, no compile)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kNvTensorRtRtxEp, ep_options, search_options);
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithNvTensorRtRtxNoCompile (NvTensorRtRtx, no compile)", load_time, inference_time);
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] OK." << std::endl;
}

// 4) Run model with NvTensorRtRtx EP and 4 compile options.
void RunWithNvTensorRtRtxCompileFourOptions(const std::string& model_path, const std::string& ep_path, bool verbose) {
  if (ep_path.empty() && verbose) {
    std::cout << "Warning: --ep_path not set; NvTensorRTRTX may not be available (only CPU)." << std::endl;
  }
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileFourOptions] Creating config (NvTensorRtRtx + 4 compile options)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kNvTensorRtRtxEp, ep_options, search_options);
  config->Overlay(R"({
    "model": {
      "decoder": {
        "compile_options": {
          "enable_ep_context": true,
          "ep_context_embed_mode": false,
          "force_compile_if_needed": true,
          "graph_optimization_level": 99
        }
      }
    }
  })");
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileFourOptions] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileFourOptions] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithNvTensorRtRtxCompileFourOptions (4 options)", load_time, inference_time);
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileFourOptions] OK." << std::endl;
}

// 5) Run model with NvTensorRtRtx EP and all compile options.
void RunWithNvTensorRtRtxCompileAllOptions(const std::string& model_path, const std::string& ep_path, bool verbose) {
  if (ep_path.empty() && verbose) {
    std::cout << "Warning: --ep_path not set; NvTensorRTRTX may not be available (only CPU)." << std::endl;
  }
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] Creating config (NvTensorRtRtx + all compile options)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kNvTensorRtRtxEp, ep_options, search_options);
  // Single config: ep_context_file_path is full path (relative to model dir) including filename, e.g. "contexts/model_ctx.onnx"
  config->Overlay(R"({
    "model": {
      "decoder": {
        "compile_options": {
          "enable_ep_context": true,
          "graph_optimization_level": 99,
          "ep_context_file_path": "contexts/ep_context_output/model_ctx.onnx",
          "ep_context_embed_mode": false,
          "force_compile_if_needed": true
        }
      }
    }
  })");
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithNvTensorRtRtxCompileAllOptions (all options)", load_time, inference_time);
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] OK." << std::endl;
}

int main(int argc, char** argv) {
  GeneratorParamsArgs generator_params_args;
  GuidanceArgs guidance_args;
  std::string model_path, ep = "follow_config", ep_path, system_prompt, user_prompt;
  bool verbose = true, debug = false, interactive = false, rewind = true;
  std::vector<std::string> image_paths, audio_paths;

  if (!ParseArgs(argc, argv, generator_params_args, guidance_args, model_path, ep, ep_path, system_prompt, user_prompt, verbose, debug, interactive, rewind, image_paths, audio_paths)) {
    return -1;
  }

  // If -e NvTensorRtRtx and --ep_path not set, default to current working directory + provider library name.
  if (ep.compare("NvTensorRtRtx") == 0 && ep_path.empty()) {
#if defined(_WIN32)
    ep_path = (fs::current_path() / "onnxruntime_providers_nv_tensorrt_rtx.dll").string();
#else
    ep_path = (fs::current_path() / "libonnxruntime_providers_nv_tensorrt_rtx.so").string();
#endif
  }

  // Enable ONNX Runtime verbose logs (global) before any Oga/ORT API is used.
  if (debug) {
    SetOrtVerboseLogging();
  }

  // Register NvTensorRTRTX EP once before any GenAI API that creates the OrtEnv, so that
  // GetEpDevices() (e.g. in ValidateCompiledModel) sees both CPU and the plugin EP.
  if (!ep_path.empty()) {
    RegisterEP(kNvTensorRtRtxEp, ep_path);
  }


  OgaHandle handle;

  std::cout << "--------------------------" << std::endl;
  std::cout << "ORT GenAI Model-Compile" << std::endl;
  std::cout << "--------------------------" << std::endl;
  std::cout << "Model path: " << model_path << std::endl;
  std::cout << "EP path (for NvTensorRtRtx): " << (ep_path.empty() ? "(none)" : ep_path) << std::endl;
  std::cout << "Verbose: " << verbose << std::endl;
  std::cout << "Debug (ORT verbose + model I/O): " << debug << std::endl;
  std::cout << "--------------------------" << std::endl;

  if (debug) SetLogger();

  std::cout << "Timings (model load, inference per case):" << std::endl;
  try {
    //RunWithCpu(model_path, ep_path, verbose);
    //RunWithCpuAndCompileOverlay(model_path, ep_path, verbose);
    RunWithNvTensorRtRtxNoCompile(model_path, ep_path, verbose);
    RunWithNvTensorRtRtxCompileFourOptions(model_path, ep_path, verbose);
    RunWithNvTensorRtRtxCompileFourOptions(model_path, ep_path, verbose);
    //RunWithNvTensorRtRtxCompileAllOptions(model_path, ep_path, verbose);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  std::cout << "All model-compile runs completed." << std::endl;
  return 0;
}
