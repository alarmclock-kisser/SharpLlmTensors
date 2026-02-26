using Microsoft.ML.Tokenizers;
using SharpLlmTensors.Runtime.Models;
using SharpLlmTensors.Shared;
using System.Text.Json;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace SharpLlmTensors.Runtime
{
    public partial class TorchService
    {
        // Aktueller Zustand des geladenen Modells
        private TorchSharp.Modules.Sequential? _loadedModel;
        private Device? _currentDevice;

        private nn.Module? _activeModel;

        public double ModelLoadProgress { get; private set; } = 0.0;
        public bool IsModelLoaded => this._loadedModel != null;



        public async Task<TorchSharpModel?> LoadModelAsync(TorchSharpModelLoadRequest loadRequest, ScalarType scalarType = ScalarType.Float16)
        {
            try
            {
                this.ModelLoadProgress = 0.0;
                var m = loadRequest.Model;
                loadRequest.ScalarT = scalarType.ToString();

                this._currentDevice = (loadRequest.ForceCpu || !cuda.is_available()) ? CPU : CUDA;
                await StaticLogger.LogAsync($"[TorchService] Initializing model on {this._currentDevice.type}...");

                if (!File.Exists(m.ConfigJsonFiles.ConfigJsonFilePath))
                {
                    throw new FileNotFoundException("Critical file missing: config.json");
                }

                var configJson = await File.ReadAllTextAsync(m.ConfigJsonFiles.ConfigJsonFilePath);
                var config = JsonSerializer.Deserialize<JsonElement>(configJson);
                string modelType = config.TryGetProperty("model_type", out var typeProp) ? typeProp.GetString()! : "llama";

                this.ModelLoadProgress = 0.1;

                // 1. Initialisiere den Tokenizer
                // Tokenizer-Ladepriorität festlegen
                string? finalVocabPath = null;
                string? finalMergesPath = null;

                var cf = m.ConfigJsonFiles;

                if (!string.IsNullOrEmpty(cf.TokenizerModelFilePath) && File.Exists(cf.TokenizerModelFilePath))
                {
                    // Prio 1: .model (Gemma/Llama Standard)
                    finalVocabPath = cf.TokenizerModelFilePath;
                }
                else if (!string.IsNullOrEmpty(cf.VocabJsonFilePath) && !string.IsNullOrEmpty(cf.MergesTxtFilePath)
                         && File.Exists(cf.VocabJsonFilePath) && File.Exists(cf.MergesTxtFilePath))
                {
                    // Prio 2: Legacy vocab.json + merges.txt (prefer this if present)
                    finalVocabPath = cf.VocabJsonFilePath;
                    finalMergesPath = cf.MergesTxtFilePath;
                }
                else if (!string.IsNullOrEmpty(cf.TokenizerJsonFilePath) && File.Exists(cf.TokenizerJsonFilePath))
                {
                    // Prio 3: tokenizer.json (HuggingFace)
                    finalVocabPath = cf.TokenizerJsonFilePath;
                }

                if (!string.IsNullOrEmpty(finalVocabPath))
                {
                    // Jetzt rufen wir Initialize mit dem EINDEUTIGEN Pfad auf
                    await StaticLogger.LogAsync($"[TorchService] Selected tokenizer files -> vocab/tokenizer: {finalVocabPath}, merges: {finalMergesPath}");
                    await this.InitializeTokenizerAsync(finalVocabPath, finalMergesPath);
                }
                else
                {
                    await StaticLogger.LogAsync("[TorchService] WARNING: No tokenizer files found at all!");
                    await this.UnloadModelAsync().ConfigureAwait(false);
                    await Task.Run(this.UnloadTokenizer).ConfigureAwait(false);
                    return null;
                }

                string tokenizerConfigPath = m.ModelFilePaths.FirstOrDefault(p => Path.GetFileName(p).Equals("tokenizer_config.json", StringComparison.OrdinalIgnoreCase)) ?? "";
                if (!string.IsNullOrEmpty(tokenizerConfigPath))
                {
                    await this.InitializeChatTemplateAsync(tokenizerConfigPath);
                }

                // Create instance using the factory
                this._activeModel = this.CreateModelInstance(modelType, config);

                // Push to cuda as scalar type (e.g., float16) to save memory (((but keep in mind that some operations might not be supported in lower precision)))
                this._activeModel.to(this._currentDevice, scalarType);
                await StaticLogger.LogAsync($"[TorchService] Model instance created and moved to {this._currentDevice.type} with scalar type {scalarType}.");

                this.ModelLoadProgress = 0.2;

                foreach (var (name, _) in this._activeModel.named_parameters())
                {
                    await StaticLogger.LogAsync($"Expected: {name}");
                }

                var sortedShards = m.ModelFilePaths.OrderBy(f => f).ToList();
                double progressStep = 0.75 / sortedShards.Count;

                foreach (var path in sortedShards)
                {
                    await StaticLogger.LogAsync($"[TorchService] Loading weights from: {Path.GetFileName(path)}");

                    // Native loading via Safetensors
                    this._activeModel.load_safetensors(path, loadRequest.StrictLoadingMode);

                    this.ModelLoadProgress += progressStep;
                }

                this._activeModel.eval();
                this.ModelLoadProgress = 1.0;
                await StaticLogger.LogAsync($"[TorchService] Model '{m.ModelName}' successfully loaded.");

                return m;
            }
            catch (Exception ex)
            {
                this.ModelLoadProgress = 0.0;
                await StaticLogger.LogAsync($"[TorchService] ERROR during load: {ex.Message}");
                await this.UnloadModelAsync();
                throw new Exception($"Failed to load model: {ex.Message}", ex);
            }
        }

        public async Task<bool?> UnloadModelAsync()
        {
            if (this._loadedModel == null)
            {
                return null;
            }

            try
            {
                this._loadedModel.Dispose();
                this._loadedModel = null;
                this._currentDevice = null;

                this.UnloadTokenizer();

                // Garbage Collection forcieren für GPU Speicher
                GC.Collect();
                GC.WaitForPendingFinalizers();

                if (cuda.is_available())
                {
                    cuda.synchronize();
                }

                this.ModelLoadProgress = 0.0;
                await StaticLogger.LogAsync($"[TorchService] Model successfully unloaded and resources freed.");
                return true;
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"[TorchService] ERROR during unload: {ex.Message}");
                return false;
            }
        }





        // Helpers (private)
        private async Task<JsonElement> LoadJsonAsync(string path)
        {
            using var stream = File.OpenRead(path);
            return await JsonSerializer.DeserializeAsync<JsonElement>(stream);
        }

        private nn.Module CreateModelInstance(string modelType, JsonElement config)
        {
            string type = modelType.ToLower();

            return type switch
            {
                // Qwen Series (including Qwen2, Qwen2.5, and Qwen3)
                "qwen2_vl" or "qwen2_5_vl" or "qwen3_vl" => new Qwen2VLModel(config),
                "qwen2" or "qwen2.5" or "qwen3" => new LlamaModel(config),

                // Gemma Series (including Gemma 2 and Gemma 3)
                "gemma2" or "gemma3" or "gemma3_text" => new GemmaModel(config),

                // Granite & Docling
                "granite" or "granitemoehybrid" => new LlamaModel(config),
                "granite-docling" or "docling" => new LlamaModel(config),

                // Default Llama
                "llama" => new LlamaModel(config),

                // LFM2
                "lfm2" => new LFM2Model(config),

                _ => throw new NotSupportedException($"Architecture '{modelType}' is not supported yet.")
            };
        }

    }
}