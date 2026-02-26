using Microsoft.ML.Tokenizers;
using SharpLlmTensors.Shared;
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace SharpLlmTensors.Runtime
{
    public partial class TorchService
    {
        private Tokenizer? _activeTokenizer;

        public bool IsTokenizerLoaded => this._activeTokenizer != null;

        /// <summary>
        /// Initializes the tokenizer based on the provided token file path. The method supports both SentencePiece (.model) and Byte-Pair Encoding (BPE) tokenizers, depending on the file format. It logs the initialization process and handles any exceptions that may occur during loading.
        /// </summary>
        public async Task InitializeTokenizerAsync(string vocabFilePath, string? mergesFilePath = null)
        {
            await StaticLogger.LogAsync($"[TorchService] Initializing tokenizer from: {Path.GetFileName(vocabFilePath)} (full: {vocabFilePath}), merges: {mergesFilePath}");

            try
            {
                // FALL 1: Klassisches SentencePiece (.model) -> Llama & Gemma
                // Das funktioniert garantiert, da LlamaTokenizer.Create statisch existiert!
                if (vocabFilePath.EndsWith(".model", StringComparison.OrdinalIgnoreCase))
                {
                    using Stream modelStream = File.OpenRead(vocabFilePath);
                    this._activeTokenizer = LlamaTokenizer.Create(modelStream);
                }
                // FALL 2: Klassisches BPE Paar (vocab.json + merges.txt)
                else if (!string.IsNullOrWhiteSpace(mergesFilePath) && File.Exists(mergesFilePath))
                {
                    using Stream vocabStream = File.OpenRead(vocabFilePath);
                    using Stream mergesStream = File.OpenRead(mergesFilePath);

                    this._activeTokenizer = await BpeTokenizer.CreateAsync(
                        vocabStream: vocabStream,
                        mergesStream: mergesStream,
                        preTokenizer: null
                    );
                }
                else
                {
                    // Case: tokenizer.json (HuggingFace) may contain a BPE model embedded
                    if (vocabFilePath.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && File.Exists(vocabFilePath))
                    {
                        try
                        {
                            var json = await File.ReadAllTextAsync(vocabFilePath).ConfigureAwait(false);
                            using var doc = JsonDocument.Parse(json);
                            if (doc.RootElement.TryGetProperty("model", out var modelEl))
                            {
                                var type = modelEl.TryGetProperty("type", out var t) ? t.GetString() : null;

                                // 1) Check for external file references inside tokenizer.json (common case)
                                if (!string.IsNullOrWhiteSpace(type) && type.Equals("BPE", StringComparison.OrdinalIgnoreCase))
                                {
                                    string? dir = Path.GetDirectoryName(vocabFilePath);
                                    string? externalVocabPath = null;
                                    string? externalMergesPath = null;

                                    if (modelEl.TryGetProperty("vocab", out var vocabRef))
                                    {
                                        if (vocabRef.ValueKind == JsonValueKind.String)
                                        {
                                            externalVocabPath = Path.IsPathRooted(vocabRef.GetString()) ? vocabRef.GetString() : Path.Combine(dir ?? string.Empty, vocabRef.GetString() ?? string.Empty);
                                        }
                                    }

                                    if (modelEl.TryGetProperty("merges", out var mergesRef))
                                    {
                                        if (mergesRef.ValueKind == JsonValueKind.String)
                                        {
                                            externalMergesPath = Path.IsPathRooted(mergesRef.GetString()) ? mergesRef.GetString() : Path.Combine(dir ?? string.Empty, mergesRef.GetString() ?? string.Empty);
                                        }
                                    }

                                    if (!string.IsNullOrWhiteSpace(externalVocabPath) && !string.IsNullOrWhiteSpace(externalMergesPath)
                                        && File.Exists(externalVocabPath) && File.Exists(externalMergesPath))
                                    {
                                        await StaticLogger.LogAsync($"[TorchService] Found external vocab+merges in tokenizer.json: {externalVocabPath}, {externalMergesPath}");
                                        using Stream vocabStream = File.OpenRead(externalVocabPath);
                                        using Stream mergesStream = File.OpenRead(externalMergesPath);
                                        this._activeTokenizer = await BpeTokenizer.CreateAsync(vocabStream, mergesStream, preTokenizer: null).ConfigureAwait(false);
                                        await StaticLogger.LogAsync("[TorchService] Tokenizer successfully loaded from tokenizer.json referencing external files (BPE).");
                                        return;
                                    }

                                    // 2) Fall back: embedded vocab + merges arrays in tokenizer.json
                                    if (modelEl.TryGetProperty("vocab", out var vocabEl) && modelEl.TryGetProperty("merges", out var mergesEl))
                                    {
                                        // create temp files for vocab and merges
                                        string tempVocab = Path.Combine(Path.GetTempPath(), $"tokenizer_vocab_{Guid.NewGuid():N}.json");
                                        string tempMerges = Path.Combine(Path.GetTempPath(), $"tokenizer_merges_{Guid.NewGuid():N}.txt");
                                        try
                                        {
                                            await File.WriteAllTextAsync(tempVocab, vocabEl.GetRawText()).ConfigureAwait(false);

                                            var sb = new StringBuilder();
                                            foreach (var item in mergesEl.EnumerateArray())
                                            {
                                                if (item.ValueKind == JsonValueKind.String)
                                                {
                                                    sb.AppendLine(item.GetString());
                                                }
                                                else if (item.ValueKind == JsonValueKind.Array)
                                                {
                                                    var parts = item.EnumerateArray().Select(x => x.GetString());
                                                    sb.AppendLine(string.Join(' ', parts));
                                                }
                                            }

                                            await File.WriteAllTextAsync(tempMerges, sb.ToString()).ConfigureAwait(false);

                                            using Stream vocabStream = File.OpenRead(tempVocab);
                                            using Stream mergesStream = File.OpenRead(tempMerges);

                                            this._activeTokenizer = await BpeTokenizer.CreateAsync(
                                                vocabStream: vocabStream,
                                                mergesStream: mergesStream,
                                                preTokenizer: null
                                            ).ConfigureAwait(false);

                                            await StaticLogger.LogAsync("[TorchService] Tokenizer successfully loaded from tokenizer.json (embedded BPE).");
                                            return;
                                        }
                                        finally
                                        {
                                            try { File.Delete(tempVocab); } catch { }
                                            try { File.Delete(tempMerges); } catch { }
                                        }
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            await StaticLogger.LogAsync($"[TorchService] ERROR parsing tokenizer.json: {ex.Message}");
                        }
                    }

                    throw new Exception("No compatible tokenizer file format found (.model or vocab/merges pair required).");
                }

                await StaticLogger.LogAsync("[TorchService] Tokenizer successfully loaded.");
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"[TorchService] ERROR loading tokenizer: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Transforms a human-readable string into an array of token IDs that the model can process. This is a critical step before feeding input into the model for inference or training.
        /// </summary>
        public long[] Tokenize(string text)
        {
            if (this._activeTokenizer == null)
            {
                throw new InvalidOperationException("Tokenizer is not loaded. Please load the model first.");
            }

            // FIX 2: Nutze EncodeToIds statt Encode.
            // Das liefert direkt eine IReadOnlyList<int> zurück.
            var ids = this._activeTokenizer.EncodeToIds(text);

            // Konvertiere ints zu longs, da TorchSharp Embedding-Layer int64 (long) erwarten
            return ids.Select(id => (long) id).ToArray();
        }

        /// <summary>
        /// Transforms token IDs back into a human-readable string. This is useful for decoding model outputs or for debugging purposes.
        /// </summary>
        public string Detokenize(long[] tokenIds)
        {
            if (this._activeTokenizer == null)
            {
                throw new InvalidOperationException("Tokenizer is not loaded. Please load the model first.");
            }

            // ML.Tokenizers Decode erwartet IEnumerable<int>
            int[] intIds = tokenIds.Select(id => (int) id).ToArray();

            // Konvertiert die IDs wieder in den String
            return this._activeTokenizer.Decode(intIds) ?? string.Empty;
        }

        /// <summary>
        /// Cleanup the tokenizer instance to free up resources. This should be called when unloading the model or when the service is shutting down.
        /// </summary>
        private void UnloadTokenizer()
        {
            this._activeTokenizer = null;
        }


    }
}
