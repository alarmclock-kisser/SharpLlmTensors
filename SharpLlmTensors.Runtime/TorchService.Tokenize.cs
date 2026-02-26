using Microsoft.ML.Tokenizers;
using SharpLlmTensors.Shared;
using System;
using System.Collections.Generic;
using System.Text;

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
            await StaticLogger.LogAsync($"[TorchService] Initializing tokenizer from: {Path.GetFileName(vocabFilePath)}");

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
