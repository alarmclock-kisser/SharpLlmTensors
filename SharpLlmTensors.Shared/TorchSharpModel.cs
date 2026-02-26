using System;
using System.Collections.Generic;
using System.Text;
using System.Globalization;
using System.Text.RegularExpressions;
using System.IO;
using System.Reflection;

namespace SharpLlmTensors.Shared
{
    public class TorchSharpModel
    {
        public string ModelName { get; set; }
        public string ModelRootDirectory { get; set; }

        public double ModelSizeInMb { get; set; }
        public double BillionParameters { get; set; }

        public DateTime LastModfified { get; set; }

        public List<string> ModelFilePaths { get; set; }
        public TorchSharpModelJsonFiles ConfigJsonFiles { get; set; }



        public TorchSharpModel(string modelRootDirectory)
        {
            if (!Directory.Exists(modelRootDirectory))
            {
                throw new Exception($"Directory not found: {modelRootDirectory}");
            }

            this.ModelRootDirectory = modelRootDirectory;
            this.ModelName = Path.GetFileName(modelRootDirectory);

            // 1. SCANNEN: Wir holen uns ALLE Dateien im Ordner
            var allFiles = Directory.GetFiles(modelRootDirectory, "*.*", SearchOption.AllDirectories).ToList();

            // 2. FILTERN: Nur die Gewichte kommen in die ModelFilePaths
            this.ModelFilePaths = allFiles.Where(f => f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)).ToList();
            this.ModelSizeInMb = this.ModelFilePaths.Sum(f => new FileInfo(f).Length) / (1024.0 * 1024.0);

            if (this.ModelFilePaths.Count == 0)
            {
                throw new Exception("No .safetensors found!");
            }

            this.LastModfified = allFiles.Select(File.GetLastWriteTime).Max();

            this.BillionParameters = TryParseBillionParametersCountFromName(this.ModelName);

            // 3. ZUWEISEN: Jetzt suchen wir in 'allFiles' nach den Configs
            this.ConfigJsonFiles = new TorchSharpModelJsonFiles(modelRootDirectory);
            foreach (var file in allFiles)
            {
                string fileName = Path.GetFileName(file).ToLower();
                if (fileName == "config.json")
                {
                    this.ConfigJsonFiles.ConfigJsonFilePath = file;
                }

                // Eindeutige Zuordnung:
                else if (fileName == "tokenizer.model")
                {
                    this.ConfigJsonFiles.TokenizerModelFilePath = file; // Hier gehört es hin!
                }
                else if (fileName == "tokenizer.json")
                {
                    this.ConfigJsonFiles.TokenizerJsonFilePath = file; // HF-Format
                }
                else if (fileName == "vocab.json")
                {
                    this.ConfigJsonFiles.VocabJsonFilePath = file; // Nur wenn es echtes vocab.json ist
                }
                else if (fileName == "merges.txt")
                {
                    this.ConfigJsonFiles.MergesTxtFilePath = file;
                }
                else if (fileName == "tokenizer_config.json")
                {
                    this.ConfigJsonFiles.TokenizerConfigJsonFilePath = file;
                }
                else if (fileName == "chat_template.jinja" || fileName == "chat_template.json")
                {
                    this.ConfigJsonFiles.ChatTemplateJsonOrJinjaFilePath = file;
                }
            }
        }


        public static double TryParseBillionParametersCountFromName(string modelName)
        {
            // Try to parse the number of parameters from the model name (e.g., "7B", "13B", "70B", "100B", "1.2B", "0.8B", "258M" -> 0.258, etc.)
            if (string.IsNullOrWhiteSpace(modelName))
            {
                return 0.0;
            }

            // Match numbers with optional decimal separators (dot or comma) followed by B or M
            var regex = new Regex(@"(?<num>\d+(?:[\.,]\d+)?)(?<unit>[BbMm])\b", RegexOptions.Compiled);
            var matches = regex.Matches(modelName);
            double maxBillion = 0.0;

            foreach (Match match in matches)
            {
                var numPart = match.Groups["num"].Value;
                var unit = match.Groups["unit"].Value;

                // Normalize comma to dot for invariant parsing
                numPart = numPart.Replace(',', '.');
                if (double.TryParse(numPart, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double value))
                {
                    double asBillion = 0.0;
                    if (unit.Equals("B", StringComparison.OrdinalIgnoreCase))
                    {
                        asBillion = value;
                    }
                    else if (unit.Equals("M", StringComparison.OrdinalIgnoreCase))
                    {
                        asBillion = value / 1000.0;
                    }

                    if (asBillion > maxBillion)
                    {
                        maxBillion = asBillion;
                    }
                }
            }

            return maxBillion; // 0 if not found or could not parse
        }
    }


    public class TorchSharpModelJsonFiles
    {
        public int Count => this.GetValidFilesCount();

        public string? AddedTokensJsonFilePath { get; set; }
        public string? ChatTemplateJsonOrJinjaFilePath { get; set; }
        public string ConfigJsonFilePath { get; set; }
        public string? GenerationConfigJsonFilePath { get; set; }
        public string? MergesTxtFilePath { get; set; }
        public string? PreprocessorConfigJsonFilePath { get; set; }
        public string? ProcessorConfigJsonFilePath { get; set; }
        public string? SpecialTokensMapJsonFilePath { get; set; }
        public string TokenizerJsonFilePath { get; set; }
        public string? TokenizerModelFilePath { get; set; }
        public string TokenizerConfigJsonFilePath { get; set; }
        public string? VideoPreprocessorConfigJsonFilePath { get; set; }
        public string? VocabJsonFilePath { get; set; }



        public TorchSharpModelJsonFiles(string modelRootDirectory)
        {
            if (!Directory.Exists(modelRootDirectory))
            {
                throw new Exception($"Model root directory does not exist: {modelRootDirectory}");
            }
            var jsonFiles = Directory.GetFiles(modelRootDirectory, "*.json", SearchOption.AllDirectories).ToList();
            
            this.ConfigJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("config.json", StringComparison.OrdinalIgnoreCase)) ?? throw new Exception($"Config JSON file not found in directory: {modelRootDirectory}");
            this.TokenizerJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("tokenizer.json", StringComparison.OrdinalIgnoreCase)) ?? jsonFiles.FirstOrDefault(f => Path.GetFileName(f).EndsWith(".model", StringComparison.OrdinalIgnoreCase)) ?? throw new Exception($"Tokenizer .json or .model file not found in directory: {modelRootDirectory}");
            this.TokenizerModelFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).EndsWith(".model", StringComparison.OrdinalIgnoreCase));
            this.TokenizerConfigJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("tokenizer_config.json", StringComparison.OrdinalIgnoreCase)) ?? throw new Exception($"Tokenizer Config JSON file not found in directory: {modelRootDirectory}");

            this.GenerationConfigJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("generation_config.json", StringComparison.OrdinalIgnoreCase));
            this.SpecialTokensMapJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("special_tokens_map.json", StringComparison.OrdinalIgnoreCase));
            this.AddedTokensJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("added_tokens.json", StringComparison.OrdinalIgnoreCase));
            this.VocabJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("vocab.json", StringComparison.OrdinalIgnoreCase));
            this.MergesTxtFilePath = Directory.GetFiles(modelRootDirectory, "merges.txt", SearchOption.AllDirectories).FirstOrDefault();
            this.ChatTemplateJsonOrJinjaFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("chat_template.json", StringComparison.OrdinalIgnoreCase)) ?? jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("chat_template.jinja", StringComparison.OrdinalIgnoreCase));
            this.ProcessorConfigJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("processor_config.json", StringComparison.OrdinalIgnoreCase));
            this.PreprocessorConfigJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("preprocessor_config.json", StringComparison.OrdinalIgnoreCase));
            this.VideoPreprocessorConfigJsonFilePath = jsonFiles.FirstOrDefault(f => Path.GetFileName(f).Equals("video_preprocessor_config.json", StringComparison.OrdinalIgnoreCase));
        }


        public int GetValidFilesCount()
        {
            try
            {
                int count = 0;
                var type = this.GetType();

                // Wir prüfen sowohl Properties als auch Fields.
                // Backing Fields von Auto-Properties sind NonPublic!
                var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

                // 1. Alle Properties prüfen (die du definiert hast)
                var properties = type.GetProperties(flags);
                foreach (var prop in properties)
                {
                    if (prop.PropertyType == typeof(string))
                    {
                        var val = prop.GetValue(this) as string;
                        if (!string.IsNullOrWhiteSpace(val) && File.Exists(val))
                        {
                            count++;
                        }
                    }
                }

                // Falls du Felder direkt nutzt oder sichergehen willst, 
                // dass nichts doppelt gezählt wird (Properties vs Backing Fields), 
                // ist der obige Weg über Properties für dein DTO am sichersten.

                return count;
            }
            catch (Exception ex)
            {
                // Annahme: StaticLogger ist in deinem Projekt verfügbar
                // StaticLogger.Log(ex); 
                Console.WriteLine($"Fehler beim Zählen der Dateien: {ex.Message}");
                return 0;
            }
        }
    }
}
