using SharpLlmTensors.Shared;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace SharpLlmTensors.Runtime
{
    public partial class TorchService
    {
        public static AppSettings AppSettings { get; private set; } = new();

        // Start with an empty list. Directories are supplied via AppSettings at startup or via API.
        public readonly BindingList<string> ModelDirectories = new BindingList<string>();
        public readonly BindingList<TorchSharpModel> ModelsBindingList = new();



        public TorchService(AppSettings appSettings)
        {
            AppSettings = appSettings ?? new AppSettings(); // AppSettings per Program.cs (DI)
            this.GetModels(AppSettings.ModelDirectories);
        }


        public List<TorchSharpModel> GetModels(IEnumerable<string>? modelDirectories = null, TorchModelsSortingOption sortingOption = TorchModelsSortingOption.Alphabetical)
        {
            if (modelDirectories != null)
            {
                foreach (var modelDirectory in modelDirectories)
                {
                    if (!this.ModelDirectories.Contains(modelDirectory))
                    {
                        this.ModelDirectories.Add(modelDirectory);
                    }
                }
            }

            // Build a safe list of candidate directories to search for .safetensors files.
            // Protect against missing/non-existing entries in ModelDirectories and IO exceptions
            var candidates = new List<string>();
            foreach (var baseDir in this.ModelDirectories)
            {
                if (string.IsNullOrWhiteSpace(baseDir))
                {
                    continue;
                }

                try
                {
                    if (Directory.Exists(baseDir))
                    {
                        // include the root itself (in case the user pointed directly at a model folder)
                        candidates.Add(baseDir);

                        // include immediate subdirectories (original behaviour)
                        foreach (var sd in Directory.EnumerateDirectories(baseDir))
                        {
                            candidates.Add(sd);
                        }
                    }
                    else
                    {
                        // Log missing directories but do not throw - this is a configuration issue
                        StaticLogger.Log($"[TorchService] Model directory does not exist: {baseDir}");
                    }
                }
                catch (Exception ex) when (ex is UnauthorizedAccessException || ex is DirectoryNotFoundException || ex is IOException)
                {
                    StaticLogger.Log($"[TorchService] Error accessing model directory '{baseDir}': {ex.Message}");
                    continue;
                }
            }

            string[] modelDirectoriesWithSafetensors = candidates
                // Use recursive search: model files may be located in nested folders under the model root
                .Where(dir => Directory.Exists(dir) && Directory.EnumerateFiles(dir, "*.safetensors", System.IO.SearchOption.AllDirectories).Any())
                .ToArray();

            foreach (var dir in modelDirectoriesWithSafetensors)
            {
                try
                {
                    TorchSharpModel model = new(dir);
                    if (!this.ModelsBindingList.Any(m => m.ModelRootDirectory == model.ModelRootDirectory))
                    {
                        this.ModelsBindingList.Add(model);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing model directory '{dir}': {ex.Message}");
                }
            }

            List<TorchSharpModel> sortedModels = sortingOption switch
            {
                TorchModelsSortingOption.Alphabetical => this.ModelsBindingList.OrderBy(m => m.ModelName).ToList(),
                TorchModelsSortingOption.ReversedAlphabetical => this.ModelsBindingList.OrderByDescending(m => m.ModelName).ToList(),
                TorchModelsSortingOption.BySizeBiggest => this.ModelsBindingList.OrderByDescending(m => m.ModelSizeInMb).ToList(),
                TorchModelsSortingOption.BySizeSmallest => this.ModelsBindingList.OrderBy(m => m.ModelSizeInMb).ToList(),
                TorchModelsSortingOption.ByLatestModified => this.ModelsBindingList.OrderByDescending(m => m.LastModfified).ToList(),
                TorchModelsSortingOption.ByOldestModified => this.ModelsBindingList.OrderBy(m => m.LastModfified).ToList(),
                TorchModelsSortingOption.ByMostParametes => this.ModelsBindingList.OrderByDescending(m => m.BillionParameters).ToList(),
                TorchModelsSortingOption.ByFewestParameters => this.ModelsBindingList.OrderBy(m => m.BillionParameters).ToList(),
                _ => this.ModelsBindingList.ToList()
            };

            return sortedModels;
        }


        public static void LogVerbose(string message)
        {
            if (AppSettings.VerboseLog)
            {
                StaticLogger.Log($" <VERBOSE> [TorchService] {message}");
            }
        }

        public static async Task LogVerboseAsync(string message, Exception? ex = null)
        {
            if (AppSettings.VerboseLog)
            {
                if (ex != null)
                {
                    await StaticLogger.LogAsync(ex, $" <VERBOSE> [TorchService] {message}", false);
                }
                else
                {
                    await StaticLogger.LogAsync($" <VERBOSE> [TorchService] {message}", false);
                }
            }
        }

    }

    public enum TorchModelsSortingOption
    {
        Alphabetical,
        ReversedAlphabetical,
        BySizeBiggest,
        BySizeSmallest,
        ByLatestModified,
        ByOldestModified,
        ByMostParametes,
        ByFewestParameters
    }
}
