using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using SharpLlmTensors.Monitoring;
using SharpLlmTensors.Runtime;
using SharpLlmTensors.Shared;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Text;

namespace SharpLlmTensors.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class TorchController : ControllerBase
    {
        private readonly AppSettings Settings;
        private readonly TorchService Service;

        private readonly GpuMonitor? GPUMonitor;
        public HardwareStatistics? HardwareStatsCache {  get; private set; }


        public TorchController(AppSettings appSettings, TorchService service, GpuMonitor? gpuMonitor = null)
        {
            this.Settings = appSettings;
            this.Service = service;
            this.GPUMonitor = gpuMonitor;
        }


        [HttpGet("appsettings")]
        public ActionResult<AppSettings> GetAppSettings()
        {
            return this.Ok(this.Settings);
        }

        [HttpGet("log")]
        public ActionResult<string[]> GetLog()
        {
            var logEntries = StaticLogger.LogEntriesBindingList.ToArray();
            return this.Ok(logEntries);
        }

        // Download latest / current log file endpoint
        [HttpGet("log-file/latest")]
        public IActionResult GetLatestLogFile()
        {
            string? latestLogFilePath = StaticLogger.LogFilePath;
            if (latestLogFilePath == null || !System.IO.File.Exists(latestLogFilePath))
            {
                return this.NotFound("No log file found.");
            }

            var fileStream = new FileStream(latestLogFilePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            var fileName = Path.GetFileName(latestLogFilePath);

            return this.File(fileStream, "text/plain", fileName);
        }

        [SupportedOSPlatform("windows")]
        [HttpGet("hw-stats")]
        public async Task<ActionResult<HardwareStatistics>?> GetHardwareStatsAsync()
        {
            try
            {
                if (this.GPUMonitor != null)
                {
                    HardwareStatistics hwStats = await this.GPUMonitor.GetCurrentHardwareStatisticsAsync();

                    this.HardwareStatsCache = hwStats;
                    GpuMonitor.HardwareStatsHistory[hwStats.CreatedAt] = hwStats;

                    return this.Ok(hwStats);
                }

                return this.NotFound();
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"Error retrieving hardware stats: {ex.Message}");
            }
        }

        [SupportedOSPlatform("windows")]
        [HttpGet("hw-stats-history")]
        public ActionResult<List<HardwareStatistics>> GetHardwareStatsHistory([FromQuery] bool fetchNew = false)
        {
            if (fetchNew)
            {
                // Trigger a new fetch of hardware stats to get the latest data point
                _ = this.GetHardwareStatsAsync();
            }

            return this.Ok(GpuMonitor.HardwareStatsHistory.Values.OrderByDescending(s => s.CreatedAt).ToList());
        }


        [HttpGet("models")]
        public ActionResult<List<TorchSharpModel>> GetModels([FromQuery] TorchModelsSortingOption sortingOption = TorchModelsSortingOption.ByLatestModified)
        {
            var models = this.Service.GetModels(null, sortingOption);
            return this.Ok(models);
        }

        [HttpGet("models-simple")]
        public ActionResult<List<string>> GetModelsSimple()
        {
            var models = this.Service.GetModels().Select(m => $"'{m.ModelName}' [{m.BillionParameters}B] ~{Math.Round(m.ModelSizeInMb)} MB, {m.ConfigJsonFiles.Count} config files").ToList();
            return this.Ok(models);
        }


        [SupportedOSPlatform("windows")]
        [HttpPost("load-model")]
        public async Task<ActionResult<TorchSharpModelLoadResponse?>> LoadModelAsync([FromBody] TorchSharpModelLoadRequest loadRequest)
        {
            DateTime requestReceived = DateTime.Now;
            TorchSharpModel? loadedModel = null;
            TorchSharpModelLoadResponse response = new();

            try
            {
                if (this.Settings.HardwareMonitoring)
                {
                    var startStatsResponse = await this.GetHardwareStatsAsync();
                    if (startStatsResponse?.Value != null)
                    {
                        response.HardwareStatsBeforeLoad = startStatsResponse.Value;
                    }
                }

                loadedModel = await this.Service.LoadModelAsync(loadRequest);
                if (loadedModel == null)
                {
                    return this.BadRequest("Failed to load model. Check server logs for details.");
                }
                response.Success = true;
                response.LoadedModel = loadedModel;

                if (response.HardwareStatsBeforeLoad == null && this.Settings.HardwareMonitoring)
                {
                    var endStatsResponse = await this.GetHardwareStatsAsync();
                    if (endStatsResponse?.Value != null)
                    {
                        response.HardwareStatsAfterLoad = endStatsResponse.Value;
                    }
                }

                return this.Ok(response);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error loading model: {ex.Message}");
                return this.StatusCode(500, $"Error loading model: {ex.Message}");
            }
            finally
            {
                DateTime requestHandled = DateTime.Now;
                TimeSpan duration = requestHandled - requestReceived;
                response.LoadingElapsedMs = (int)duration.TotalMilliseconds;

                await StaticLogger.LogAsync($"Model load request handled. Duration: {response.LoadingElapsedMs:N0} ms.");
            }
        }

        [SupportedOSPlatform("windows")]
        [HttpPost("load-model-simple")]
        public async Task<ActionResult<TorchSharpModelLoadResponse?>> LoadModelSimpleAsync([FromQuery] string? modelName = null, int fuzzyMatch = -1, [FromQuery] bool forceCpu = false, [FromQuery] bool strictLoading = false, [FromQuery] string scalarType = "Float16", CancellationToken ct = default)
        {
            modelName ??= string.IsNullOrEmpty(this.Settings.DefaultModelName) ? null : this.Settings.DefaultModelName;
            if (string.IsNullOrEmpty(modelName))
            {
                Console.WriteLine("No model name provided in query and no default model configured.");
                return this.StatusCode(500, "No model name provided in query and no default model configured.");
            }

            var model = this.Service.GetModels().FirstOrDefault(m => m.ModelName.Equals(modelName, StringComparison.OrdinalIgnoreCase));
            if (model == null)
            {
                if (fuzzyMatch > 0)
                {
                    // Can you rephrase below line please to have fuzzyMatch as int for max of chars that may differ from exact match, <0: fuzzyMatch disabled, 0: exact match, >0: max char differences allowed
                    model = this.Service.GetModels().FirstOrDefault(m => GetCharDifference(m.ModelName, modelName) <= fuzzyMatch);
                }
                if (model == null)
                {
                    return this.NotFound($"Model '{modelName}' not found.");
                }
            }

            TorchSharpModelLoadRequest loadRequest = new(model, forceCpu, scalarType, strictLoading);
            HardwareStatistics? hardwareStatsBeforeLoad = null;
            DateTime loadingDateTime = DateTime.Now;

            try
            {
                if (this.Settings.HardwareMonitoring && this.GPUMonitor != null)
                {
                    hardwareStatsBeforeLoad = await this.GPUMonitor.GetCurrentHardwareStatisticsAsync();
                }

                var loadResult = await this.Service.LoadModelAsync(loadRequest);
                if (loadResult == null)
                {
                    return this.BadRequest("Failed to load model. Check server logs for details.");
                }

                TorchSharpModelLoadResponse resp = new()
                {
                    LoadedModel = loadResult,
                    HardwareStatsBeforeLoad = hardwareStatsBeforeLoad,
                    HardwareStatsAfterLoad = this.Settings.HardwareMonitoring && this.GPUMonitor != null ? (await this.GPUMonitor.GetCurrentHardwareStatisticsAsync()) : null,
                    LoadingElapsedMs = (int) (DateTime.Now - loadingDateTime).TotalMilliseconds,
                    Success = loadRequest != null
                };

                return resp;
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"Error loading model: {ex.Message}");
            }
        }

        [HttpGet("progress-load-model")]
        public ActionResult<double> GetModelLoadProgress()
        {
            double progress = this.Service.ModelLoadProgress;
            return this.Ok(progress);
        }



        [HttpDelete("unload-model")]
        public async Task<ActionResult<bool?>> UnloadModelAsync()
        {
            try
            {
                bool? result = await this.Service.UnloadModelAsync();
                if (result == null)
                {
                    return this.NotFound();
                }

                return this.Ok(result);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error unloading model: {ex.Message}");
                return this.StatusCode(500, $"Error unloading model: {ex.Message}");
            }
        }





        [HttpGet("generate")]
        public async Task<IActionResult> GenerateTextAsync([FromQuery] string prompt = "Hi, this is a test, I hope you are doing well! Greetings.", [FromQuery] int maxTokens = 48, [FromQuery] bool logResponse = true, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(prompt))
            {
                return this.BadRequest("Prompt cannot be empty.");
            }

            DateTime taskStarted = DateTime.Now;
            string result = string.Empty;

            try
            {
                // HttpContext.RequestAborted leitet den Abbruch weiter, falls der Client die Anfrage abgibt
                result = await this.Service.GenerateTextAsync(prompt, maxTokens, ct);
                return this.Ok(new { text = result });
            }
            catch (OperationCanceledException)
            {
                // Wird geworfen, wenn der Request abgebrochen wird (oft durch das Framework)
                await StaticLogger.LogAsync("Client disconnected. Generation cancelled.");
                return this.StatusCode(499, "Client Closed Request");
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error during generation: {ex.Message}");
                return this.StatusCode(500, $"Error during generation: {ex.Message}");
            }
            finally
            {
                DateTime taskEnded = DateTime.Now;
                TimeSpan duration = taskEnded - taskStarted;
                await StaticLogger.LogAsync($"Generation finished. Duration: {Math.Round(duration.TotalSeconds, 3)} seconds.");
                if (logResponse)
                {
                    await StaticLogger.LogAsync($"Generated text: {result}");
                }
            }
        }

        [HttpGet("generate-stream")]
        [Produces("text/event-stream")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task GenerateStreamAsync([FromQuery] string prompt = "Hi, this is a test, I hope you are doing well! Greetings.", [FromQuery] int maxTokens = 128, [FromQuery] bool logTokens = true, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(prompt))
            {
                this.Response.StatusCode = 400;
                await this.Response.WriteAsync("Prompt cannot be empty.");
                return;
            }

            // Schaltet das Buffering von Kestrel komplett ab
            var responseFeature = this.Response.HttpContext.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
            responseFeature?.DisableBuffering();

            // WICHTIG für Server-Sent Events (SSE)
            this.Response.Headers.ContentType = "text/event-stream";
            this.Response.Headers.CacheControl = "no-cache";

            DateTime taskStarted = DateTime.Now;
            var responseBuilder = new StringBuilder();

            try
            {
                // HttpContext.RequestAborted als CancellationToken übergeben
                var tokenStream = this.Service.GenerateTextStreamAsync(prompt, maxTokens, logTokens, this.HttpContext.RequestAborted);

                await foreach (var chunk in tokenStream.WithCancellation(ct).ConfigureAwait(false))
                {
                    if (string.IsNullOrEmpty(chunk))
                    {
                        continue;
                    }

                    responseBuilder.Append(chunk);
                    await this.Response.WriteAsync($"data: {chunk}\n\n", ct);
                    await this.Response.Body.FlushAsync(ct);
                }


                // Signalisiert dem Client, dass der Stream erfolgreich beendet ist
                await this.Response.WriteAsync("data: [DONE]\n\n", ct);
                await this.Response.Body.FlushAsync();
            }
            catch (OperationCanceledException)
            {
                await StaticLogger.LogAsync("Client disconnected. Stream cancelled.");
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error during generation stream: {ex.Message}");
                await this.Response.WriteAsync($"event: error\ndata: {ex.Message}\n\n", ct);
                await this.Response.Body.FlushAsync();
            }
            finally
            {
                DateTime taskEnded = DateTime.Now;
                TimeSpan duration = taskEnded - taskStarted;
                await StaticLogger.LogAsync($"Generation stream ended. Duration: {Math.Round(duration.TotalSeconds, 3)} seconds.");
                if (logTokens)
                {
                    await StaticLogger.LogAsync($"Aggregated generated text: {responseBuilder.ToString()}");
                }
            }
        }










        // Private helpers (parsing and logics...)
        private static int GetCharDifference(string? s, string? t)
        {
            s ??= string.Empty;
            t ??= string.Empty;

            int n = s.Length;
            int m = t.Length;
            if (n == 0) return m;
            if (m == 0) return n;

            int[] prev = new int[m + 1];
            int[] curr = new int[m + 1];

            for (int j = 0; j <= m; j++) prev[j] = j;

            for (int i = 1; i <= n; i++)
            {
                curr[0] = i;
                for (int j = 1; j <= m; j++)
                {
                    int cost = s[i - 1] == t[j - 1] ? 0 : 1;
                    curr[j] = Math.Min(Math.Min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
                }

                // swap prev and curr
                var tmp = prev;
                prev = curr;
                curr = tmp;
            }

            return prev[m];
        }
    }
}
