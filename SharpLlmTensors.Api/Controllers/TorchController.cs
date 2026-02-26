using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using SharpLlmTensors.Runtime;
using SharpLlmTensors.Shared;
using System.Runtime.CompilerServices;
using System.Text;

namespace SharpLlmTensors.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class TorchController : ControllerBase
    {
        private readonly AppSettings Settings;
        private readonly TorchService Service;

        public TorchController(AppSettings appSettings, TorchService service)
        {
            this.Settings = appSettings;
            this.Service = service;
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

        [HttpGet("models")]
        public ActionResult<List<TorchSharpModel>> GetModels([FromQuery] TorchModelsSortingOption sortingOption = TorchModelsSortingOption.Alphabetical)
        {
            var models = this.Service.GetModels(null, sortingOption);
            return this.Ok(models);
        }

        [HttpGet("models-simple")]
        public ActionResult<List<string>> GetModelsSimple()
        {
            var models = this.Service.GetModels().Select(m => $"'{m.ModelName}' [{m.BillionParameters}B] ~{Math.Round(m.ModelSizeInMb)} MB, {m.ConfigJsonFiles.FilesCount} config files").ToList();
            return this.Ok(models);
        }



        [HttpPost("load-model")]
        public async Task<ActionResult<TorchSharpModel?>> LoadModelAsync([FromBody] TorchSharpModelLoadRequest loadRequest)
        {
            DateTime requestReceived = DateTime.Now;

            try
            {
                var loadedModel = await this.Service.LoadModelAsync(loadRequest);
                if (loadedModel == null)
                {
                    return this.BadRequest("Failed to load model. Check server logs for details.");
                }

                return this.Ok(loadedModel);
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
                await StaticLogger.LogAsync($"Model load request handled. Duration: {Math.Round(duration.TotalSeconds, 3)} seconds.");
            }
        }

        [HttpPost("load-model-simple")]
        public async Task<ActionResult<TorchSharpModel?>> LoadModelSimpleAsync([FromQuery] string modelName = "Qwen2.5-VL-3B", [FromQuery] bool fuzzyMatch = false, [FromQuery] bool forceCpu = false)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                return this.BadRequest("Model name must be provided in query parameters when loadRequest body is not provided.");
            }

            TorchSharpModel? model = this.Service.GetModels().FirstOrDefault(m => m.ModelName.Equals(modelName, StringComparison.OrdinalIgnoreCase));
            if (model == null)
            {
                if (fuzzyMatch)
                {
                    model = this.Service.GetModels().FirstOrDefault(m => m.ModelName.Contains(modelName, StringComparison.OrdinalIgnoreCase));
                }

                if (model == null)
                {
                    return this.NotFound($"Model '{modelName}' not found.");
                }
            }

            var loadRequest = new TorchSharpModelLoadRequest(model, forceCpu);

            DateTime taskStarted = DateTime.Now;

            try
            {
                var loadedModel = await this.Service.LoadModelAsync(loadRequest);
                if (loadedModel == null)
                {
                    return this.BadRequest("Failed to load model. Check server logs for details.");
                }

                return this.Ok(loadedModel);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex);
                return this.BadRequest("Error loading model: " + ex.Message);
            }
            finally
            {
                DateTime taskEnded = DateTime.Now;
                TimeSpan duration = taskEnded - taskStarted;
                await StaticLogger.LogAsync($"Model load (simple) request handled. Duration: {Math.Round(duration.TotalSeconds, 3)} seconds.");
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

    }
}
