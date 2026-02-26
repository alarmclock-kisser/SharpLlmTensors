using SharpLlmTensors.Api.Filters;
using SharpLlmTensors.Monitoring;
using SharpLlmTensors.Runtime;
using SharpLlmTensors.Shared;
using System.Diagnostics;

namespace SharpLlmTensors.Api
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Get appsettings.json
            string logDir = builder.Configuration.GetValue<string>("LogDirectory", Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory));
            if (string.IsNullOrEmpty(logDir))
            {
                logDir = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
            }
            Console.WriteLine($"Log file(s) directory path: {logDir}");
            bool createLogFile = builder.Configuration.GetValue<bool>("CreateLogFile");
            Console.WriteLine(createLogFile ? "Log file will be created." : "NO log file will be created.");
            int maxPreviousLogFiles = builder.Configuration.GetValue<int>("MaxPreviousLogFiles", -1);
            Console.WriteLine(maxPreviousLogFiles < 0 ? "All previous log files will be kept." : maxPreviousLogFiles == 0 ? "No previous log files will be kept!" : $"Up to {maxPreviousLogFiles} previous log files will be kept.");
            var appSettings = builder.Configuration.GetSection("AppSettings").Get<AppSettings>() ?? new AppSettings();

            // Configure StaticLogger
            StaticLogger.InitializeLogFiles(logDir, createLogFile, maxPreviousLogFiles);
            Console.WriteLine("StaticLogger initialized.");

            // CORS policy
            const string CorsPolicy = "AllowApi";
            builder.Services.AddCors(options =>
            {
                options.AddPolicy(CorsPolicy, policy =>
                {
                    policy
                        .AllowAnyHeader()
                        .AllowAnyMethod()
                        .AllowCredentials()
                        .SetIsOriginAllowed(_ => true);
                });
            });

            // Add services to the container.
            builder.Services.AddSingleton(appSettings);
            builder.Services.AddSingleton<TorchService>();
            if (appSettings.HardwareMonitoring)
            {
                builder.Services.AddSingleton<GpuMonitor>();
            }

            builder.Services.AddControllers();
            // Register Swagger generator (Swashbuckle) so ISwaggerProvider is available
            builder.Services.AddSwaggerGen(c =>
            {
                c.OperationFilter<ElapsedTimeOperationFilter>();
            });

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHsts();
            app.UseHttpsRedirection();

            app.UseCors(CorsPolicy);

            // Show elapsed ms
            app.Use(async (ctx, next) =>
            {
                var sw = Stopwatch.StartNew();
                ctx.Response.OnStarting(() =>
                {
                    sw.Stop();
                    ctx.Response.Headers["X-Elapsed-Ms"] = sw.ElapsedMilliseconds.ToString();
                    return Task.CompletedTask;
                });
                await next();
            });

            app.UseAuthorization();
            app.MapControllers();

            app.Run();
        }
    }
}
