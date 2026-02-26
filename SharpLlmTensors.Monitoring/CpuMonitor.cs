using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using SharpLlmTensors.Shared;
using System.Management;

namespace SharpLlmTensors.Monitoring
{
    [SupportedOSPlatform("windows")]
    public static class CpuMonitor
    {
        private static readonly PerformanceCounter[] _cpuCounters = CreateCpuCounters();
        private static readonly TimeSpan _samplingInterval = TimeSpan.FromMilliseconds(250);
        private static DateTime _lastSampleUtc = DateTime.MinValue;
        private static double[] _lastUsages = [];
        private static readonly Lock _sampleLock = new();

        private static PerformanceCounter[] CreateCpuCounters()
        {
            int coreCount = Environment.ProcessorCount;
            var counters = new PerformanceCounter[coreCount];

            for (int i = 0; i < coreCount; i++)
            {
                counters[i] = new PerformanceCounter("Processor", "% Processor Time", i.ToString(), true);
                // erste Probe, damit der nächste Wert „richtig“ ist
                _ = counters[i].NextValue();
            }

            _lastUsages = new double[coreCount];
            return counters;
        }

        /// <summary>
        /// CPU-Auslastung pro logischem Prozessor (0.0f - 1.0f).
        /// Nicht-blockierend: liefert gecachte Werte, wenn Intervall noch nicht abgelaufen.
        /// </summary>
        public static Task<double[]> GetThreadUsagesAsync(CancellationToken cancellationToken = default)
        {
            lock (_sampleLock)
            {
                var now = DateTime.UtcNow;
                var elapsed = now - _lastSampleUtc;

                if (elapsed > _samplingInterval * 4)
                {
                    for (int i = 0; i < _cpuCounters.Length; i++)
                    {
                        _ = _cpuCounters[i].NextValue();
                    }
                    Thread.Sleep(_samplingInterval);
                }
                else if (elapsed < _samplingInterval && _lastUsages.Length == _cpuCounters.Length)
                {
                    return Task.FromResult((double[]) _lastUsages.Clone());
                }

                int coreCount = _cpuCounters.Length;
                var usages = new double[coreCount];

                for (int i = 0; i < coreCount; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    float percent = _cpuCounters[i].NextValue();
                    if (percent < 0f)
                    {
                        percent = 0f;
                    }

                    if (percent > 100f)
                    {
                        percent = 100f;
                    }

                    usages[i] = percent / 100f;
                }

                _lastUsages = usages;
                _lastSampleUtc = now;
                return Task.FromResult((double[]) usages.Clone());
            }
        }

        /// <summary>
        /// Sync-Wrapper, falls du irgendwo keine async-Methode aufrufen willst.
        /// </summary>
        public static double[] GetThreadUsages()
            => GetThreadUsagesAsync().GetAwaiter().GetResult();

        /// <summary>
        /// Malt die CPU-Auslastung pro Kern als Bitmap. Async, da das Rendern bei vielen Kernen etwas dauern kann.
        /// </summary>
        public static Task<Bitmap> RenderCoresBitmapAsync(float[] usages, int width, int height, Color? backColor = null, Color? renderPercentagesColor = null, CancellationToken ct = default)
        {
            backColor ??= Color.White;
            return Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();

                int count = Math.Max(1, usages?.Length ?? 1);

                // Compute grid: try to make it as square as possible
                int cols = (int) Math.Ceiling(Math.Sqrt(count));
                int rows = (int) Math.Ceiling(count / (double) cols);

                var bmp = new Bitmap(Math.Max(1, width), Math.Max(1, height));
                using (var g = Graphics.FromImage(bmp))
                {
                    g.Clear(backColor.Value);
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;

                    // Padding and cell sizes
                    int pad = 2;
                    int gridW = width - pad * (cols + 1);
                    int gridH = height - pad * (rows + 1);
                    if (gridW < cols)
                    {
                        gridW = cols;
                    }

                    if (gridH < rows)
                    {
                        gridH = rows;
                    }

                    int cellW = gridW / cols;
                    int cellH = gridH / rows;

                    using var borderPen = new Pen(Color.Black, 1f);
                    using var fillBrush = new SolidBrush(Color.FromArgb(64, 160, 255));
                    using var highBrush = new SolidBrush(Color.FromArgb(255, 96, 96));

                    for (int i = 0; i < count; i++)
                    {
                        ct.ThrowIfCancellationRequested();
                        int r = i / cols;
                        int c = i % cols;
                        int x = pad + c * (cellW + pad);
                        int y = pad + r * (cellH + pad);

                        // Outer rect
                        var rect = new Rectangle(x, y, cellW, cellH);
                        g.DrawRectangle(borderPen, rect);

                        // Fill proportionally from bottom based on usage
                        float u = usages?[i] ?? 0;
                        if (u < 0f)
                        {
                            u = 0f;
                        }

                        if (u > 1f)
                        {
                            u = 1f;
                        }

                        int filledH = (int) Math.Round(u * cellH);
                        if (filledH > 0)
                        {
                            var fillRect = new Rectangle(x + 1, y + cellH - filledH + 1, Math.Max(1, cellW - 2), Math.Max(1, filledH - 2));
                            // use red above 80%
                            var brush = u >= 0.8f ? highBrush : fillBrush;
                            g.FillRectangle(brush, fillRect);
                        }

                        // Optionally render the percentage text centered in the cell.
                        if (renderPercentagesColor.HasValue)
                        {
                            using var textBrush = new SolidBrush(renderPercentagesColor.Value);
                            // Percentage text (rounded integer percent)
                            string percentText = $"{Math.Round(u * 100f)}%";

                            // Determine dynamic font size so the text fits inside the cell.
                            // Start with a reasonable maximum relative to cell size and decrease until it fits or reaches a minimum size.
                            float maxFont = Math.Min(cellW, cellH) * 0.45f;
                            float fontSize = Math.Max(6f, maxFont);

                            using var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };

                            // Measure and adjust font size. Use GraphicsUnit.Pixel for consistent measurements in pixels.
                            for (;;)
                            {
                                using var testFont = new Font(SystemFonts.DefaultFont.FontFamily, fontSize, FontStyle.Bold, GraphicsUnit.Pixel);
                                var size = g.MeasureString(percentText, testFont);
                                // Add a small padding
                                if ((size.Width > cellW - 4 || size.Height > cellH - 4) && fontSize > 6f)
                                {
                                    fontSize -= 1f;
                                    continue;
                                }
                                break;
                            }

                            using var textFont = new Font(SystemFonts.DefaultFont.FontFamily, fontSize, FontStyle.Bold, GraphicsUnit.Pixel);
                            g.DrawString(percentText, textFont, textBrush, rect, sf);
                        }
                    }
                }

                return bmp;
            }, ct);
        }

        // -------- Speicher (physisch) --------
        // Die Speicherabfragen sind sehr schnell und blockieren nicht nennenswert.
        // Async bringt hier praktisch nichts, daher bleiben sie synchron.

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
        private struct MEMORYSTATUSEX
        {
            public uint dwLength;
            public uint dwMemoryLoad;
            public ulong ullTotalPhys;
            public ulong ullAvailPhys;
            public ulong ullTotalPageFile;
            public ulong ullAvailPageFile;
            public ulong ullTotalVirtual;
            public ulong ullAvailVirtual;
            public ulong ullAvailExtendedVirtual;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX lpBuffer);

        private static MEMORYSTATUSEX GetMemoryStatus()
        {
            var status = new MEMORYSTATUSEX
            {
                dwLength = (uint) Marshal.SizeOf<MEMORYSTATUSEX>()
            };

            if (!GlobalMemoryStatusEx(ref status))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }

            return status;
        }

        /// <summary>
        /// Gesamter physischer Speicher in BYTES.
        /// </summary>
        public static long GetTotalMemoryBytes()
        {
            var status = GetMemoryStatus();
            return (long) status.ullTotalPhys;
        }

        /// <summary>
        /// Verwendeter physischer Speicher in BYTES.
        /// </summary>
        public static long GetUsedMemoryBytes()
        {
            var status = GetMemoryStatus();
            ulong used = status.ullTotalPhys - status.ullAvailPhys;
            return (long) used;
        }

        public static string GetCpuName()
        {
            try
            {
                return new ManagementObjectSearcher("select Name from Win32_Processor")
                    .Get()
                    .Cast<ManagementObject>()
                    .Select(mo => mo["Name"]?.ToString()?.Trim())
                    .FirstOrDefault(name => !string.IsNullOrEmpty(name)) ?? "N/A";
            }
            catch (Exception ex)
            {
                StaticLogger.Log(ex);
            }

            return "Unknown CPU";
        }
    }

}

