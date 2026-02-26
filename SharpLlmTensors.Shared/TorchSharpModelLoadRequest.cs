using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLlmTensors.Shared
{
    public class TorchSharpModelLoadRequest
    {
        public TorchSharpModel Model { get; set; }

        public bool ForceCpu { get; set; } = false;

        public string ScalarT { get; set; } = "Float16";

        public bool StrictLoadingMode { get; set; } = false;


        public TorchSharpModelLoadRequest(TorchSharpModel model, bool forceCpu = false, string scalarT = "Float16", bool strictLoading = false)
        {
            this.Model = model ?? throw new ArgumentNullException(nameof(model));
            this.ForceCpu = forceCpu;
            this.ScalarT = scalarT;
            this.StrictLoadingMode = strictLoading;
        }
    }
}
