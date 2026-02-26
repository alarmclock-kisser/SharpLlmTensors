using Microsoft.OpenApi;
using System.Collections.Generic;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace SharpLlmTensors.Api.Filters
{
    public class ElapsedTimeOperationFilter : IOperationFilter
    {
        public void Apply(OpenApiOperation operation, OperationFilterContext context)
        {
            if (operation.Responses == null)
            {
                return;
            }

            foreach (var resp in operation.Responses.Values)
            {
                // prepare header instance
                var elapsedHeader = new OpenApiHeader
                {
                    Description = "Elapsed time in ms",
                    Schema = new OpenApiSchema { Type = JsonSchemaType.String }
                };

                // If we have the concrete OpenApiResponse type we can set the Headers property
                if (resp is OpenApiResponse concrete)
                {
                    if (concrete.Headers == null)
                    {
                        concrete.Headers = new Dictionary<string, IOpenApiHeader>();
                    }

                    concrete.Headers["X-Elapsed-Ms"] = elapsedHeader;
                }
                else
                {
                    // Fallback: try to modify the headers dictionary exposed via the interface
                    var headers = resp.Headers as IDictionary<string, IOpenApiHeader>;
                    if (headers != null)
                    {
                        headers["X-Elapsed-Ms"] = elapsedHeader;
                    }
                }
            }
        }


    }
}
