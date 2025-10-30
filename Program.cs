using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.Services.AddHttpClient("ollama", static c =>
{
    c.BaseAddress = new Uri("http://localhost:11434");
    c.Timeout = TimeSpan.FromSeconds(90);
});
builder.Services.ConfigureHttpJsonOptions(static o =>
{
    o.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
    o.SerializerOptions.PropertyNameCaseInsensitive = true;
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

// --------------------------------- Endpoint ---------------------------------
app.MapPost("/hashtags", static async (
    HashtagRequest req,
    IHttpClientFactory httpFactory,
    CancellationToken ct) =>
{
    // Valida��o b�sica
    if (string.IsNullOrWhiteSpace(req.Text))
        return Results.BadRequest(new { error = "O campo 'text' � obrigat�rio." });

    var model = string.IsNullOrWhiteSpace(req.Model) ? "llama3.2:3b" : req.Model!.Trim();
    var n = req.Count ?? 10;
    if (n < 1) n = 10;
    if (n > 30) n = 30;

    // Prompt (raw string) � sem escapes confusos
    var prompt = $$"""
Tarefa: Gere exatamente {{n}} hashtags relevantes para o texto abaixo.
Regras obrigat�rias:
- Retorne SOMENTE JSON que siga o schema fornecido (sem coment�rios).
- Exatamente {{n}} itens.
- Cada hashtag deve come�ar com '#', sem espa�os, sem emojis, apenas letras, n�meros ou '_' (underline).
- N�o repita hashtags (case-insensitive).
- Use a MESMA l�ngua do texto.
Texto:
{{req.Text.Trim()}}
""";

    // JSON Schema (structured outputs)
    var schema = new
    {
        type = "object",
        additionalProperties = false,
        properties = new
        {
            hashtags = new
            {
                type = "array",
                minItems = n,
                maxItems = n,
                uniqueItems = true,
                items = new
                {
                    type = "string",
                    pattern = @"^#[\p{L}\p{N}_]+$"
                }
            }
        },
        required = new[] { "hashtags" }
    };

    var client = httpFactory.CreateClient("ollama");

    var body = new
    {
        model,
        prompt,
        format = schema,
        stream = false,
        options = new
        {
            temperature = 0,
            num_ctx = 1024,    // contexto menor (economiza RAM/VRAM)
            num_predict = 64,  // limita tokens de sa�da
            num_gpu = 0        // for�a CPU (evita DirectML/VRAM)
        }
    };

    using var httpResp = await client.PostAsJsonAsync("/api/generate", body, ct);
    if (!httpResp.IsSuccessStatusCode)
    {
        var err = await httpResp.Content.ReadAsStringAsync(ct);
        return Results.Json(
            new { error = $"Falha ao gerar com o modelo '{model}'. Detalhes: {err}" },
            statusCode: StatusCodes.Status502BadGateway
        );
    }

    var gen = await httpResp.Content.ReadFromJsonAsync<OllamaGenerateResponse>(cancellationToken: ct);
    if (gen is null || string.IsNullOrWhiteSpace(gen.Response))
    {
        return Results.Json(
            new { error = "Resposta vazia do modelo." },
            statusCode: StatusCodes.Status502BadGateway
        );
    }
    // Parse do JSON em 'response'
    JsonArray? hashtagsJson;
    try
    {
        var node = JsonNode.Parse(gen.Response)?.AsObject();
        hashtagsJson = node?["hashtags"]?.AsArray();
    }
    catch
    {
        return Results.Json(
            new { error = "N�o foi poss�vel interpretar a sa�da JSON do modelo." },
            statusCode: StatusCodes.Status502BadGateway
        );
    }

    // Sanitiza��o e valida��es finais
    var rx = new Regex(@"^#[\p{L}\p{N}_]+$", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    var list = new List<string>();

    if (hashtagsJson is not null)
    {
        foreach (var item in hashtagsJson)
        {
            var s = item?.GetValue<string>()?.Trim() ?? "";
            if (string.IsNullOrEmpty(s)) continue;
            if (!s.StartsWith('#')) s = "#" + s;
            s = Regex.Replace(s, @"\s+", "_"); // defesa extra
            if (rx.IsMatch(s) && set.Add(s)) list.Add(s);
            if (list.Count == n) break;
        }
    }

    if (list.Count != n)
    {
        return Results.Json(
            new { error = $"O modelo n�o retornou exatamente {n} hashtags v�lidas ap�s valida��o.", got = list },
            statusCode: StatusCodes.Status502BadGateway
        );
    }

    return Results.Ok(new HashtagResponse(model, n, list));
})
.WithName("PostHashtags")
.Produces<HashtagResponse>(StatusCodes.Status200OK)
.Produces(StatusCodes.Status400BadRequest)
.Produces(StatusCodes.Status502BadGateway);

app.Run();

// ------------------------------- Tipos (no fim) ------------------------------
public record HashtagRequest(string Text, int? Count, string? Model);
public record HashtagResponse(string Model, int Count, List<string> Hashtags);
public sealed class OllamaGenerateResponse
{
    public string? Model { get; set; }
    public string? Response { get; set; } // JSON em string vindo do Ollama
    public bool Done { get; set; }
}
