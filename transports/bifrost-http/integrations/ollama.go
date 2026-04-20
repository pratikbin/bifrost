package integrations

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/bytedance/sonic"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// OllamaRouter handles Ollama-compatible API endpoints.
type OllamaRouter struct {
	*GenericRouter
}

// OllamaStop unmarshals from either a JSON string or an array of strings.
type OllamaStop []string

func (s *OllamaStop) UnmarshalJSON(data []byte) error {
	var str string
	if err := sonic.Unmarshal(data, &str); err == nil {
		*s = OllamaStop{str}
		return nil
	}
	var arr []string
	if err := sonic.Unmarshal(data, &arr); err == nil {
		*s = OllamaStop(arr)
		return nil
	}
	return errors.New("stop must be a string or array of strings")
}

// OllamaOptions holds Ollama model generation options.
type OllamaOptions struct {
	Temperature      *float64   `json:"temperature,omitempty"`
	TopP             *float64   `json:"top_p,omitempty"`
	TopK             *int       `json:"top_k,omitempty"`
	MinP             *float64   `json:"min_p,omitempty"`
	Seed             *int       `json:"seed,omitempty"`
	NumPredict       *int       `json:"num_predict,omitempty"`
	NumCtx           *int       `json:"num_ctx,omitempty"`
	Stop             OllamaStop `json:"stop,omitempty"`
	FrequencyPenalty *float64   `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64   `json:"presence_penalty,omitempty"`
}

// OllamaTool is a tool definition in Ollama format.
type OllamaTool struct {
	Type     string      `json:"type"`
	Function OllamaToolFn `json:"function"`
}

// OllamaToolFn is the function definition within a tool.
type OllamaToolFn struct {
	Name        string                          `json:"name"`
	Description string                          `json:"description,omitempty"`
	Parameters  *schemas.ToolFunctionParameters `json:"parameters,omitempty"`
}

// OllamaToolCall is a tool call in a message or response.
type OllamaToolCall struct {
	Function OllamaToolCallFn `json:"function"`
}

// OllamaToolCallFn holds the function name and arguments (as a JSON object, not stringified).
type OllamaToolCallFn struct {
	Name      string      `json:"name"`
	Arguments interface{} `json:"arguments"`
}

// OllamaChatMessage is a single message in Ollama chat format.
type OllamaChatMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	Thinking  string           `json:"thinking,omitempty"`
}

// OllamaGenerateRequest is the request type for Ollama's /api/generate endpoint.
type OllamaGenerateRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Suffix  string         `json:"suffix,omitempty"`
	Images  []string       `json:"images,omitempty"`
	System  string         `json:"system,omitempty"`
	Format  interface{}    `json:"format,omitempty"` // "json" or JSON schema object
	Think   interface{}    `json:"think,omitempty"`  // bool or "high"/"medium"/"low"
	Stream  *bool          `json:"stream,omitempty"`
	Options *OllamaOptions `json:"options,omitempty"`
}

func (r *OllamaGenerateRequest) IsStreamingRequested() bool {
	return r.Stream == nil || *r.Stream
}

// OllamaChatRequest is the request type for Ollama's /api/chat endpoint.
type OllamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []OllamaChatMessage `json:"messages"`
	Tools    []OllamaTool        `json:"tools,omitempty"`
	Format   interface{}         `json:"format,omitempty"` // "json" or JSON schema object
	Think    interface{}         `json:"think,omitempty"`  // bool or "high"/"medium"/"low"
	Stream   *bool               `json:"stream,omitempty"`
	Options  *OllamaOptions      `json:"options,omitempty"`
}

func (r *OllamaChatRequest) IsStreamingRequested() bool {
	return r.Stream == nil || *r.Stream
}

// OllamaEmbedRequest is the request type for Ollama's /api/embed endpoint.
type OllamaEmbedRequest struct {
	Model      string         `json:"model"`
	Input      interface{}    `json:"input"` // string or []string
	Truncate   *bool          `json:"truncate,omitempty"`
	Dimensions *int           `json:"dimensions,omitempty"`
	Options    *OllamaOptions `json:"options,omitempty"`
}

// OllamaGenerateResponse is the response for /api/generate.
type OllamaGenerateResponse struct {
	Model           string `json:"model"`
	CreatedAt       string `json:"created_at"`
	Response        string `json:"response"`
	Thinking        string `json:"thinking,omitempty"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
}

// OllamaChatResponse is the response for /api/chat.
type OllamaChatResponse struct {
	Model           string            `json:"model"`
	CreatedAt       string            `json:"created_at"`
	Message         OllamaChatMessage `json:"message"`
	Done            bool              `json:"done"`
	DoneReason      string            `json:"done_reason,omitempty"`
	PromptEvalCount int               `json:"prompt_eval_count,omitempty"`
	EvalCount       int               `json:"eval_count,omitempty"`
}

// OllamaEmbedResponse is the response for /api/embed.
type OllamaEmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}

// OllamaErrorResponse is the error response for Ollama.
type OllamaErrorResponse struct {
	Error string `json:"error"`
}

func nowISO() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

// ollamaThinkToReasoning maps Ollama's think field to Bifrost ChatReasoning.
func ollamaThinkToReasoning(think interface{}) *schemas.ChatReasoning {
	if think == nil {
		return nil
	}
	switch v := think.(type) {
	case bool:
		return &schemas.ChatReasoning{Enabled: &v}
	case string:
		return &schemas.ChatReasoning{Effort: &v}
	}
	return nil
}

// ollamaFormatToResponseFormat maps Ollama's format field to Bifrost ResponseFormat.
// "json" → {"type":"json_object"}, anything else is passed through.
func ollamaFormatToResponseFormat(format interface{}) *interface{} {
	if format == nil {
		return nil
	}
	var rf interface{}
	if s, ok := format.(string); ok && s == "json" {
		rf = map[string]interface{}{"type": "json_object"}
	} else {
		rf = format
	}
	return &rf
}

// ollamaToolsToBifrost converts Ollama tool definitions to Bifrost ChatTool slice.
func ollamaToolsToBifrost(tools []OllamaTool) []schemas.ChatTool {
	result := make([]schemas.ChatTool, 0, len(tools))
	for _, t := range tools {
		fn := &schemas.ChatToolFunction{
			Name:       t.Function.Name,
			Parameters: t.Function.Parameters,
		}
		if t.Function.Description != "" {
			desc := t.Function.Description
			fn.Description = &desc
		}
		result = append(result, schemas.ChatTool{
			Type:     schemas.ChatToolTypeFunction,
			Function: fn,
		})
	}
	return result
}

// ollamaChatMessageToBifrost converts an OllamaChatMessage to a schemas.ChatMessage.
func ollamaChatMessageToBifrost(m OllamaChatMessage) schemas.ChatMessage {
	msg := schemas.ChatMessage{
		Role: schemas.ChatMessageRole(m.Role),
	}

	// Build content blocks when images are present; plain string otherwise.
	if len(m.Images) > 0 {
		blocks := make([]schemas.ChatContentBlock, 0, 1+len(m.Images))
		if m.Content != "" {
			blocks = append(blocks, schemas.ChatContentBlock{
				Type: schemas.ChatContentBlockTypeText,
				Text: &m.Content,
			})
		}
		for _, img := range m.Images {
			url := "data:image/jpeg;base64," + img
			blocks = append(blocks, schemas.ChatContentBlock{
				Type:           schemas.ChatContentBlockTypeImage,
				ImageURLStruct: &schemas.ChatInputImage{URL: url},
			})
		}
		msg.Content = &schemas.ChatMessageContent{ContentBlocks: blocks}
	} else if m.Content != "" {
		msg.Content = &schemas.ChatMessageContent{ContentStr: &m.Content}
	}

	// Tool calls on assistant messages.
	if len(m.ToolCalls) > 0 {
		calls := make([]schemas.ChatAssistantMessageToolCall, 0, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			argsStr := ""
			if tc.Function.Arguments != nil {
				if b, err := sonic.Marshal(tc.Function.Arguments); err == nil {
					argsStr = string(b)
				}
			}
			idx := uint16(i)
			name := tc.Function.Name
			fnType := "function"
			calls = append(calls, schemas.ChatAssistantMessageToolCall{
				Index: idx,
				Type:  &fnType,
				Function: schemas.ChatAssistantMessageToolCallFunction{
					Name:      &name,
					Arguments: argsStr,
				},
			})
		}
		msg.ChatAssistantMessage = &schemas.ChatAssistantMessage{ToolCalls: calls}
	}

	return msg
}

func ollamaOptionsToExtraParams(opts *OllamaOptions) map[string]interface{} {
	extra := make(map[string]interface{})
	if opts == nil {
		return extra
	}
	if opts.MinP != nil {
		extra["min_p"] = *opts.MinP
	}
	if opts.NumCtx != nil {
		extra["num_ctx"] = *opts.NumCtx
	}
	return extra
}

func ollamaOptionsToChatParams(opts *OllamaOptions, format, think interface{}, tools []OllamaTool) *schemas.ChatParameters {
	params := &schemas.ChatParameters{}
	hasFields := false

	if opts != nil {
		hasFields = true
		params.Temperature = opts.Temperature
		params.TopP = opts.TopP
		params.TopK = opts.TopK
		params.Seed = opts.Seed
		params.MaxCompletionTokens = opts.NumPredict
		params.FrequencyPenalty = opts.FrequencyPenalty
		params.PresencePenalty = opts.PresencePenalty
		if len(opts.Stop) > 0 {
			params.Stop = []string(opts.Stop)
		}
		if extra := ollamaOptionsToExtraParams(opts); len(extra) > 0 {
			params.ExtraParams = extra
		}
	}

	if rf := ollamaFormatToResponseFormat(format); rf != nil {
		params.ResponseFormat = rf
		hasFields = true
	}

	if r := ollamaThinkToReasoning(think); r != nil {
		params.Reasoning = r
		hasFields = true
	}

	if len(tools) > 0 {
		params.Tools = ollamaToolsToBifrost(tools)
		hasFields = true
	}

	if !hasFields {
		return nil
	}
	return params
}

func ollamaOptionsToTextParams(opts *OllamaOptions, suffix string) *schemas.TextCompletionParameters {
	if opts == nil && suffix == "" {
		return nil
	}
	params := &schemas.TextCompletionParameters{}
	if opts != nil {
		params.Temperature = opts.Temperature
		params.TopP = opts.TopP
		params.Seed = opts.Seed
		params.MaxTokens = opts.NumPredict
		params.FrequencyPenalty = opts.FrequencyPenalty
		params.PresencePenalty = opts.PresencePenalty
		if len(opts.Stop) > 0 {
			params.Stop = []string(opts.Stop)
		}
		extra := ollamaOptionsToExtraParams(opts)
		if opts.TopK != nil {
			extra["top_k"] = *opts.TopK
		}
		if len(extra) > 0 {
			params.ExtraParams = extra
		}
	}
	if suffix != "" {
		params.Suffix = &suffix
	}
	return params
}

func toOllamaErrorResponse(err *schemas.BifrostError) *OllamaErrorResponse {
	if err.Error != nil {
		return &OllamaErrorResponse{Error: err.Error.Message}
	}
	return &OllamaErrorResponse{Error: "an unknown error occurred"}
}

func extractTextFromTextCompletionChoice(c schemas.BifrostResponseChoice) string {
	if c.TextCompletionResponseChoice != nil && c.TextCompletionResponseChoice.Text != nil {
		return *c.TextCompletionResponseChoice.Text
	}
	if c.ChatNonStreamResponseChoice != nil && c.ChatNonStreamResponseChoice.Message != nil {
		msg := c.ChatNonStreamResponseChoice.Message
		if msg.Content != nil && msg.Content.ContentStr != nil {
			return *msg.Content.ContentStr
		}
	}
	return ""
}

func extractThinkingFromNonStreamMessage(msg *schemas.ChatMessage) string {
	if msg == nil || msg.ChatAssistantMessage == nil {
		return ""
	}
	if msg.ChatAssistantMessage.Reasoning != nil {
		return *msg.ChatAssistantMessage.Reasoning
	}
	return ""
}

func extractTextFromChatStreamChoice(c schemas.BifrostResponseChoice) string {
	if c.ChatStreamResponseChoice != nil && c.ChatStreamResponseChoice.Delta != nil {
		if c.ChatStreamResponseChoice.Delta.Content != nil {
			return *c.ChatStreamResponseChoice.Delta.Content
		}
	}
	return ""
}

func extractThinkingFromChatStreamChoice(c schemas.BifrostResponseChoice) string {
	if c.ChatStreamResponseChoice != nil && c.ChatStreamResponseChoice.Delta != nil {
		if c.ChatStreamResponseChoice.Delta.Reasoning != nil {
			return *c.ChatStreamResponseChoice.Delta.Reasoning
		}
	}
	return ""
}

// extractToolCallsFromNonStreamMessage converts Bifrost tool calls to Ollama format.
// Bifrost stores arguments as a stringified JSON string; Ollama expects a JSON object.
func extractToolCallsFromNonStreamMessage(msg *schemas.ChatMessage) []OllamaToolCall {
	if msg == nil || msg.ChatAssistantMessage == nil || len(msg.ChatAssistantMessage.ToolCalls) == 0 {
		return nil
	}
	result := make([]OllamaToolCall, 0, len(msg.ChatAssistantMessage.ToolCalls))
	for _, tc := range msg.ChatAssistantMessage.ToolCalls {
		name := ""
		if tc.Function.Name != nil {
			name = *tc.Function.Name
		}
		var args interface{}
		if tc.Function.Arguments != "" {
			if err := sonic.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				args = tc.Function.Arguments
			}
		}
		result = append(result, OllamaToolCall{
			Function: OllamaToolCallFn{Name: name, Arguments: args},
		})
	}
	return result
}

func toOllamaGenerateResponse(resp *schemas.BifrostTextCompletionResponse) *OllamaGenerateResponse {
	text := ""
	thinking := ""
	doneReason := ""
	promptEval, evalCount := 0, 0

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		text = extractTextFromTextCompletionChoice(c)
		if c.ChatNonStreamResponseChoice != nil {
			thinking = extractThinkingFromNonStreamMessage(c.ChatNonStreamResponseChoice.Message)
		}
		if c.FinishReason != nil {
			doneReason = *c.FinishReason
		}
	}
	if resp.Usage != nil {
		promptEval = resp.Usage.PromptTokens
		evalCount = resp.Usage.CompletionTokens
	}
	return &OllamaGenerateResponse{
		Model:           resp.Model,
		CreatedAt:       nowISO(),
		Response:        text,
		Thinking:        thinking,
		Done:            true,
		DoneReason:      doneReason,
		PromptEvalCount: promptEval,
		EvalCount:       evalCount,
	}
}

func toOllamaChatResponse(resp *schemas.BifrostChatResponse) *OllamaChatResponse {
	msg := OllamaChatMessage{Role: "assistant"}
	doneReason := ""
	promptEval, evalCount := 0, 0

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		if c.ChatNonStreamResponseChoice != nil && c.ChatNonStreamResponseChoice.Message != nil {
			m := c.ChatNonStreamResponseChoice.Message
			if m.Content != nil && m.Content.ContentStr != nil {
				msg.Content = *m.Content.ContentStr
			}
			msg.Thinking = extractThinkingFromNonStreamMessage(m)
			msg.ToolCalls = extractToolCallsFromNonStreamMessage(m)
		}
		if c.FinishReason != nil {
			doneReason = *c.FinishReason
		}
	}
	if resp.Usage != nil {
		promptEval = resp.Usage.PromptTokens
		evalCount = resp.Usage.CompletionTokens
	}
	return &OllamaChatResponse{
		Model:           resp.Model,
		CreatedAt:       nowISO(),
		Message:         msg,
		Done:            true,
		DoneReason:      doneReason,
		PromptEvalCount: promptEval,
		EvalCount:       evalCount,
	}
}

// ndjsonLine marshals v and appends a newline — the NDJSON streaming wire format.
func ndjsonLine(v interface{}) (string, error) {
	b, err := sonic.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b) + "\n", nil
}

func toOllamaGenerateStreamChunk(resp *schemas.BifrostTextCompletionResponse) (string, interface{}, error) {
	text := ""
	thinking := ""
	doneReason := ""
	done := false

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		text = extractTextFromTextCompletionChoice(c)
		if c.ChatStreamResponseChoice != nil && c.ChatStreamResponseChoice.Delta != nil {
			if c.ChatStreamResponseChoice.Delta.Reasoning != nil {
				thinking = *c.ChatStreamResponseChoice.Delta.Reasoning
			}
		}
		if c.FinishReason != nil {
			done = true
			doneReason = *c.FinishReason
		}
	}

	chunk := &OllamaGenerateResponse{
		Model:      resp.Model,
		CreatedAt:  nowISO(),
		Response:   text,
		Thinking:   thinking,
		Done:       done,
		DoneReason: doneReason,
	}
	if done && resp.Usage != nil {
		chunk.PromptEvalCount = resp.Usage.PromptTokens
		chunk.EvalCount = resp.Usage.CompletionTokens
	}

	line, err := ndjsonLine(chunk)
	if err != nil {
		return "", nil, err
	}
	return "", line, nil
}

func toOllamaChatStreamChunk(resp *schemas.BifrostChatResponse) (string, interface{}, error) {
	msg := OllamaChatMessage{Role: "assistant"}
	doneReason := ""
	done := false

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		msg.Content = extractTextFromChatStreamChoice(c)
		msg.Thinking = extractThinkingFromChatStreamChoice(c)
		if c.FinishReason != nil {
			done = true
			doneReason = *c.FinishReason
		}
	}

	chunk := &OllamaChatResponse{
		Model:      resp.Model,
		CreatedAt:  nowISO(),
		Message:    msg,
		Done:       done,
		DoneReason: doneReason,
	}
	if done && resp.Usage != nil {
		chunk.PromptEvalCount = resp.Usage.PromptTokens
		chunk.EvalCount = resp.Usage.CompletionTokens
	}

	line, err := ndjsonLine(chunk)
	if err != nil {
		return "", nil, err
	}
	return "", line, nil
}

// CreateOllamaRouteConfigs returns route configurations for Ollama-compatible endpoints.
func CreateOllamaRouteConfigs(pathPrefix string) []RouteConfig {
	var routes []RouteConfig

	// POST /api/generate → text completion
	routes = append(routes, RouteConfig{
		Type:   RouteConfigTypeOllama,
		Path:   pathPrefix + "/api/generate",
		Method: "POST",
		GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
			return schemas.TextCompletionRequest
		},
		GetRequestTypeInstance: func(ctx context.Context) interface{} {
			return &OllamaGenerateRequest{}
		},
		RequestConverter: func(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
			r, ok := req.(*OllamaGenerateRequest)
			if !ok {
				return nil, errors.New("invalid request type for ollama generate")
			}
			provider, model := schemas.ParseModelString(r.Model, schemas.Ollama)
			prompt := r.Prompt
			if r.System != "" {
				prompt = r.System + "\n\n" + prompt
			}
			return &schemas.BifrostRequest{
				TextCompletionRequest: &schemas.BifrostTextCompletionRequest{
					Provider: provider,
					Model:    model,
					Input:    &schemas.TextCompletionInput{PromptStr: &prompt},
					Params:   ollamaOptionsToTextParams(r.Options, r.Suffix),
				},
			}, nil
		},
		TextResponseConverter: func(ctx *schemas.BifrostContext, resp *schemas.BifrostTextCompletionResponse) (interface{}, error) {
			return toOllamaGenerateResponse(resp), nil
		},
		ErrorConverter: func(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
			return toOllamaErrorResponse(err)
		},
		StreamConfig: &StreamConfig{
			TextStreamResponseConverter: func(ctx *schemas.BifrostContext, resp *schemas.BifrostTextCompletionResponse) (string, interface{}, error) {
				return toOllamaGenerateStreamChunk(resp)
			},
			ErrorConverter: func(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
				line, _ := ndjsonLine(toOllamaErrorResponse(err))
				return line
			},
		},
	})

	// POST /api/chat → chat completion
	routes = append(routes, RouteConfig{
		Type:   RouteConfigTypeOllama,
		Path:   pathPrefix + "/api/chat",
		Method: "POST",
		GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
			return schemas.ChatCompletionRequest
		},
		GetRequestTypeInstance: func(ctx context.Context) interface{} {
			return &OllamaChatRequest{}
		},
		RequestConverter: func(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
			r, ok := req.(*OllamaChatRequest)
			if !ok {
				return nil, errors.New("invalid request type for ollama chat")
			}
			provider, model := schemas.ParseModelString(r.Model, schemas.Ollama)
			msgs := make([]schemas.ChatMessage, 0, len(r.Messages))
			for _, m := range r.Messages {
				msgs = append(msgs, ollamaChatMessageToBifrost(m))
			}
			return &schemas.BifrostRequest{
				ChatRequest: &schemas.BifrostChatRequest{
					Provider: provider,
					Model:    model,
					Input:    msgs,
					Params:   ollamaOptionsToChatParams(r.Options, r.Format, r.Think, r.Tools),
				},
			}, nil
		},
		ChatResponseConverter: func(ctx *schemas.BifrostContext, resp *schemas.BifrostChatResponse) (interface{}, error) {
			return toOllamaChatResponse(resp), nil
		},
		ErrorConverter: func(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
			return toOllamaErrorResponse(err)
		},
		StreamConfig: &StreamConfig{
			ChatStreamResponseConverter: func(ctx *schemas.BifrostContext, resp *schemas.BifrostChatResponse) (string, interface{}, error) {
				return toOllamaChatStreamChunk(resp)
			},
			ErrorConverter: func(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
				line, _ := ndjsonLine(toOllamaErrorResponse(err))
				return line
			},
		},
	})

	// POST /api/embed → embeddings
	routes = append(routes, RouteConfig{
		Type:   RouteConfigTypeOllama,
		Path:   pathPrefix + "/api/embed",
		Method: "POST",
		GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
			return schemas.EmbeddingRequest
		},
		GetRequestTypeInstance: func(ctx context.Context) interface{} {
			return &OllamaEmbedRequest{}
		},
		RequestConverter: func(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
			r, ok := req.(*OllamaEmbedRequest)
			if !ok {
				return nil, errors.New("invalid request type for ollama embed")
			}
			provider, model := schemas.ParseModelString(r.Model, schemas.Ollama)
			var input schemas.EmbeddingInput
			switch v := r.Input.(type) {
			case string:
				input.Text = &v
			case []interface{}:
				strs := make([]string, 0, len(v))
				for _, item := range v {
					if s, ok := item.(string); ok {
						strs = append(strs, s)
					}
				}
				input.Texts = strs
			case []string:
				input.Texts = v
			default:
				return nil, fmt.Errorf("unsupported input type for ollama embed: %T", r.Input)
			}

			var params *schemas.EmbeddingParameters
			if r.Dimensions != nil || r.Truncate != nil {
				params = &schemas.EmbeddingParameters{Dimensions: r.Dimensions}
				if r.Truncate != nil {
					params.ExtraParams = map[string]interface{}{"truncate": *r.Truncate}
				}
			}

			return &schemas.BifrostRequest{
				EmbeddingRequest: &schemas.BifrostEmbeddingRequest{
					Provider: provider,
					Model:    model,
					Input:    &input,
					Params:   params,
				},
			}, nil
		},
		EmbeddingResponseConverter: func(ctx *schemas.BifrostContext, resp *schemas.BifrostEmbeddingResponse) (interface{}, error) {
			embeddings := make([][]float64, 0, len(resp.Data))
			for _, d := range resp.Data {
				embeddings = append(embeddings, d.Embedding.EmbeddingArray)
			}
			return &OllamaEmbedResponse{
				Model:      resp.Model,
				Embeddings: embeddings,
			}, nil
		},
		ErrorConverter: func(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
			return toOllamaErrorResponse(err)
		},
	})

	return routes
}

// NewOllamaRouter creates a new OllamaRouter.
func NewOllamaRouter(client *bifrost.Bifrost, handlerStore lib.HandlerStore, logger schemas.Logger) *OllamaRouter {
	return &OllamaRouter{
		GenericRouter: NewGenericRouter(client, handlerStore, CreateOllamaRouteConfigs("/ollama"), nil, logger),
	}
}
