package chatclient

import (
	"time"

	"github.com/sashabaranov/go-openai"
	ai "github.com/sashabaranov/go-openai"
)

type (
	History []ai.ChatCompletionMessage

	Stream struct {
		Chunk string
		Err   error
	}

	ClientConfig struct {
		openai.ChatCompletionRequest
		// ApiUrl is the URL of the OpenAI API.
		ApiUrl string

		// ApiKey is the authentication key required to access the OpenAI API.
		ApiKey string

		// Model specifies the name or identifier of the chat model to be used.
		Model string

		// ApiTimeout specifies the maximum duration to wait for a response from the OpenAI API.
		ApiTimeout time.Duration

		// Temperature controls the randomness of the model's responses. Higher values lead to more random responses.
		Temperature *float32

		// TopP is the nucleus sampling probability threshold, controlling the randomness of token generation.
		TopP *float32 `json:"top_p,omitempty"`

		// PresencePenalty decreases the likelihood of already chosen tokens, controlling repetition in generation.
		PresencePenalty float32 `json:"presence_penalty,omitempty"`

		// FrequencyPenalty reduces the probability of repeated tokens, decreasing redundancy in the output.
		FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`

		// LogitBias allows manual adjustment of the probability of specified tokens during generation.
		LogitBias map[string]int `json:"logit_bias,omitempty"`

		// MemoryTokenSize specifies the maximum number of tokens to remember in the conversation history.
		MemoryTokenSize *int

		// MemoryMessageSize specifies the maximum number of messages to remember in the conversation history.
		MemoryMessageSize *int

		// ModeratePromptMessage indicates whether the client should check the prompt message agains the moderation endpoint.
		ModeratePromptMessage *bool

		// ModerateResponse indicates whether the client should check the response against the moderation endpoint.
		ModerateResponse *bool
	}
)
