package chatclient

import (
	"time"

	"github.com/sashabaranov/go-openai"
	ai "github.com/sashabaranov/go-openai"
)

type (
	History []ai.ChatCompletionMessage

	ChatConfig = openai.ChatCompletionRequest

	Stream struct {
		Chunk string
		Err   error
	}

	ClientConfig struct {
		// ApiUrl is the URL of the OpenAI API.
		ApiUrl string

		// ApiKey is the authentication key required to access the OpenAI API.
		ApiKey string

		// ApiTimeout specifies the maximum duration to wait for a response from the OpenAI API.
		ApiTimeout time.Duration

		// MemoryTokenSize specifies the maximum number of tokens to remember in the conversation history.
		MemoryTokenSize *int

		// MemoryMessageSize specifies the maximum number of messages to remember in the conversation history.
		MemoryMessageSize *int

		// ModeratePromptMessage indicates whether the client should check the prompt message agains the moderation endpoint.
		ModeratePromptMessage *bool

		// ModerateResponse indicates whether the client should check the response against the moderation endpoint.
		ModerateResponse *bool

		ChatConfig ChatConfig
	}
)
