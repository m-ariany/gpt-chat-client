package chatclient

import (
	"time"

	ai "github.com/sashabaranov/go-openai"
)

type (
	History []ai.ChatCompletionMessage

	Stream struct {
		Chunk string
		Err   error
	}

	ClientConfig struct {
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

		// LimitMemoryByToken indicates whether the client should limit memory usage based on the number of tokens.
		LimitMemoryByToken *bool

		// LimitMemoryByMessage indicates whether the client should limit memory usage based on the number of messages.
		LimitMemoryByMessage *bool

		// MemoryTokenSize specifies the maximum number of tokens to remember in the conversation history.
		MemoryTokenSize *int

		// MemoryMessageSize specifies the maximum number of messages to remember in the conversation history.
		MemoryMessageSize *int

		// MemorizeAssistantMessages indicates whether the client should remember assistant messages in the conversation history.
		MemorizeAssistantMessages *bool
	}
)
