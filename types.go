package main

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
		ApiKey                    string
		Model                     string
		ApiUrl                    string
		ApiTimeout                time.Duration
		Temperature               *float32
		LimitMemoryByToken        *bool
		LimitMemoryByMessage      *bool
		MemoryTokenSize           *int
		MemoryMessageSize         *int
		MemorizeAssistantMessages *bool
	}
)
