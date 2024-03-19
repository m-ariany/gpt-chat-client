package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"time"

	ai "github.com/sashabaranov/go-openai"
)

type Client struct {
	client                    *ai.Client
	history                   History
	model                     string
	apiTimeout                time.Duration
	temperature               float32
	tokenizer                 tokenizer
	limitMemoryByToken        bool
	limitMemoryByMessage      bool
	memoryTokenSize           int
	memoryMessageSize         int
	memorizeAssistantMessages bool
	totalConsumedTokens       int
	mu                        sync.Mutex
}

// Create a new chat client
func NewClient(cnf ClientConfig) (*Client, error) {
	clientConfig := ai.DefaultConfig(cnf.ApiKey)
	clientConfig.BaseURL = cnf.ApiUrl
	tokenizer, err := newTokenzier()
	if err != nil {
		return nil, err
	}

	c := &Client{
		client: ai.NewClientWithConfig(clientConfig),
		// Typically, a conversation is formatted with a system message first,
		// followed by alternating user and assistant messages.
		// Ref: https://platform.openai.com/docs/guides/chat/introduction
		history:                   History{},
		model:                     cnf.Model,
		apiTimeout:                cnf.ApiTimeout,
		memorizeAssistantMessages: true,
		tokenizer:                 tokenizer,
		mu:                        sync.Mutex{},
	}

	c.applyConfig(cnf)

	return c, nil
}

func (c *Client) applyConfig(config ClientConfig) {

	if len(config.Model) > 0 {
		c.model = config.Model
	}
	if config.ApiTimeout > 0 {
		c.apiTimeout = config.ApiTimeout
	}
	if config.Temperature != nil {
		// doc: https://github.com/sashabaranov/go-openai#frequently-asked-questions
		if *config.Temperature == 0 {
			c.temperature = 0.0000001
		} else {
			c.temperature = *config.Temperature
		}
	}
	if config.LimitMemoryByMessage != nil {
		c.limitMemoryByMessage = *config.LimitMemoryByMessage
	}
	if config.LimitMemoryByToken != nil {
		c.limitMemoryByToken = *config.LimitMemoryByToken
	}
	if config.MemoryMessageSize != nil {
		c.memoryMessageSize = *config.MemoryMessageSize
	}
	if config.MemoryTokenSize != nil {
		c.memoryTokenSize = *config.MemoryTokenSize
	}
	if config.MemorizeAssistantMessages != nil {
		c.memorizeAssistantMessages = *config.MemorizeAssistantMessages
	}
}

// Clone a new chat client with an empty history
func (c *Client) Clone() *Client {
	return &Client{
		client:                    c.client,
		history:                   History{},
		model:                     c.model,
		apiTimeout:                c.apiTimeout,
		temperature:               c.temperature,
		tokenizer:                 c.tokenizer,
		limitMemoryByToken:        c.limitMemoryByToken,
		limitMemoryByMessage:      c.limitMemoryByMessage,
		memoryTokenSize:           c.memoryTokenSize,
		memoryMessageSize:         c.memoryMessageSize,
		memorizeAssistantMessages: c.memorizeAssistantMessages,
		mu:                        sync.Mutex{},
	}
}

// Clone a new chat client with an empty history and defined ClientConfig
func (c *Client) CloneWithConfig(config ClientConfig) *Client {
	cc := c.Clone()
	cc.applyConfig(config)
	return cc
}

// Instruct the model by setting the system message
func (c *Client) Instruct(instruction string) {
	if len(c.history) == 0 { // insert
		c.history = append(c.history, ai.ChatCompletionMessage{
			Role:    ai.ChatMessageRoleSystem,
			Content: instruction,
		})
	} else { // update
		c.history[0] = ai.ChatCompletionMessage{
			Role:    ai.ChatMessageRoleSystem,
			Content: instruction,
		}
	}
}

// Prompt sends a message to the model to get a response
func (c *Client) Prompt(ctx context.Context, prompt string) (string, error) {
	retryHandler := newRetryHandler(time.Second*20, time.Second*5, 5)
	var err error
	var data string

	retryHandler.Do(func() error {
		data, err = c.prompt(ctx, prompt)
		if err != nil {
			log.Printf("retry calling openai %v", err)
		}
		return err
	})

	return data, err
}

// Prompt sends a message to the model to get a stream response
func (c *Client) PromptStream(ctx context.Context, question string) <-chan Stream {

	ch := make(chan Stream)

	go func() {
		defer close(ch)

		req := c.newChatCompletionRequest(question, true)
		ctx, cancel := context.WithTimeout(ctx, c.apiTimeout)
		defer cancel()

		stream, err := c.client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			err = fmt.Errorf("failed to create chat completion stream %w", err)
			ch <- Stream{Err: err}
			return
		}
		defer stream.Close()

		sb := strings.Builder{}
		for {
			data, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}

			if err != nil {
				ch <- Stream{Err: err}
				break
			}

			chunk := data.Choices[0].Delta.Content
			select {
			case ch <- Stream{Chunk: chunk}:
			case <-time.After(time.Second * 5):
				// do not return or break
			}

			sb.WriteString(chunk)
		}

		response := sb.String()
		if len(response) == 0 {
			return
		}

		c.postResponse(response)
	}()

	return ch
}

// Import history to the client
func (c *Client) ImportHistory(history History) {
	c.history = append(c.history, history...)
	c.trimHistory()
}

// Export current history of the client
func (c *Client) ExportHistory(includeSystemMsg bool) History {
	if includeSystemMsg {
		return c.history
	}
	return c.history[1:]
}

// Get total number of input and output tokens consumed by the client
func (c *Client) TotalConsumedTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.totalConsumedTokens
}

func (c *Client) prompt(ctx context.Context, question string) (string, error) {

	req := c.newChatCompletionRequest(question, false)
	ctx, cancel := context.WithTimeout(ctx, c.apiTimeout)
	defer cancel()
	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		err = fmt.Errorf("failed to create chat completion %w", err)
		return "", err
	}

	data := resp.Choices[0].Message.Content
	c.postResponse(data)
	return data, nil
}

func (c *Client) newChatCompletionRequest(question string, stream bool) ai.ChatCompletionRequest {

	/*
		Ref: https://platform.openai.com/docs/guides/chat/introduction
		Including the conversation history helps the models to give relevant answers to the prior conversation.
		Because the models have no memory of past requests, all relevant information must be supplied via the conversation.
	*/
	c.history = append(c.history, ai.ChatCompletionMessage{
		Role:    ai.ChatMessageRoleUser,
		Content: question,
	})

	if c.limitMemoryByMessage || c.limitMemoryByToken {
		c.trimHistory()
	}

	return ai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    c.history,
		Temperature: c.temperature,
		Stream:      stream,
	}
}

// Trim history to fit the maximum number of tokens or messages allowed.
func (c *Client) trimHistory() {
	// limitMemoryByToken has priority over limitMemoryByMessage.
	// if both are given, only apply limitMemoryByToken
	if c.limitMemoryByToken {
		c.trimHistoryToMatchTokenLimit()
	} else if c.limitMemoryByMessage {
		c.trimHistoryToMatchMessageLimit()
	}
}

func (c *Client) trimHistoryToMatchTokenLimit() error {
	// exclude instruction from the operation
	// shave the oldest messages first
	historyAsString, err := c.history.ToString()
	if err != nil {
		return err
	}

	for c.tokenizer.CountTokens(historyAsString) > c.memoryTokenSize {
		copy(c.history[1:], c.history[2:])
		c.history = c.history[:len(c.history)-1]
	}

	return nil
}

func (c *Client) trimHistoryToMatchMessageLimit() {
	memorySize := c.memoryMessageSize
	// exclude instruction from the operation
	if len(c.history)-1 <= memorySize {
		return
	}
	// shave the oldest messages first
	c.history = append(c.history[:1], c.history[1+len(c.history)-memorySize:]...)
}

func (c *Client) postResponse(r string) {
	if c.memorizeAssistantMessages {
		c.history = append(c.history, ai.ChatCompletionMessage{
			Role:    ai.ChatMessageRoleAssistant,
			Content: r,
		})
	}

	c.billConsumedTokens()
}

// OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
func (c *Client) billConsumedTokens() {

	var tokensPerMessage, tokensPerName int
	switch c.model {
	case "gpt-3.5-turbo-0613",
		"gpt-3.5-turbo-16k-0613",
		"gpt-4-0314",
		"gpt-4-32k-0314",
		"gpt-4-0613",
		"gpt-4-32k-0613":
		tokensPerMessage = 3
		tokensPerName = 1
	case "gpt-3.5-turbo-0301":
		tokensPerMessage = 4 // every message follows <|start|>{role/name}\n{content}<|end|>\n
		tokensPerName = -1   // if there's a name, the role is omitted
	default:
		if strings.Contains(c.model, "gpt-3.5-turbo") {
			log.Println("warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
			tokensPerMessage = 3
			tokensPerName = 1
		} else if strings.Contains(c.model, "gpt-4") {
			log.Println("warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
			tokensPerMessage = 3
			tokensPerName = 1
		} else {
			err := fmt.Errorf("num_tokens_from_messages() is not implemented for model %s see https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens", c.model)
			log.Println(err)
			return
		}
	}

	var numTokens int
	for _, message := range c.history {
		numTokens += tokensPerMessage
		numTokens += len(c.tokenizer.Encode(message.Content, nil, nil))
		numTokens += len(c.tokenizer.Encode(message.Role, nil, nil))
		numTokens += len(c.tokenizer.Encode(message.Name, nil, nil))
		if message.Name != "" {
			numTokens += tokensPerName
		}
	}
	numTokens += 3 // every reply is primed with <|start|>assistant<|message|>

	c.mu.Lock()
	c.totalConsumedTokens += numTokens
	c.mu.Unlock()
}
