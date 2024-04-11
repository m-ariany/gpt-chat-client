package chatclient

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	ai "github.com/sashabaranov/go-openai"
)

type Client struct {
	client                *ai.Client
	history               History
	model                 string
	apiTimeout            time.Duration
	temperature           float32
	topP                  float32
	maxTokens             int
	presencePenalty       float32
	frequencyPenalty      float32
	logitBias             map[string]int
	tokenizer             tokenizer
	memoryTokenSize       *int
	memoryMessageSize     *int
	totalConsumedTokens   int
	moderatePromptMessage bool
	moderateResponse      bool
}

// NewClient instantiates a new chat client. Note that clients are not concurrency-safe. For concurrent usage, it's recommended to create separate client instances.
func NewClient(cnf ClientConfig) (*Client, error) {
	if len(cnf.ApiKey) == 0 {
		return nil, fmt.Errorf("ApiKey must be present")
	}

	clientConfig := ai.DefaultConfig(cnf.ApiKey)
	if len(cnf.ApiUrl) > 0 {
		clientConfig.BaseURL = cnf.ApiUrl
	} else {
		clientConfig.BaseURL = "https://api.openai.com/v1"
	}

	tokenizer, err := newTokenzier()
	if err != nil {
		return nil, err
	}

	c := &Client{
		client: ai.NewClientWithConfig(clientConfig),
		// Typically, a conversation is formatted with a system message first,
		// followed by alternating user and assistant messages.
		// Ref: https://platform.openai.com/docs/guides/chat/introduction
		history:    History{},
		model:      cnf.Model,
		apiTimeout: cnf.ApiTimeout,
		tokenizer:  tokenizer,
	}

	c.applyConfig(cnf, true)

	return c, nil
}

func (c *Client) applyConfig(config ClientConfig, normalize bool) {

	if len(config.Model) > 0 {
		c.model = config.Model
	} else if normalize {
		c.model = "gpt-3.5-turbo"
	}

	if config.ApiTimeout > 0 {
		c.apiTimeout = config.ApiTimeout
	} else if normalize {
		c.apiTimeout = time.Minute * 2
	}

	if config.Temperature != nil {
		// doc: https://github.com/sashabaranov/go-openai#frequently-asked-questions
		if *config.Temperature == 0 {
			c.temperature = 0.0000001
		} else {
			c.temperature = *config.Temperature
		}
	}

	if config.TopP != nil {
		c.topP = *config.TopP
	} else if normalize {
		c.topP = 1
	}

	c.maxTokens = config.MaxTokens

	c.presencePenalty = config.PresencePenalty

	c.frequencyPenalty = config.FrequencyPenalty

	if len(config.LogitBias) > 0 {
		c.logitBias = config.LogitBias
	} else if normalize {
		c.logitBias = make(map[string]int)
	}

	if config.MemoryMessageSize != nil {
		c.memoryMessageSize = config.MemoryMessageSize
	}

	if config.MemoryTokenSize != nil {
		c.memoryTokenSize = config.MemoryTokenSize
	}

	if config.ModeratePromptMessage != nil {
		c.moderatePromptMessage = *config.ModeratePromptMessage
	}

	if config.ModerateResponse != nil {
		c.moderateResponse = *config.ModerateResponse
	}
}

// Clone a new chat client with an empty history
func (c *Client) Clone() *Client {
	return &Client{
		client:                c.client,
		history:               History{},
		model:                 c.model,
		apiTimeout:            c.apiTimeout,
		temperature:           c.temperature,
		topP:                  c.topP,
		presencePenalty:       c.presencePenalty,
		frequencyPenalty:      c.frequencyPenalty,
		logitBias:             c.logitBias,
		tokenizer:             c.tokenizer,
		memoryTokenSize:       c.memoryTokenSize,
		memoryMessageSize:     c.memoryMessageSize,
		moderatePromptMessage: c.moderatePromptMessage,
		moderateResponse:      c.moderateResponse,
	}
}

// CloneWithConfig creates a new chat client instance with an empty history,
// inheriting configurations from the source client while allowing overrides through the specified ClientConfig.
// This method enables the modification of selected parameters from the source client's configuration.
func (c *Client) CloneWithConfig(config ClientConfig) *Client {
	cc := c.Clone()
	cc.applyConfig(config, false)
	return cc
}

// Instruct sends an instruction to the client, providing system message.
// If length of the instruction exceeds the allowed context length of the underlying model, it returns an error.
func (c *Client) Instruct(instruction string) error {

	if c.tokenizer.CountTokens(instruction) > getModel(c.model).MaxInstructionLength() {
		return fmt.Errorf("max length of instruction is %d", getModel(c.model).MaxInstructionLength())
	}

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

	return nil
}

// Instruct sends an instruction to the client, providing system message.
// If length of the instruction exceeds the allowed context length of the underlying model, it trims the instruction to fit.
func (c *Client) InstructWithLengthFix(instruction string) {

	for c.tokenizer.CountTokens(instruction) > getModel(c.model).MaxInstructionLength() {
		diffToken := c.tokenizer.CountTokens(instruction) - getModel(c.model).MaxInstructionLength()
		diffChar := diffToken * 3 // each token is roughly 3 latin characters
		instruction = instruction[:len(instruction)-diffChar]
	}

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

// Prompt sends a prompt to the OpenAI API for generating a response.
// It returns the generated response or an error.
// Errors returned can be of types ErrModerationUserInput or ErrModerationModelOutput
// if moderation flags are enabled and moderation fails, otherwise, it can be other types of errors from the underlying operations.
func (c *Client) Prompt(ctx context.Context, prompt string) (string, error) {

	if c.moderatePromptMessage {
		err := c.moderateInput(ctx, prompt)
		if err == ErrModeration {
			return "", ErrModerationUserInput
		}
		if err != nil {
			return "", err
		}
	}

	retryHandler := newRetryHandler(time.Second*5, 5)
	var err error
	var response string

	retryHandler.Do(func() error {
		response, err = c.prompt(ctx, prompt)
		if err != nil {
			log.Printf("retry calling openai %v", err)
		}
		return err
	})

	if err != nil {
		return "", err
	}

	if c.moderateResponse {
		err := c.moderateInput(ctx, response)
		if err == ErrModeration {
			return "", ErrModerationModelOutput
		}
		if err != nil {
			return "", err
		}
	}

	return response, nil
}

// PromptStream sends a prompt to the OpenAI API for generating a response,
// and returns a channel of Stream objects containing response chunks or errors.
// The Chunk field in Stream struct contains response chunks,
// and the Err field indicates any errors encountered during the streaming process.
// Errors returned can be of types ErrModerationUserInput if moderation flags are enabled and moderation fails,
// otherwise, it can be other types of errors from the underlying operations.
//
// Since respose is returned as stream to the client, no moderation on the response can be done in this level.
func (c *Client) PromptStream(ctx context.Context, question string) <-chan Stream {

	ch := make(chan Stream)

	go func() {
		defer close(ch)

		if c.moderatePromptMessage {
			err := c.moderateInput(ctx, question)
			if err == ErrModeration {
				ch <- Stream{Err: ErrModerationUserInput}
				return
			}
			if err != nil {
				ch <- Stream{Err: err}
				return
			}
		}

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
			if err != nil {
				ch <- Stream{Err: err}
				break
			}

			chunk := data.Choices[0].Delta.Content
			select {
			case ch <- Stream{Chunk: chunk}:
			case <-ctx.Done():
				// do not return or break as the next stream.Recv() will return error and exit the loop
			}

			sb.WriteString(chunk)
		}

		c.postStreamResponse(sb.String())
	}()

	return ch
}

// Import history to the client
func (c *Client) ImportHistory(history History) {
	c.history = append(c.history, history...)
	c.trimHistory()
}

// Export current history of the client
func (c *Client) ExportHistory() History {
	return c.history
}

// Get total number of input and output tokens consumed by the client
func (c *Client) TotalConsumedTokens() int {
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
	c.billConsumedTokens(resp.Usage.TotalTokens)
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

	c.trimHistory()

	request := ai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    c.history,
		Temperature: c.temperature,
		Stream:      stream,
	}

	if c.maxTokens > 0 {
		request.MaxTokens = c.maxTokens
	}

	return request
}

// Trim history to fit the maximum number of tokens or messages allowed.
func (c *Client) trimHistory() {

	if c.memoryTokenSize != nil {
		c.trimHistoryToMatchTokenLimit(*c.memoryTokenSize)
	}

	if c.memoryMessageSize != nil {
		c.trimHistoryToMatchMessageLimit()
	}

	// to make sure that the remained context does not exceed the allowed model's context length
	c.trimHistoryToMatchTokenLimit(getModel(c.model).ContextLength())
}

func (c *Client) trimHistoryToMatchTokenLimit(limit int) error {
	// there is only a system message
	if len(c.history) == 1 {
		return nil
	}

	// exclude instruction from the operation
	historyToString := func() (string, error) {
		return c.history[1:].ToString()
	}

	historyAsString, err := historyToString()
	if err != nil {
		return err
	}

	for c.tokenizer.CountTokens(historyAsString) > limit {
		// only system message and one additional message is remained.
		// delete the additional message.
		if len(c.history) == 2 {
			c.history = c.history[:1]
			break
		}

		// shave the oldest messages first
		copy(c.history[1:], c.history[2:])
		c.history = c.history[:len(c.history)-1]

		if historyAsString, err = historyToString(); err != nil {
			return err
		}
	}

	return nil
}

func (c *Client) trimHistoryToMatchMessageLimit() {
	memorySize := *c.memoryMessageSize
	// exclude instruction from the operation
	if len(c.history)-1 <= memorySize {
		return
	}
	// shave the oldest messages first
	c.history = append(c.history[:1], c.history[1+len(c.history)-memorySize:]...)
}

func (c *Client) postStreamResponse(r string) {
	if len(r) == 0 {
		return
	}

	c.history = append(c.history, ai.ChatCompletionMessage{
		Role:    ai.ChatMessageRoleAssistant,
		Content: r,
	})

	history, err := c.history.ToString()
	if err != nil {
		log.Println("failed to bill consumed tokens")
	}
	n := c.tokenizer.CountTokens(history)
	c.billConsumedTokens(n)
}

func (c *Client) billConsumedTokens(n int) {
	c.totalConsumedTokens += n
}

// moderateInput sends the input string to the OpenAI API for moderation.
// It returns an error if there's an issue with the API call or if the input is flagged for moderation.
// Otherwise, it returns nil.
func (c *Client) moderateInput(ctx context.Context, input string) error {

	result, err := c.client.Moderations(ctx, ai.ModerationRequest{
		Input: input,
		Model: ai.ModerationTextStable,
	})

	if err != nil {
		return err
	}

	if result.Results[0].Flagged {
		return ErrModeration
	}

	return nil
}
