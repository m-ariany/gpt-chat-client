# OpenAI Chat Client

This Go package provides a client for interacting with OpenAI's chat API. It allows users to interact with various functionalities of OpenAI's chat models, including sending prompts, receiving responses, managing conversation history, and configuring client settings.

## Features:

- **Streaming/Non-Streaming Responses**: Supports both types of responses.
- **Moderation**: Available on user input and model output.
- **Context Size Management**: Manage context size via message count or token limit. By default, context will not grow beyond the context window of the underlying model.
- **History Import/Export**: Facilitates easy data transfer.
- **Token Consumption Tracking**: Returns total consumed tokens per client object.

## Installation

To use this package, you need to have Go installed. You can install the package using the following command:

```bash
go get github.com/m-ariany/gpt-chat-client
```

## Creating a Client

To create a new chat client, you can use the `NewClient` function. It requires a `ClientConfig` struct containing necessary configurations such as API key, model, and API URL. Example:

```go
import (
   gpt "github.com/m-ariany/gpt-chat-client"
)

config := gpt.ClientConfig{
    ApiKey:     "your_openai_api_key",
    ApiUrl:     "https://api.openai.com/v1",
    Model:      "gpt-4-turbo",
    ApiTimeout: time.Second * 30, // optional: specify API timeout
}

client, err := gpt.NewClient(config)
if err != nil {
    // handle error
}
```

You can clone a client with or without configurations using the `Clone` and `CloneWithConfig` functions, respectively.


### Instructing the Model

You can instruct the model by setting a system message using the `Instruct` function:

```go
err := client.Instruct("Your instruction message")
if err != nil {
    fmt.Println(err)  // instruction message is longer than the context window size of the underlying model.
}
```

You can call the `InstructWithLengthFix` method to automatically trim the instruction if it exceeds the context window size of the underlying model.
```go
client.InstructWithLengthFix("Your very long instruction message")
```

### Sending a Prompt

You can send a prompt to the model to get a response using the `Prompt` function:

```go
response, err := client.Prompt(ctx, "Your prompt message")
if err != nil {
    // handle error
}
```

### Sending a Prompt with Streaming Response

To get a stream of responses from the model, you can use the `PromptStream` function:

```go
stream := client.PromptStream(ctx, "Your prompt message")

for response := range stream {
    if response.Err != nil {
        // handle error
    } else {
        // process response.Chunk
    }
}
```

### Enable moderation on the user input or model response

You can set the `ModeratePromptMessage` or `ModerateResponse` flags to use OpenAI moderation endpoint on the user's input or model's output:

```go

import (
    gpt "github.com/m-ariany/gpt-chat-client"
)

...

config := gpt.ClientConfig{
     // other configurations
     ...

     // Set moderation flags
     ModeratePromptMessage: true
     ModerateResponse: true
}

client, _ := gpt.NewClient(config)

response, err := client.Prompt(ctx, "You are a fucking asshole!")

if err == gpt.ErrModerationUserInput {
    fmt.Println("User input is potentially harmful!")
}
```

### Importing and Exporting History

You can import and export conversation history using the `ImportHistory` and `ExportHistory` functions, respectively:

```go
// Import history
client.ImportHistory(history)

// Export history
history := client.ExportHistory(includeSystemMsg)
```

### Getting Total Consumed Tokens

To get the total number of input and output tokens consumed by the client, you can use the `TotalConsumedTokens` function:

```go
totalTokens := client.TotalConsumedTokens()
```

## Additional Information

For more details on the functionalities and configurations, please refer to the code documentation and OpenAI's documentation.

## License

This package is licensed under the MIT License. See the LICENSE file for details.
