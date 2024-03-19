package chatclient

import (
	"github.com/pkoukk/tiktoken-go"
)

var _tokenizer *tiktoken.Tiktoken

func initTokenizer() error {
	if _tokenizer != nil {
		return nil
	}

	tkm, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return err
	}

	_tokenizer = tkm

	return nil
}

type tokenizer struct {
	*tiktoken.Tiktoken
}

func newTokenzier() (tokenizer, error) {
	if err := initTokenizer(); err != nil {
		return tokenizer{}, err
	}

	return tokenizer{Tiktoken: _tokenizer}, nil
}

func (t tokenizer) CountTokens(s string) int {
	token := t.Tiktoken.Encode(s, nil, nil)
	return len(token)
}
