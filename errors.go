package chatclient

import (
	"errors"
)

var (
	ErrModeration            = errors.New("input is potentially harmful")
	ErrModerationUserInput   = errors.New("input is potentially harmful")
	ErrModerationModelOutput = errors.New("output is potentially harmful")
)
