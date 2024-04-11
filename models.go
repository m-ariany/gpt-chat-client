package chatclient

import (
	"strings"
)

// Ref: https://platform.openai.com/docs/models

type ModelCategory int

const (
	gpt35Model ModelCategory = iota
	gpt4Model
	unknown

	instructionTokenBuffer = 100
)

// Model Spec represent some attributes of the model
type llmModel interface {
	Name() string
	MaxInstructionLength() int
	ContextLength() int
	CompletionLength() int
}

type baseGpt struct{}

func (g baseGpt) CompletionLength() int {
	return 4096
}

// <!+++ gpt4

type gpt4PreviewBase struct {
	baseGpt
}

func (g gpt4PreviewBase) ContextLength() int {
	return 128 * 1000
}

func (g gpt4PreviewBase) MaxInstructionLength() int {
	return g.ContextLength() - instructionTokenBuffer
}

type gpt4Turbo struct {
	gpt4PreviewBase
}

func (g gpt4Turbo) Name() string {
	return "gpt-4-turbo"
}

type gpt40125Preview struct {
	gpt4PreviewBase
}

func (g gpt40125Preview) Name() string {
	return "gpt-4-0125-preview"
}

type gpt4TurboPreview struct {
	gpt40125Preview
}

func (g gpt4TurboPreview) Name() string {
	return "gpt-4-turbo-preview"
}

type gpt41106Preview struct {
	gpt4PreviewBase
}

func (g gpt41106Preview) Name() string {
	return "gpt-4-1106-preview"
}

type gpt41106VisionPreview struct {
	gpt4PreviewBase
}

func (g gpt41106VisionPreview) Name() string {
	return "gpt-4-1106-vision-preview"
}

type gpt4VisionPreview struct {
	gpt41106VisionPreview
}

func (g gpt4VisionPreview) Name() string {
	return "gpt-4-1106-preview"
}

type gpt40613 struct {
	baseGpt
}

func (g gpt40613) Name() string {
	return "gpt-4-0613"
}

func (g gpt40613) ContextLength() int {
	return 8192
}

func (g gpt40613) MaxInstructionLength() int {
	return g.ContextLength() - instructionTokenBuffer
}

type gpt4 struct {
	gpt40613
}

func (g gpt4) Name() string {
	return "gpt-4"
}

type gpt432k0613 struct {
	baseGpt
}

func (g gpt432k0613) Name() string {
	return "gpt-4-32k-0613"
}

func (g gpt432k0613) ContextLength() int {
	return 32768
}

func (g gpt432k0613) MaxInstructionLength() int {
	return g.ContextLength() - instructionTokenBuffer
}

type gpt432k struct {
	gpt432k0613
}

func (g gpt432k) Name() string {
	return "gpt-4-32k"
}

// gpt4 ---!>

// <!+++ gpt35

type gpt35Base struct {
	baseGpt
}

func (g gpt35Base) ContextLength() int {
	return 16385
}

func (g gpt35Base) MaxInstructionLength() int {
	return g.ContextLength() - instructionTokenBuffer
}

type gpt35turbo1106 struct {
	gpt35Base
}

func (g gpt35turbo1106) Name() string {
	return "gpt-3.5-turbo-1106"
}

type gpt35turbo0125 struct {
	gpt35Base
}

func (g gpt35turbo0125) Name() string {
	return "gpt-3.5-turbo-0125"
}

type gpt35turbo struct {
	gpt35turbo0125
}

func (g gpt35turbo) Name() string {
	return "gpt-3.5-turbo"
}

// gpt35 ---!>

func getModel(m string) llmModel {
	switch strings.TrimSpace(strings.ToLower(m)) {
	case gpt4Turbo{}.Name():
		return gpt4Turbo{}
	case gpt4TurboPreview{}.Name():
		return gpt4TurboPreview{}
	case gpt4VisionPreview{}.Name():
		return gpt4VisionPreview{}
	case gpt41106VisionPreview{}.Name():
		return gpt41106VisionPreview{}
	case gpt40125Preview{}.Name():
		return gpt40125Preview{}
	case gpt41106Preview{}.Name():
		return gpt41106Preview{}
	case gpt4{}.Name():
		return gpt4{}
	case gpt40613{}.Name():
		return gpt40613{}
	case gpt432k{}.Name():
		return gpt432k{}
	case gpt432k0613{}.Name():
		return gpt432k0613{}

	case gpt35turbo0125{}.Name():
		return gpt35turbo0125{}
	case gpt35turbo{}.Name():
		return gpt35turbo{}
	case gpt35turbo1106{}.Name():
		return gpt35turbo1106{}
	}
	return nil
}

func getModelCategory(m string) ModelCategory {
	switch strings.TrimSpace(strings.ToLower(m)) {

	case gpt4Turbo{}.Name():
		return gpt4Model
	case gpt4TurboPreview{}.Name():
		return gpt4Model
	case gpt4VisionPreview{}.Name():
		return gpt4Model
	case gpt41106VisionPreview{}.Name():
		return gpt4Model
	case gpt40125Preview{}.Name():
		return gpt4Model
	case gpt41106Preview{}.Name():
		return gpt4Model
	case gpt4{}.Name():
		return gpt4Model
	case gpt40613{}.Name():
		return gpt4Model
	case gpt432k{}.Name():
		return gpt4Model
	case gpt432k0613{}.Name():
		return gpt4Model

	case gpt35turbo0125{}.Name():
		return gpt35Model
	case gpt35turbo{}.Name():
		return gpt35Model
	case gpt35turbo1106{}.Name():
		return gpt35Model
	}
	return unknown
}
