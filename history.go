package main

import (
	"encoding/json"
)

func (h History) ToJson() ([]byte, error) {
	return json.Marshal(h)
}

func (h History) ToString() (string, error) {
	b, err := json.Marshal(h)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
