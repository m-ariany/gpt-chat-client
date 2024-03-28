package chatclient

import (
	"crypto/rand"
	"math"
	"math/big"
	"sync"
	"time"
)

type retryHandler struct {
	rndMu    sync.Mutex
	maxDelay time.Duration
	maxRetry int
}

func newRetryHandler(maxDelay time.Duration, maxRetry int) *retryHandler {
	return &retryHandler{
		rndMu:    sync.Mutex{},
		maxDelay: maxDelay,
		maxRetry: maxRetry,
	}
}

type CallFunc func() error

func (b *retryHandler) Do(c CallFunc) {
	for i := 0; i < b.maxRetry+1; i++ {
		if err := c(); err == nil {
			return
		}
		if i < b.maxRetry {
			b.backoff(i)
		}
	}
}

// Backoff is blocking and will return after the backoff duration.
func (b *retryHandler) backoff(retryCount int) {

	if b.maxDelay == 0 {
		return
	}

	b.rndMu.Lock()
	defer b.rndMu.Unlock()

	t := time.Duration(1<<uint(retryCount)) * time.Second
	backoff := time.Duration(math.Min(float64(t), float64(b.maxDelay)))
	center := backoff / 2
	var ri = int64(center)
	var jitter = newRnd(ri)

	sleepTime := time.Duration(math.Abs(float64(ri + jitter)))
	if sleepTime > b.maxDelay {
		sleepTime = b.maxDelay
	}
	<-time.After(sleepTime)
}

func newRnd(cap int64) int64 {
	// Generate a random number between 0 and cap
	randomInt, err := rand.Int(rand.Reader, big.NewInt(cap-1))
	if err != nil {
		return 0
	}

	return randomInt.Int64()
}
