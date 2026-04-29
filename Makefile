IMAGE ?= llmpostor:latest

.PHONY: build test docker-build docker-run

build:
	go build -o bin/llmpostor ./cmd/llmpostor

test:
	go test ./...

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8080:8080 $(IMAGE)
