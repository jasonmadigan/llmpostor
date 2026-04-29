FROM golang:1.26 AS builder

WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /llmpostor ./cmd/llmpostor

FROM gcr.io/distroless/static:nonroot

COPY --from=builder /llmpostor /llmpostor

EXPOSE 8080

USER 65532:65532

ENTRYPOINT ["/llmpostor"]
