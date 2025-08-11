VERSION         := 0.2.0
PROVIDER_NAME   := vertex-model-deployment
PROVIDER_PATH   := github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment

.PHONY: build clean test lint

build: clean
	go mod download
	go build -o ./build/pulumi-resource-$(PROVIDER_NAME) ./cmd

test: build
	go test -v -race -count=1 -timeout=30s -coverprofile=coverage.out ./...

clean:
	rm -rf ./sdk/*
	go mod tidy
	go mod verify

lint:
	docker run --rm -v $$(pwd):/app \
		-v $$(go env GOCACHE):/.cache/go-build -e GOCACHE=/.cache/go-build \
		-v $$(go env GOMODCACHE):/.cache/mod -e GOMODCACHE=/.cache/mod \
		-w /app golangci/golangci-lint:v2.1.6 \
		golangci-lint run --fix --verbose --output.text.colors

upgrade:
	go get -u ./...

install: gen-sdk
	@echo "Installing provider..."
	@cp ./build/pulumi-resource-$(PROVIDER_NAME) $(shell go env GOPATH)/bin/
	@echo "Provider installed successfully"

gen-sdk: build
	@echo "Generating SDKs..."
	@pulumi package gen-sdk ./build/pulumi-resource-$(PROVIDER_NAME)
	@echo "SDKs generated successfully"
	cd sdk/go && go mod init github.com/davidmontoyago/gcp-vertex-model-deployment/sdk/go && go mod tidy
	pulumi plugin install resource gcp-vertex-model-deployment v0.2.0 -f ./build/pulumi-resource-$(PROVIDER_NAME)
