GOOS							?= $${GOOS:-linux}
GOARCH						?= $${GOARCH:-amd64}
PROVIDER_VERSION	?= $${PROVIDER_VERSION:-v0.0.0}
PROVIDER_NAME			:= gcp-vertex-model-deployment
PROVIDER_PATH			:= github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment
PLUGIN_NAME				:= pulumi-resource-$(PROVIDER_NAME)-$(PROVIDER_VERSION)-$(GOOS)-$(GOARCH)

.PHONY: build clean test lint

build: clean
	go mod download
	go build -o ./build/pulumi-resource-$(PROVIDER_NAME) ./cmd

test: build
	go test -v -race -count=1 -timeout=30s -coverprofile=coverage.out ./...

clean:
	go mod tidy
	go mod verify

pulumi-clean:
	rm -rf ./sdk/*
	rm -rf ./build/*

lint:
	docker run --rm -v $$(pwd):/app \
		-v $$(go env GOCACHE):/.cache/go-build -e GOCACHE=/.cache/go-build \
		-v $$(go env GOMODCACHE):/.cache/mod -e GOMODCACHE=/.cache/mod \
		-w /app golangci/golangci-lint:v2.1.6 \
		golangci-lint run --fix --verbose --output.text.colors

upgrade:
	go get -u ./...

gen-sdk: build
	@echo "Generating SDKs..."
	@pulumi package gen-sdk ./build/pulumi-resource-$(PROVIDER_NAME)
	@echo "SDKs generated successfully"
	cd sdk/go && go mod init github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/sdk/go && go mod tidy

plugin-local: plugin
	@echo "Installing provider..."
	pulumi plugin install resource gcp-vertex-model-deployment $(PROVIDER_VERSION) --file ./build/$(PLUGIN_NAME)
	@echo "Plugin installed successfully"

plugin: gen-sdk
	@set -eu && \
		mkdir -p ./build && \
		echo "Building $(PROVIDER_NAME) with version $(PROVIDER_VERSION): $(PLUGIN_NAME)" && \
		CGO_ENABLED=0 GOOS=$(GOOS) GOARCH=$(GOARCH) go build -trimpath \
			-ldflags "-s -w -X $(PROVIDER_PATH)/pkg/version.Version=$(PROVIDER_VERSION)" \
			-o ./build/$(PLUGIN_NAME) ./cmd
	@echo "Plugin built successfully"
	@echo "Compressing..."
	tar -czvf ./build/$(PLUGIN_NAME).tar.gz ./build/$(PLUGIN_NAME)
	@echo "Compressed successfully"
