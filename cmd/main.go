package main

import (
	"context"
	"fmt"
	"os"

	"github.com/pulumi/pulumi-go-provider/infer"

	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/resources"
	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/version"
)

func main() {
	provider, err := infer.NewProviderBuilder().
		WithResources(
			infer.Resource(&resources.VertexModelDeployment{}),
		).
		WithNamespace("davidmontoyago").
		WithDisplayName("pulumi-gcp-vertex-model-deployment").
		WithLicense("Apache-2.0").
		WithKeywords("pulumi", "gcp", "vertex", "model").
		WithDescription("Deploy AI models to Vertex endpoints").
		WithRepository("github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment").
		Build()

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error building provider: %s", err.Error())
		os.Exit(1)
	}

	// Name of the pulumi plugin.
	pluginName := "gcp-vertex-model-deployment"
	err = provider.Run(context.Background(), pluginName, version.Version)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running provider: %s", err.Error())
		os.Exit(1)
	}
}
