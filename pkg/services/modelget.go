package services

import (
	"context"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
)

// ModelGetter allows getting models from the registry.
type ModelGetter interface {
	Get(ctx context.Context, modelName string) (*aiplatformpb.Model, error)
	Close() error
}

// VertexModelGet implements the ModelGetter interface for Vertex AI.
type VertexModelGet struct {
	modelClient VertexModelClient
	modelName   string
}

// NewVertexModelGet creates a new VertexModelGet with the provided model client.
func NewVertexModelGet(_ context.Context, modelClient VertexModelClient, modelName string) *VertexModelGet {
	return &VertexModelGet{
		modelClient: modelClient,
		modelName:   modelName,
	}
}

// Get gets a model from Vertex AI.
func (g *VertexModelGet) Get(ctx context.Context, modelName string) (*aiplatformpb.Model, error) {
	getReq := &aiplatformpb.GetModelRequest{
		Name: modelName,
	}

	model, err := g.modelClient.GetModel(ctx, getReq)
	if err != nil {
		return nil, err
	}

	return model, nil
}

// Close closes the model client.
func (g *VertexModelGet) Close() error {
	return g.modelClient.Close()
}
