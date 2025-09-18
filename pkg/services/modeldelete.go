package services

import (
	"context"
	"fmt"
	"log"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
)

// ModelDeleter allows deleting models from the registry.
type ModelDeleter interface {
	Delete(ctx context.Context, modelName string) error
	Close() error
}

// VertexModelDelete implements the ModelDeleter interface for Vertex AI.
type VertexModelDelete struct {
	modelClient VertexModelClient
}

// NewVertexModelDelete creates a new VertexModelDelete with the provided model client.
func NewVertexModelDelete(_ context.Context, modelClient VertexModelClient) *VertexModelDelete {
	return &VertexModelDelete{
		modelClient: modelClient,
	}
}

// Delete deletes a model from Vertex AI.
func (d *VertexModelDelete) Delete(ctx context.Context, modelName string) error {
	deleteReq := &aiplatformpb.DeleteModelRequest{
		// Model name is already in the format projects/{project}/locations/{location}/models/{model ID}
		Name: modelName,
	}

	deleteOperation, err := d.modelClient.DeleteModel(ctx, deleteReq)
	if err != nil {
		return fmt.Errorf("failed to delete model: %w", err)
	}

	if deleteOperation == nil {
		log.Printf("Warning: model delete operation is nil?!? This must be a mocked client. Logging error and moving on.")
	} else {
		err = deleteOperation.Wait(ctx)
		if err != nil {
			return fmt.Errorf("failed to wait for deletion: %w", err)
		}
	}

	return nil
}
