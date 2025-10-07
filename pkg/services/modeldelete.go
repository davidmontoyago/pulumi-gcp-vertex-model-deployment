package services

import (
	"context"
	"fmt"
	"log"
	"strings"

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
	projectID   string
	region      string
}

// NewVertexModelDelete creates a new VertexModelDelete with the provided model client.
func NewVertexModelDelete(_ context.Context, modelClient VertexModelClient, projectID, region string) *VertexModelDelete {
	return &VertexModelDelete{
		modelClient: modelClient,
		projectID:   projectID,
		region:      region,
	}
}

// Delete deletes a model from Vertex AI.
func (d *VertexModelDelete) Delete(ctx context.Context, modelName string) error {
	modelFullName := modelName
	if !strings.HasPrefix(modelName, "projects/") {
		modelFullName = fmt.Sprintf("projects/%s/locations/%s/models/%s",
			d.projectID, d.region, modelName)
	}
	deleteReq := &aiplatformpb.DeleteModelRequest{
		Name: modelFullName,
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
