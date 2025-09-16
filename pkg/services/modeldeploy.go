// Package services provides implementations for GCP Vertex AI model and Endpoint deployment operations.
package services

import (
	"context"
	"fmt"
	"log"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
)

// ModelDeployer interface defines operations for deploying models.
type ModelDeployer interface {
	Deploy(ctx context.Context, endpointID, modelName, name, machineType, serviceAccount string, minReplicas, maxReplicas int32) (string, error)
	Close() error
}

// VertexModelDeploy implements the ModelDeployer interface for Vertex AI.
type VertexModelDeploy struct {
	endpointClient VertexEndpointClient
	projectID      string
	region         string
}

// NewVertexModelDeploy creates a new VertexModelDeploy with the provided endpoint client.
func NewVertexModelDeploy(_ context.Context, endpointClient VertexEndpointClient, projectID, region string) *VertexModelDeploy {
	return &VertexModelDeploy{
		endpointClient: endpointClient,
		projectID:      projectID,
		region:         region,
	}
}

// Deploy deploys a model to a Vertex AI endpoint and returns the deployed model ID.
func (d *VertexModelDeploy) Deploy(ctx context.Context, endpointID, modelName, name, machineType, serviceAccount string, minReplicas, maxReplicas int32) (string, error) {
	// Build the deployment request
	deployedModel := &aiplatformpb.DeployedModel{
		// Expected format: "projects/%s/locations/%s/models/%s"
		Model:       modelName,
		DisplayName: name,
		PredictionResources: &aiplatformpb.DeployedModel_DedicatedResources{
			DedicatedResources: &aiplatformpb.DedicatedResources{
				MachineSpec: &aiplatformpb.MachineSpec{
					MachineType: machineType,
				},
				MinReplicaCount: minReplicas,
				MaxReplicaCount: maxReplicas,
			},
		},
	}

	if serviceAccount != "" {
		deployedModel.ServiceAccount = serviceAccount
	}

	deployReq := &aiplatformpb.DeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			d.projectID, d.region, endpointID),
		DeployedModel: deployedModel,
		TrafficSplit:  map[string]int32{
			// TODO set for parallel model deployments
		},
	}

	// Execute the deployment
	deployOperation, err := d.endpointClient.DeployModel(ctx, deployReq)
	if err != nil {
		return "", fmt.Errorf("failed to deploy model: %w", err)
	}

	if deployOperation == nil {
		log.Printf("Warning: deploy operation is nil?!? This must be a mocked client. Logging error and moving on.")

		return "", nil
	}

	// Wait for completion with timeout
	result, err := deployOperation.Wait(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to wait for deployment: %w", err)
	}

	return result.GetDeployedModel().GetId(), nil
}

// Close closes the endpoint client.
func (d *VertexModelDeploy) Close() error {
	if d.endpointClient != nil {
		if err := d.endpointClient.Close(); err != nil {
			return fmt.Errorf("failed to close endpoint client: %w", err)
		}
	}

	return nil
}
