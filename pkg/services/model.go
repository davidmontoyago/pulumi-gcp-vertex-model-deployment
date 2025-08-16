// Package services provides implementations for GCP Vertex AI operations.
package services

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	gax "github.com/googleapis/gax-go/v2"
	"github.com/googleapis/gax-go/v2/apierror"
)

// ModelUpload represents the parameters needed to upload a model to Vertex AI.
type ModelUpload struct {
	Name                             string
	ModelImageURL                    string
	ModelArtifactsBucketURI          string
	ServiceAccountEmail              string
	ModelPredictionInputSchemaURI    string
	ModelPredictionOutputSchemaURI   string
	ModelPredictionBehaviorSchemaURI string
}

// ModelUploader interface defines operations for uploading models.
type ModelUploader interface {
	Upload(ctx context.Context, uploadParams ModelUpload) (string, error)
	Close() error
}

// ModelDeployer interface defines operations for deploying models.
type ModelDeployer interface {
	Deploy(ctx context.Context, endpointID, modelName, name, machineType, serviceAccount string, minReplicas, maxReplicas int32) (string, error)
	Close() error
}

// VertexModelUpload implements the ModelUploader interface for Vertex AI.
type VertexModelUpload struct {
	modelClient VertexModelClient
	projectID   string
	region      string
	labels      map[string]string
}

// VertexModelDeploy implements the ModelDeployer interface for Vertex AI.
type VertexModelDeploy struct {
	endpointClient VertexEndpointClient
	projectID      string
	region         string
}

// NewVertexModelUpload creates a new VertexModelUpload with the provided model client.
func NewVertexModelUpload(_ context.Context, modelClient VertexModelClient, projectID, region string, labels map[string]string) *VertexModelUpload {
	return &VertexModelUpload{
		modelClient: modelClient,
		projectID:   projectID,
		region:      region,
		labels:      labels,
	}
}

// Upload uploads a model to Vertex AI and returns the model name.
func (u *VertexModelUpload) Upload(ctx context.Context, params ModelUpload) (string, error) {

	predictionSchema := &aiplatformpb.PredictSchemata{
		// Schema for the model input
		InstanceSchemaUri: params.ModelPredictionInputSchemaURI,
		// Schema for the model output
		PredictionSchemaUri: params.ModelPredictionOutputSchemaURI,
	}
	if params.ModelPredictionBehaviorSchemaURI != "" {
		// Schema for the model inference behavior. Optional depending on the model.
		predictionSchema.ParametersSchemaUri = params.ModelPredictionBehaviorSchemaURI
	}

	modelUploadOp, err := u.modelClient.UploadModel(ctx, &aiplatformpb.UploadModelRequest{
		// TODO support non traditional / global models
		// Endpoint to which the model is attached. It can be regional or global, depending on the model type.
		Parent:         fmt.Sprintf("projects/%s/locations/%s", u.projectID, u.region),
		ServiceAccount: params.ServiceAccountEmail,
		Model: &aiplatformpb.Model{
			DisplayName: params.Name,
			Description: "Uploaded model for " + params.ModelImageURL,
			ContainerSpec: &aiplatformpb.ModelContainerSpec{
				ImageUri: params.ModelImageURL,
				// TODO make me configurable
				Args: []string{
					"--allow_precompilation=false",
					"--disable_optimizer=true",
					"--saved_model_tags='serve,tpu'",
					"--use_tfrt=true",
				},
			},
			Labels: u.labels,
			// May be optional for custom models but required for TensorFlow pre-built images.
			// See:
			// - https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts#framework-requirements
			ArtifactUri: params.ModelArtifactsBucketURI,
			// Paths to the model's prediction schemas
			PredictSchemata: predictionSchema,
		},
	}, gax.WithTimeout(5*time.Minute))
	if err != nil {
		var apiError *apierror.APIError
		if errors.As(err, &apiError) {
			// TODO DRY up
			log.Printf("Model upload returned APIError details: %v\n", apiError)
			log.Printf("APIError reason: %v\n", apiError.Reason())
			log.Printf("APIError details : %v\n", apiError.Details())
			// If a gRPC transport was used you can extract the
			// google.golang.org/grpc/status.Status from the error
			log.Printf("APIError GRPCStatus: %+v\n", apiError.GRPCStatus())
			log.Printf("APIError HTTPCode: %+v\n", apiError.HTTPCode())
		}

		return "", fmt.Errorf("failed to upload model: %w", err)
	}

	if modelUploadOp == nil {
		log.Printf("Warning: model upload operation is nil?!? This must be a mocked client. Logging error and moving on.")

		return "", nil
	}

	// TODO make timeout configurable
	modelUploadResult, err := modelUploadOp.Wait(context.Background(), gax.WithTimeout(10*time.Minute))
	if err != nil {
		if modelUploadOp.Done() {
			log.Printf("Model upload operation completed with failure: %v\n", err)
		}
		var apiError *apierror.APIError
		if errors.As(err, &apiError) {
			// TODO DRY up
			log.Printf("Model upload returned APIError details: %v\n", apiError)
			log.Printf("APIError reason: %v\n", apiError.Reason())
			log.Printf("APIError details : %v\n", apiError.Details())
			log.Printf("APIError help: %v\n", apiError.Details().Help)
			// If a gRPC transport was used you can extract the
			// google.golang.org/grpc/status.Status from the error
			log.Printf("APIError GRPCStatus: %v\n", apiError.GRPCStatus())
			log.Printf("APIError HTTPCode: %v\n", apiError.HTTPCode())
		}

		return "", fmt.Errorf("failed to wait for model upload: %w", err)
	}

	return modelUploadResult.GetModel(), nil
}

// Close closes the model client.
func (u *VertexModelUpload) Close() error {
	if u.modelClient != nil {
		if err := u.modelClient.Close(); err != nil {
			return fmt.Errorf("failed to close model client: %w", err)
		}
	}

	return nil
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
