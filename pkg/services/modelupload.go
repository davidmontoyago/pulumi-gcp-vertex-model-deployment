// Package services provides implementations for GCP Vertex AI model upload operations.
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
	PredictRoute                     string
	HealthRoute                      string
	Args                             []string
	EnvVars                          map[string]string
	Port                             int32
}

// ModelUploader interface defines operations for uploading models.
type ModelUploader interface {
	Upload(ctx context.Context, uploadParams ModelUpload) (string, error)
	Close() error
}

// VertexModelUpload implements the ModelUploader interface for Vertex AI.
type VertexModelUpload struct {
	modelClient VertexModelClient
	projectID   string
	region      string
	labels      map[string]string
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
	envVars := []*aiplatformpb.EnvVar{}
	for name, value := range params.EnvVars {
		envVars = append(envVars, &aiplatformpb.EnvVar{
			Name:  name,
			Value: value,
		})
	}
	modelServerPort := params.Port
	if modelServerPort == 0 {
		modelServerPort = 8080
	}

	modelArgs := &aiplatformpb.Model{
		DisplayName: params.Name,
		Description: "Uploaded model for " + params.ModelImageURL,
		ContainerSpec: &aiplatformpb.ModelContainerSpec{
			ImageUri: params.ModelImageURL,
			Args:     params.Args,
			Env:      envVars,
			Ports: []*aiplatformpb.Port{
				{
					ContainerPort: modelServerPort,
				},
			},
		},
		Labels:      u.labels,
		ArtifactUri: params.ModelArtifactsBucketURI,
	}

	if params.ModelPredictionInputSchemaURI != "" {
		modelArgs.PredictSchemata = &aiplatformpb.PredictSchemata{
			// Schema for the model input
			InstanceSchemaUri: params.ModelPredictionInputSchemaURI,
			// Schema for the model output
			PredictionSchemaUri: params.ModelPredictionOutputSchemaURI,
		}
		if params.ModelPredictionBehaviorSchemaURI != "" {
			// Schema for the model inference behavior. Optional depending on the model.
			modelArgs.PredictSchemata.ParametersSchemaUri = params.ModelPredictionBehaviorSchemaURI
		}
	}

	if params.PredictRoute != "" {
		modelArgs.ContainerSpec.PredictRoute = params.PredictRoute
	}
	if params.HealthRoute != "" {
		modelArgs.ContainerSpec.HealthRoute = params.HealthRoute
	}

	modelUploadOp, err := u.modelClient.UploadModel(ctx, &aiplatformpb.UploadModelRequest{
		// TODO support non traditional / global models
		// GCP endpoint to which the model is attached. It can be regional or global, depending on the model type.
		Parent:         fmt.Sprintf("projects/%s/locations/%s", u.projectID, u.region),
		ServiceAccount: params.ServiceAccountEmail,
		Model:          modelArgs,
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

		return "MOCKED_MODEL_NAME", nil
	}

	modelUploadResult, err := modelUploadOp.Wait(ctx, gax.WithTimeout(10*time.Minute))
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
