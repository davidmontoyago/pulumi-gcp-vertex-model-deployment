# pulumi-gcp-vertex-model-deployment

[![Develop](https://github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/actions/workflows/develop.yaml/badge.svg)](https://github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/actions/workflows/develop.yaml) [![Go Coverage](https://raw.githubusercontent.com/wiki/davidmontoyago/pulumi-gcp-vertex-model-deployment/coverage.svg)](https://raw.githack.com/wiki/davidmontoyago/pulumi-gcp-vertex-model-deployment/coverage.html) [![Go Reference](https://pkg.go.dev/badge/github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment.svg)](https://pkg.go.dev/github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment)

The model registry is the integration point for all Vertex ML capabilities. This Pulumi custom provider allows deploying models to the Vertex model registry.

### Deploy model for batched prediction jobs
Upload model with a docker image to the model registry:
```go
modelArtifactsURI := pulumi.Sprintf("gs://%s/models/my-model", myArtifactsBucket.Name)
modelImageURL := "us-central1-docker.pkg.dev/my-project/my-docker-registry/pytorch-cpu.1-12-bert-with-cpr@latest"

modelDeployment, err := vertexmodeldeployment.NewVertexModelDeployment(ctx, "model-for-batch-prediction",
  &vertexmodeldeployment.VertexModelDeploymentArgs{
    ProjectId:                      pulumi.String("my-project"),
    Region:                         pulumi.String("us-central1"),
    ModelArtifactsBucketUri:        modelArtifactsURI,
    ModelImageUrl:                  modelImageURL,
    ModelPredictionInputSchemaUri:  pulumi.Sprintf("%s/%s", modelArtifactsURI, v.ModelPredictionInputSchemaPath),
    ModelPredictionOutputSchemaUri: pulumi.Sprintf("%s/%s", modelArtifactsURI, v.ModelPredictionOutputSchemaPath),
    ServiceAccount:                 modelServiceAccount.Email,
    PredictRoute:                   pulumi.String("/predict"),
    HealthRoute:                    pulumi.String("/health"),
  },
  pulumi.Parent(v), pulumi.DependsOn(dependencies),
)
if err != nil {
  return fmt.Errorf("failed to deploy model /o\\: %w", err)
}
```
Returned model name should be set on the batch prediction job.

### Deploy model to an endpoint

Upload and deploy model with a docker image to a given Vertex endpoint:
```go
modelArtifactsURI := pulumi.Sprintf("gs://%s/models/my-model", myArtifactsBucket.Name)
modelImageURL := "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"

inputSchemaURI := pulumi.Sprintf("gs://%s/schemas/input_schema.yaml", myArtifactsBucket.Name)
outputSchemaURI := pulumi.Sprintf("gs://%s/schemas/output_schema.yaml", myArtifactsBucket.Name)

modelDeployment, err := vertexmodeldeployment.NewVertexModelDeployment(ctx, "model-for-endpoint", &vertexmodeldeployment.VertexModelDeploymentArgs{
  ProjectId:                       pulumi.String("my-project"),
  Region:                          pulumi.String("us-central1"),
  ModelArtifactsBucketUri:         modelArtifactsURI,
  ModelImageUrl:                   pulumi.String(modelImageURL),
  ModelPredictionInputSchemaUri:   inputSchemaURI,
  ModelPredictionOutputSchemaUri:  outputSchemaURI,
  ServiceAccount:                  modelServiceAccount.Email,

  // Optional deployment to an Endpoint. Not required for batched prediction jobs
  EndpointModelDeployment: &vertexmodeldeployment.EndpointModelDeploymentArgs{
    EndpointId:     myendpoint.Name,
    MachineType:    pulumi.String("g2-standard-8"),
    MinReplicas:    pulumi.Int(1),
    MaxReplicas:    pulumi.Int(3),
    TrafficPercent: pulumi.Int(100),
  },
}, pulumi.Parent(v))
if err != nil {
  return fmt.Errorf("failed to deploy model /o\\: %w", err)
}
```

See:
- https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
- https://cloud.google.com/vertex-ai/docs/general/deployment#prepare_to_deploy_a_model_to_an_endpoint
- https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.models/upload
- https://github.com/hashicorp/terraform-provider-google/issues/15303

## Getting Started

Install for local dev:
```sh
PROVIDER_VERSION=0.0.0 GOOS=darwin GOARCH=arm64 make plugin-local
```

Get the SDK:
```sh
go get github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/sdk/go
```
