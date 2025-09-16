# pulumi-gcp-vertex-model-deployment

[![Develop](https://github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/actions/workflows/develop.yaml/badge.svg)](https://github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/actions/workflows/develop.yaml) [![Go Coverage](https://raw.githubusercontent.com/wiki/davidmontoyago/pulumi-gcp-vertex-model-deployment/coverage.svg)](https://raw.githack.com/wiki/davidmontoyago/pulumi-gcp-vertex-model-deployment/coverage.html) [![Go Reference](https://pkg.go.dev/badge/github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment.svg)](https://pkg.go.dev/github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment)


Pulumi custom provider to upload a model to Vertex and optionally deploy it to a Vertex Endpoint.

Upload and deploy a model from a Docker image to a Vertex endpoint:
```go
modelArtifactsURI := pulumi.Sprintf("gs://%s/models/my-model", myArtifactsBucket.Name)
modelImageURL := "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"

_, err = vertexmodeldeployment.NewVertexModelDeployment(ctx, "vertex-model-deployment", &vertexmodeldeployment.VertexModelDeploymentArgs{
  ProjectId:               pulumi.String(v.Project),
  Region:                  pulumi.String(v.Region),
  EndpointId:              endpoint.Name,
  ModelArtifactsBucketUri: modelArtifactsURI,
  ModelImageUrl:           modelImageURL,
  MachineType:             v.MachineType,
  MinReplicas:             v.MinReplicaCount,
  MaxReplicas:             v.MaxReplicaCount,
  ServiceAccount:          modelServiceAccount.Email,
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
