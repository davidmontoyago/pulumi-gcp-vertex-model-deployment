# pulumi-gcp-vertex-model-deployment

Pulumi custom provider for Vertex models.

Upload and deploy a model from a Docker image to a Vertex endpoint:
```go
modelArtifactsURI := pulumi.Sprintf("gs://%s/models/my-model", myArtifactsBucket.Name)
modelImageURL := "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"

_, err = vertexmodeldeployment.NewVertexModelDeployment(ctx, v.newResourceName("vertex-model-deployment", "", 64), &vertexmodeldeployment.VertexModelDeploymentArgs{
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
```
PROVIDER_VERSION=0.0.0 GOOS=darwin GOARCH=arm64 make plugin-local
```
