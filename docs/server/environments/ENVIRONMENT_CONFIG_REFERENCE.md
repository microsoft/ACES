# SABER Environment Configuration Reference

This document provides comprehensive documentation for configuring Docker environments in SABER using the `environments.yaml` format. SABER supports both ephemeral sandbox environments (created per episode) and permanent environments (running for server lifetime).

## Overview

SABER environments are defined in `environments.yaml` files located in your domain's server configuration directory. The environment system supports:

- **Container Templates**: Reusable container specifications
- **Environment Compositions**: Complete environment setups combining containers and networks
- **Network Templates**: Network configurations for container communication
- **Resource Management**: Memory, CPU, and execution limits
- **Permanent vs. Ephemeral**: Different lifecycle patterns for different use cases

## Configuration Structure

```yaml
# Container Templates - reusable container specifications
containers:
  container_name:
    image: "image:tag"
    container_name: "actual_container_name"
    # ... container configuration

# Environment Compositions - complete environment setups
environments:
  environment_name:
    # Environment type and networking
    network: "network_name"
    permanent: true|false

    # Service composition (for permanent environments)
    services:
      - name: "service_name"
        container: "container_template_name"

    # Direct execution container (for sandbox environments)
    execution: "container_template_name"

    # Resource limits and configuration
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.5"
      execution_timeout: 900

# Network Templates - network configurations
networks:
  network_name:
    driver: "bridge"
    external: true|false
    name: "actual_network_name"
    # ... network configuration
```

## Container Templates

Container templates define reusable container specifications that can be referenced by environments.

### Basic Container Template

```yaml
containers:
  my-app-container:
    image: "my-app:latest"
    container_name: "my-app-instance"
    ports:
      - "8080:8080"
    environment:
      - "APP_ENV=production"
      - "DATABASE_URL=mysql://root:password@db:3306/app"
    volumes:
      - "/host/data:/container/data"
    working_dir: "/app"
    command: ["python", "app.py"]
```

### Container Configuration Options

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `image` | string | Docker image name and tag | Yes |
| `container_name` | string | Name for the container instance | Yes |
| `ports` | array | Port mappings in "host:container" format | No |
| `environment` | array | Environment variables as "KEY=value" | No |
| `volumes` | array | Volume mounts in "host:container" format | No |
| `working_dir` | string | Working directory inside container | No |
| `command` | array/string | Command to run in container | No |
| `user` | string | User to run container as | No |
| `privileged` | boolean | Run container in privileged mode | No |
| `cap_add` | array | Linux capabilities to add | No |
| `cap_drop` | array | Linux capabilities to drop | No |

### Security-Focused Container Example

```yaml
containers:
  secure-sandbox:
    image: "saber-python-sandbox:latest"
    container_name: "python-sandbox"
    working_dir: "/workspace"
    user: "tooluser:tooluser"
    read_only: true
    cap_drop:
      - "ALL"
    cap_add:
      - "DAC_OVERRIDE"  # Only if needed for file operations
    environment:
      - "PYTHONPATH=/workspace"
      - "PYTHONUNBUFFERED=1"
    tmpfs:
      - "/tmp:rw,noexec,nosuid,size=100m"
```

## Environment Compositions

Environments combine containers, networks, and configurations into complete execution environments.

### Permanent Environment

Permanent environments run for the server lifetime and provide persistent services.

```yaml
environments:
  database_environment:
    network: "app-network"
    permanent: true
    services:
      - name: "mysql-db"
        container: "mysql-container"
      - name: "redis-cache"
        container: "redis-container"
    resource_limits:
      total_memory: "2g"
      total_cpu: "1.0"
      execution_timeout: null  # No timeout for permanent environments
```

### Sandbox Environment

Sandbox environments are created per episode for isolated execution.

```yaml
environments:
  python_sandbox:
    network: "shared-network"
    permanent_environment_connectivity: true  # Connect to permanent networks
    execution: "python-sandbox-container"
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.5"
      execution_timeout: 900  # 15 minutes
```

### Environment Configuration Options

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `network` | string | Network template to use | Yes |
| `permanent` | boolean | Whether this is a permanent environment | No (default: false) |
| `services` | array | Service definitions (permanent environments) | No |
| `execution` | string | Container template for execution (sandbox environments) | No |
| `permanent_environment_connectivity` | boolean | Auto-connect to permanent networks | No |
| `resource_limits` | object | Resource constraints and limits | No |

### Resource Limits Configuration

```yaml
resource_limits:
  total_memory: "1g"           # Total memory limit (Docker format)
  total_cpu: "0.5"             # CPU limit (fraction of core)
  execution_timeout: 900       # Execution timeout in seconds (null for no limit)
  max_processes: 50            # Maximum number of processes
  max_open_files: 1000         # Maximum open file descriptors
```

## Network Templates

Network templates define Docker networks for container communication.

### Basic Network

```yaml
networks:
  app-network:
    driver: bridge
    name: "saber-app-network"
```

### External Network

```yaml
networks:
  existing-network:
    driver: bridge
    external: true
    name: "pre-existing-network"
```

### Network with Custom Configuration

```yaml
networks:
  custom-network:
    driver: bridge
    name: "custom-saber-network"
    ipam:
      driver: default
      config:
        - subnet: "172.20.0.0/16"
          gateway: "172.20.0.1"
    driver_opts:
      com.docker.network.bridge.name: "saber-br0"
      com.docker.network.driver.mtu: "1500"
```

### Network Configuration Options

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `driver` | string | Network driver (bridge, overlay, etc.) | Yes |
| `name` | string | Actual Docker network name | Yes |
| `external` | boolean | Whether network already exists | No |
| `ipam` | object | IP Address Management configuration | No |
| `driver_opts` | object | Driver-specific options | No |

## Complete Example Configurations

### Web Application Penetration Testing

```yaml
# Permanent database + ephemeral browser sandbox
containers:
  vulnerable-webapp:
    image: "dvwa:latest"
    container_name: "vulnerable-webapp"
    ports:
      - "80:80"
    environment:
      - "MYSQL_HOST=webapp-db"
      - "MYSQL_DATABASE=dvwa"
      - "MYSQL_USER=dvwa"
      - "MYSQL_PASSWORD=password"

  webapp-database:
    image: "mysql:8.0"
    container_name: "webapp-db"
    environment:
      - "MYSQL_ROOT_PASSWORD=admin"
      - "MYSQL_DATABASE=dvwa"
      - "MYSQL_USER=dvwa"
      - "MYSQL_PASSWORD=password"
    command: "--secure-file-priv=/var/lib/mysql-files"

  browser-sandbox:
    image: "saber-browser-sandbox:latest"
    container_name: "browser-sandbox"
    working_dir: "/workspace"
    environment:
      - "DISPLAY=:99"
    volumes:
      - "/tmp/.X11-unix:/tmp/.X11-unix"

environments:
  # Permanent web application environment
  vulnerable_webapp:
    network: "pentest-network"
    permanent: true
    services:
      - name: "webapp"
        container: "vulnerable-webapp"
      - name: "database"
        container: "webapp-database"
    resource_limits:
      total_memory: "1g"
      total_cpu: "0.8"
      execution_timeout: null

  # Ephemeral browser sandbox
  browser_sandbox:
    network: "pentest-network"
    permanent_environment_connectivity: true
    execution: "browser-sandbox"
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.5"
      execution_timeout: 1800  # 30 minutes

networks:
  pentest-network:
    driver: bridge
    external: true
    name: "saber-pentest-network"
```

### Malware Analysis Environment

```yaml
containers:
  analysis-sandbox:
    image: "saber-malware-sandbox:latest"
    container_name: "malware-sandbox"
    working_dir: "/samples"
    user: "analyst:analyst"
    cap_drop:
      - "ALL"
    cap_add:
      - "SYS_PTRACE"  # For debugging tools
    environment:
      - "ANALYST_HOME=/home/analyst"
    tmpfs:
      - "/tmp:rw,noexec,nosuid,size=200m"
      - "/var/tmp:rw,noexec,nosuid,size=100m"

  honeypot-service:
    image: "honeypot:latest"
    container_name: "honeypot"
    ports:
      - "21:21"    # FTP
      - "22:22"    # SSH
      - "80:80"    # HTTP
      - "443:443"  # HTTPS
    environment:
      - "HONEYPOT_MODE=high_interaction"

environments:
  # Permanent honeypot for network interaction
  honeypot_network:
    network: "malware-network"
    permanent: true
    services:
      - name: "honeypot"
        container: "honeypot-service"
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.3"
      execution_timeout: null

  # Isolated malware analysis sandbox
  malware_sandbox:
    network: "malware-network"
    permanent_environment_connectivity: true
    execution: "analysis-sandbox"
    resource_limits:
      total_memory: "2g"
      total_cpu: "1.0"
      execution_timeout: 3600  # 1 hour

networks:
  malware-network:
    driver: bridge
    name: "saber-malware-network"
    ipam:
      driver: default
      config:
        - subnet: "172.25.0.0/16"
          gateway: "172.25.0.1"
```

## Best Practices

### Security Guidelines

1. **Principle of Least Privilege**
   ```yaml
   containers:
     secure-container:
       user: "nonroot:nonroot"
       read_only: true
       cap_drop: ["ALL"]
       cap_add: ["DAC_OVERRIDE"]  # Only specific caps needed
   ```

2. **Resource Limits**
   ```yaml
   resource_limits:
     total_memory: "512m"    # Prevent memory exhaustion
     total_cpu: "0.5"        # Limit CPU usage
     execution_timeout: 900  # Prevent runaway processes
   ```

3. **Network Isolation**
   ```yaml
   networks:
     isolated-network:
       driver: bridge
       internal: true  # No external connectivity
   ```

### Performance Optimization

1. **Container Image Best Practices**
   - Use specific image tags, not `latest`
   - Prefer lightweight base images (Alpine, distroless)
   - Pre-install common tools in custom images

2. **Resource Allocation**
   - Size memory limits based on actual usage patterns
   - Use CPU limits to prevent container monopolization
   - Set reasonable execution timeouts

3. **Network Configuration**
   - Reuse networks across related environments
   - Use external networks for persistent connectivity
   - Configure appropriate subnet sizes

### Environment Lifecycle

1. **Permanent Environments**
   - Use for databases, persistent services, shared resources
   - Start with server initialization
   - Persist across multiple episodes
   - Require explicit shutdown

2. **Sandbox Environments**
   - Use for isolated execution contexts
   - Created per episode or session
   - Automatically cleaned up after use
   - Should be stateless and reproducible

## Troubleshooting

### Common Issues

1. **Container Startup Failures**
   - Verify image availability: `docker images`
   - Check resource limits are reasonable
   - Ensure required environment variables are set

2. **Network Connectivity Issues**
   - Verify network exists: `docker network ls`
   - Check if external networks are created
   - Ensure containers are on same network

3. **Resource Limit Errors**
   - Monitor container resource usage
   - Adjust memory/CPU limits based on actual needs
   - Check host system resources

### Debug Commands

```bash
# List all SABER networks
docker network ls | grep saber

# Inspect container configuration
docker inspect <container_name>

# Check container logs
docker logs <container_name>

# Monitor resource usage
docker stats <container_name>
```

## Integration with SABER Components

### Task Configuration

Reference environments in your `tasks.yaml`:

```yaml
tasks:
  - task_id: "web_pentest"
    sandbox_environment: "browser_sandbox"    # Ephemeral environment
    # ... other task configuration

# Domain-level permanent environment
permanent_environment: "vulnerable_webapp"
```

### Environment Resolution

SABER resolves environment references through the EnvironmentLoader:

1. Parse `environments.yaml` configuration
2. Resolve container and network templates
3. Create environment specifications
4. Validate resource limits and constraints
5. Pass to appropriate environment manager (sandbox or permanent)

This environment configuration system provides the foundation for SABER's flexible and secure execution environments, supporting both simple isolated sandboxes and complex multi-service permanent environments.
